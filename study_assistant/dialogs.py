"""Dialogs: PDF import, quizzes, history, agent editor, and local model settings."""

from __future__ import annotations

import sys
import threading
from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QKeySequenceEdit,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .macos import KEY_CODES, Shortcut, shortcut_label
from .hardware import detect_hardware, download_choices, fetch_models_by_role, is_installed, models_by_role
from .providers import (
    GEMINI,
    KEY_URL,
    OLLAMA,
    PROVIDER_LABELS,
    best_model,
    get_api_key,
    make_client,
    model_fits,
)
from .gemini import check_models, health_label, mask_key, recommend
from .ollama import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_HOST,
    NOTES_MODES,
    OllamaClient,
    OllamaError,
)
from .theme import ACCENTS, COLORS, creativity_label
from .widgets import AgentAvatar


def _group_lectures(documents: list[dict]) -> list[dict]:
    """Combine multiple PDF files assigned to the same class/lecture space."""
    grouped: dict[tuple[str, str], dict] = {}
    for document in documents:
        key = (document["class_name"], document["lecture_name"])
        lecture = grouped.setdefault(key, {"class_name": key[0], "lecture_name": key[1], "page_count": 0})
        lecture["page_count"] += document["page_count"]
    return sorted(
        grouped.values(),
        key=lambda item: (item["class_name"].casefold(), item["lecture_name"].casefold()),
    )



class ImportDetailsDialog(QDialog):
    def __init__(self, filename: str, parent: QWidget | None = None, default_class_name: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Add lecture PDF")
        self.class_name = QLineEdit()
        self.class_name.setText(default_class_name)
        self.class_name.setPlaceholderText("e.g. CS 354")
        self.lecture_name = QLineEdit(Path(filename).stem)
        self.lecture_name.setPlaceholderText("e.g. Scheduling")
        form = QFormLayout()
        form.addRow("Class", self.class_name)
        form.addRow("Lecture", self.lecture_name)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)


class BatchImportDetailsDialog(QDialog):
    """Collect per-PDF class and lecture labels before importing a selection."""

    def __init__(
        self, filenames: list[str], parent: QWidget | None = None, default_class_name: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Add {len(filenames)} lecture PDFs")
        self.resize(760, min(620, 220 + 44 * len(filenames)))
        self.setMinimumSize(620, 280)
        intro = QLabel("Choose the class and lecture name for each PDF. Names default to each filename.")
        intro.setObjectName("dialogHelp")
        intro.setWordWrap(True)
        self.table = QTableWidget(len(filenames), 3)
        self.table.setHorizontalHeaderLabels(["PDF file", "Class", "Lecture"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setColumnWidth(0, 270)
        self.table.setColumnWidth(1, 150)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.file_paths = [Path(filename) for filename in filenames]
        for row, path in enumerate(self.file_paths):
            file_item = QTableWidgetItem(path.name)
            file_item.setToolTip(str(path))
            file_item.setFlags(file_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, file_item)
            self.table.setItem(row, 1, QTableWidgetItem(default_class_name))
            self.table.setItem(row, 2, QTableWidgetItem(path.stem))
        self.table.resizeRowsToContents()
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Import PDFs")
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(self.table, 1)
        layout.addWidget(buttons)
        self.table.setFocus()
        if filenames:
            self.table.setCurrentCell(0, 1)

    def import_specs(self) -> list[tuple[Path, str, str]]:
        return [
            (path, self.table.item(row, 1).text().strip(), self.table.item(row, 2).text().strip())
            for row, path in enumerate(self.file_paths)
        ]

    def _accept_if_valid(self) -> None:
        for row, (path, class_name, lecture_name) in enumerate(self.import_specs(), start=1):
            if not class_name or not lecture_name:
                QMessageBox.warning(
                    self,
                    "Missing class or lecture",
                    f"Enter both a class and lecture name for {path.name} (row {row}).",
                )
                self.table.setCurrentCell(row - 1, 1 if not class_name else 2)
                return
        self.accept()



class QuizSetupDialog(QDialog):
    def __init__(self, documents: list[dict], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Build a practice quiz")
        self.resize(620, 540)
        self.documents = _group_lectures(documents)
        self.class_filter = QComboBox()
        self.class_filter.addItem("All classes", None)
        for class_name in sorted({doc["class_name"] for doc in documents}, key=str.casefold):
            self.class_filter.addItem(class_name, class_name)
        self.class_filter.currentIndexChanged.connect(self.refresh_lectures)
        self.lecture_list = QListWidget()
        self.question_count = QSpinBox()
        self.question_count.setRange(1, 10)
        self.question_count.setValue(5)
        self.question_type = QComboBox()
        self.question_type.addItem("Mixed short answer + multiple choice", "mixed")
        self.question_type.addItem("Short answer", "short_answer")
        self.question_type.addItem("Multiple choice", "multiple_choice")
        form = QFormLayout()
        form.addRow("Class", self.class_filter)
        form.addRow("Question count", self.question_count)
        form.addRow("Question style", self.question_type)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select the lectures to include. Questions will use fresh examples and scenarios, with citations to their source pages."))
        layout.addLayout(form)
        layout.addWidget(QLabel("Lectures"))
        layout.addWidget(self.lecture_list, 1)
        layout.addWidget(buttons)
        self.refresh_lectures()

    @property
    def scopes(self) -> list[tuple[str, str]]:
        selected = []
        for index in range(self.lecture_list.count()):
            item = self.lecture_list.item(index)
            if item.checkState() == Qt.CheckState.Checked:
                selected.append(tuple(item.data(Qt.ItemDataRole.UserRole)))
        return selected

    def refresh_lectures(self) -> None:
        self.lecture_list.clear()
        class_name = self.class_filter.currentData()
        for doc in self.documents:
            if class_name is not None and doc["class_name"] != class_name:
                continue
            item = QListWidgetItem(f"{doc['class_name']} · {doc['lecture_name']} ({doc['page_count']} pages)")
            item.setData(Qt.ItemDataRole.UserRole, (doc["class_name"], doc["lecture_name"]))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.lecture_list.addItem(item)

    def _accept_if_valid(self) -> None:
        if not self.scopes:
            QMessageBox.information(self, "Choose lectures", "Select at least one lecture to quiz.")
            return
        self.accept()


class QuizRunDialog(QDialog):
    answer_submitted = Signal(object, str)

    def __init__(self, questions: list[dict], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.questions = questions
        self.question_index = 0
        self.grading = False
        self.setWindowTitle("Practice quiz")
        self.resize(760, 680)
        self.progress_label = QLabel()
        self.prompt_label = QLabel()
        self.prompt_label.setWordWrap(True)
        self.sources_label = QLabel()
        self.sources_label.setWordWrap(True)
        self.input_layout = QVBoxLayout()
        self.input_host = QWidget()
        self.input_host.setLayout(self.input_layout)
        self.feedback = QLabel()
        self.feedback.setWordWrap(True)
        self.feedback.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.submit_button = QPushButton("Check answer")
        self.submit_button.clicked.connect(self.submit_answer)
        self.next_button = QPushButton("Next question")
        self.next_button.clicked.connect(self.next_question)
        self.next_button.hide()
        self.cancel_button = QPushButton("Close quiz")
        self.cancel_button.clicked.connect(self.reject)
        controls = QHBoxLayout()
        controls.addWidget(self.submit_button)
        controls.addWidget(self.next_button)
        controls.addStretch(1)
        controls.addWidget(self.cancel_button)
        layout = QVBoxLayout(self)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.prompt_label)
        layout.addWidget(QLabel("Source pages:"))
        layout.addWidget(self.sources_label)
        layout.addWidget(self.input_host, 1)
        layout.addWidget(self.feedback)
        layout.addLayout(controls)
        self._load_question()

    def _load_question(self) -> None:
        self.question = self.questions[self.question_index]
        self.progress_label.setText(f"Question {self.question_index + 1} of {len(self.questions)} · {self.question['question_type'].replace('_', ' ').title()}")
        self.prompt_label.setText(self.question["prompt"])
        labels = []
        for source_id in self.question["source_ids"]:
            source = self.question["source_map"][source_id]
            labels.append(f"{source['class_name']} · {source['lecture_name']} · {Path(source['file_path']).name} · p. {source['page_number']}")
        self.sources_label.setText("\n".join(labels))
        while self.input_layout.count():
            child = self.input_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.answer_input = None
        self.answer_group = None
        if self.question["question_type"] == "multiple_choice":
            self.answer_group = QButtonGroup(self)
            for index, option in enumerate(self.question["options"]):
                button = QRadioButton(f"{chr(65 + index)}. {option}")
                self.answer_group.addButton(button, index)
                self.input_layout.addWidget(button)
        else:
            self.answer_input = QPlainTextEdit()
            self.answer_input.setPlaceholderText("Explain your reasoning in your own words…")
            self.input_layout.addWidget(self.answer_input)
        self.feedback.clear()
        self.submit_button.setEnabled(True)
        self.submit_button.show()
        self.next_button.hide()

    def submit_answer(self) -> None:
        if self.answer_group:
            choice = self.answer_group.checkedId()
            if choice < 0:
                QMessageBox.information(self, "Choose an option", "Select an answer first.")
                return
            answer = chr(65 + choice)
        else:
            answer = self.answer_input.toPlainText().strip()
            if not answer:
                QMessageBox.information(self, "Write an answer", "Write a short answer first.")
                return
        self.submit_button.setEnabled(False)
        self.answer_submitted.emit(self.question, answer)

    def show_grade(self, result: dict) -> None:
        self.grading = False
        self.cancel_button.setEnabled(True)
        correct_label = "Correct" if result["is_correct"] else "Needs review"
        expected = self.question["correct_answer"]
        if self.question["question_type"] == "multiple_choice" and expected in {"A", "B", "C", "D"}:
            option_index = ord(expected) - ord("A")
            expected = f"{expected}. {self.question['options'][option_index]}"
        self.feedback.setText(
            f"{correct_label} · {result['score']:.0%}\n\n{result['feedback']}\n\n"
            f"Expected answer: {expected}\n\n"
            f"Why: {self.question['explanation']} {result['explanation_citations']}"
        )
        self.submit_button.hide()
        self.next_button.setText("Finish quiz" if self.question_index + 1 == len(self.questions) else "Next question")
        self.next_button.show()

    def set_grading(self, grading: bool) -> None:
        self.grading = grading
        self.cancel_button.setEnabled(not grading)

    def grading_failed(self, error: str = "") -> None:
        self.set_grading(False)
        self.submit_button.setEnabled(True)
        detail = f"\n\n{error}" if error else " Check the error message and try grading it again."
        self.feedback.setText("Your answer was not recorded. Grading failed." + detail)

    def reject(self) -> None:
        if not self.grading:
            super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt method name
        if self.grading:
            event.ignore()
        else:
            super().closeEvent(event)

    def next_question(self) -> None:
        self.question_index += 1
        if self.question_index >= len(self.questions):
            self.accept()
            return
        self._load_question()


class WeakSpotsDialog(QDialog):
    review_requested = Signal(str)

    def __init__(self, performance: list[dict], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.performance = performance
        self.setWindowTitle("Topics to review")
        self.resize(620, 500)
        self.topics = QListWidget()
        self.topics.currentItemChanged.connect(self.show_topic)
        self.detail = QLabel("Repeated misses rise to the top. Later successful attempts reduce a topic's weakness score.")
        self.detail.setWordWrap(True)
        self.review_button = QPushButton("Create a targeted review quiz")
        self.review_button.clicked.connect(self.review_topic)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        buttons = QHBoxLayout()
        buttons.addWidget(self.review_button)
        buttons.addStretch(1)
        buttons.addWidget(close_button)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Weak topics are ranked by recency-weighted accuracy."))
        layout.addWidget(self.topics, 1)
        layout.addWidget(self.detail)
        layout.addLayout(buttons)
        if not performance:
            self.topics.addItem("No quiz attempts yet. Complete a quiz to start tracking topics.")
            self.review_button.setEnabled(False)
        else:
            for entry in performance:
                item = QListWidgetItem(
                    f"{entry['topic']} · {entry['accuracy']:.0%} accuracy · "
                    f"{entry['attempts']} attempts · {entry['misses']} misses"
                )
                item.setData(Qt.ItemDataRole.UserRole, entry)
                self.topics.addItem(item)
            self.topics.setCurrentRow(0)

    def show_topic(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if not current or not current.data(Qt.ItemDataRole.UserRole):
            return
        entry = current.data(Qt.ItemDataRole.UserRole)
        self.detail.setText(
            f"{entry['topic']}: recent answers average {entry['accuracy']:.0%}. "
            f"Missed {entry['misses']} of {entry['attempts']} attempts; last reviewed {entry['last_attempt']}."
        )

    def review_topic(self) -> None:
        current = self.topics.currentItem()
        if current and current.data(Qt.ItemDataRole.UserRole):
            self.review_requested.emit(current.data(Qt.ItemDataRole.UserRole)["topic"])
            self.accept()


class ClassHistoryDialog(QDialog):
    source_opened = Signal(object)

    def __init__(self, class_name: str, history: list[dict], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{class_name} · question history")
        self.resize(760, 560)
        self.history = history
        self.entries = QListWidget()
        self.entries.currentRowChanged.connect(self.show_entry)
        for entry in history:
            self.entries.addItem(f"{entry['created_at']} · {entry['question']}")
        self.detail = QTextBrowser()
        self.sources = QListWidget()
        self.sources.itemClicked.connect(lambda item: self.source_opened.emit(item.data(Qt.ItemDataRole.UserRole)))
        split = QSplitter()
        detail_panel = QWidget()
        detail_layout = QVBoxLayout(detail_panel)
        detail_layout.addWidget(self.detail, 2)
        detail_layout.addWidget(QLabel("Cited pages — select one to preview"))
        detail_layout.addWidget(self.sources, 1)
        split.addWidget(self.entries)
        split.addWidget(detail_panel)
        layout = QVBoxLayout(self)
        layout.addWidget(split)
        if history:
            self.entries.setCurrentRow(0)

    def show_entry(self, index: int) -> None:
        self.sources.clear()
        if not 0 <= index < len(self.history):
            self.detail.clear()
            return
        entry = self.history[index]
        self.detail.setPlainText(f"You: {entry['question']}\n\nAssistant: {entry['answer']}")
        for source in entry["sources"]:
            label = (
                f"{source['class_name']} · {source['lecture_name']} · "
                f"{Path(source['file_path']).name} · p. {source['page_number']}"
            )
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, source)
            self.sources.addItem(item)


# ---------------------------------------------------------------------------------------------
# Agents and local models
# ---------------------------------------------------------------------------------------------

class ModelDownload(QObject):
    """Pulls a model on a daemon thread, so it outlives the dialog and never blocks quitting.

    Ollama resumes a partial download, so stopping when the app quits loses nothing.
    """

    progress = Signal(str)
    done = Signal(str, str)  # model name, then a readable error (empty on success)

    active: dict[tuple[str, str], "ModelDownload"] = {}

    def __init__(self, host: str, name: str) -> None:
        super().__init__()
        self.host, self.name = host, name
        self.latest = f"Starting download of {name}…"

    @classmethod
    def start(cls, host: str, name: str) -> "ModelDownload":
        download = cls.active.get((host, name))
        if download is None:
            download = cls.active[host, name] = cls(host, name)
            threading.Thread(target=download._run, daemon=True).start()
        return download

    def _report(self, text: str) -> None:
        if text != self.latest:
            self.latest = text
            self.progress.emit(text)

    def _run(self) -> None:
        error = ""
        try:
            OllamaClient(self.host).pull(self.name, self._report)
        except OllamaError as failure:
            error = str(failure)
        del self.active[self.host, self.name]
        self.done.emit(self.name, error)


class ModelField(QWidget):
    """A dropdown of the installed models that fit one role, plus a menu to download more.

    The list can't be typed into: only models Ollama reports as installed (and able to do the job)
    can be chosen. A saved model that has since been removed stays visible, marked NOT INSTALLED,
    so it isn't silently swapped for another one.
    """

    installed_changed = Signal()

    def __init__(self, role: str, host: str, parent: QWidget | None = None, cloud: bool = False) -> None:
        super().__init__(parent)
        self.role, self.host = role, host
        self.cloud = cloud  # Gemini: models live on Google's servers, so nothing to download
        self.hardware = detect_hardware()
        self.installed: list[str] = []
        self.choices: dict = {}
        self._followed: set[ModelDownload] = set()
        self._requested: str | None = None

        self.combo = QComboBox()
        self.combo.setEditable(False)
        self.combo.setMaxVisibleItems(16)
        self.combo.currentIndexChanged.connect(self._refresh)

        self.download_button = QPushButton()
        self.download_button.setToolTip("Download the selected model with Ollama")
        self.download_button.clicked.connect(self.download)
        self.more_button = QPushButton("+ GET MORE")
        self.more_button.setToolTip("Download another model with Ollama")
        self.more_menu = QMenu(self)
        self.more_button.setMenu(self.more_menu)
        self.status = QLabel()
        self.status.setObjectName("dialogHelp")
        self.status.setWordWrap(True)
        self.status.hide()

        row = QHBoxLayout()
        row.addWidget(self.combo, 1)
        row.addWidget(self.download_button)
        row.addWidget(self.more_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(row)
        layout.addWidget(self.status)

    def set_models(self, installed: Sequence[str], current: str, empty_label: str | None = None,
                   offline: bool = False) -> None:
        """Fill the list with installed models. `empty_label` adds a first entry meaning "use the default".

        `offline` means Ollama couldn't be reached, so nothing is known to be missing yet.
        """
        self.offline = offline
        # Keep a saved "name" as written: rewriting it to "name:latest" would re-index every page.
        self.installed = [current if current and name == f"{current}:latest" else name for name in installed]
        self._empty_label = empty_label
        self.combo.blockSignals(True)
        self.combo.clear()
        if empty_label:
            self.combo.addItem(empty_label, "")
        for name in self.installed:
            note = health_label(name) if self.cloud else ""
            self.combo.addItem(f"{name}  ·  {note}" if note else name, name)
        if current and not is_installed(current, self.installed):
            missing = "NOT AVAILABLE" if self.cloud else "NOT INSTALLED"
            unknown = "OFFLINE" if self.cloud else "OLLAMA OFFLINE"
            self.combo.addItem(f"{current}  ·  {unknown if offline else missing}", current)
        if self.combo.count() == 0:
            self.combo.addItem("No models available" if self.cloud else "No models installed · use + GET MORE", "")
            self.combo.model().item(0).setEnabled(False)
        index = self.combo.findData(current) if current else (0 if empty_label else -1)
        self.combo.setCurrentIndex(index if index >= 0 else 0)
        self.combo.blockSignals(False)
        self._build_menu()
        self._refresh()

    def set_cloud(self, cloud: bool) -> None:
        self.cloud = cloud
        self.more_button.setVisible(not cloud)

    def _build_menu(self) -> None:
        self.more_menu.clear()
        self.more_button.setVisible(not self.cloud)
        if self.cloud:
            self.choices = {}
            return
        choices = download_choices(self.role, self.installed, self.hardware)
        self.choices = {choice.name: choice for choice in choices}
        for title, fits in (("FITS THIS MAC", True), ("LARGE FOR THIS MAC", False)):
            group = [choice for choice in choices if choice.fits == fits]
            if not group:
                continue
            self.more_menu.addSection(title)
            for choice in group:
                action = self.more_menu.addAction(f"{choice.name}  ·  {choice.size_gb:g} GB")
                action.triggered.connect(lambda _checked=False, name=choice.name: self.download(name))
        self.more_menu.addSeparator()
        browse = self.more_menu.addAction("Other models: run  ollama pull <name>  in Terminal, then reopen")
        browse.setEnabled(False)

    def model_name(self) -> str:
        """The exact Ollama name of the chosen entry ("" means the default)."""
        name = self.combo.currentData()
        return name if isinstance(name, str) else ""

    def _refresh(self) -> None:
        name = self.model_name()
        download = ModelDownload.active.get((self.host, name)) if name else None
        missing = bool(name) and not is_installed(name, self.installed) and not getattr(self, "offline", False)
        choice = self.choices.get(name)
        self.download_button.setVisible(missing and not self.cloud)
        self.download_button.setEnabled(download is None)
        self.download_button.setText(f"DOWNLOAD {choice.size_gb:g} GB" if choice else "DOWNLOAD")
        if download:
            self._follow(download)
            self._show_status(download.latest)
        elif missing and self.cloud:
            self._show_status(f"Gemini doesn't offer {name} to this API key. Pick one from the list.")
        elif missing:
            self._show_status(f"{name} isn't installed, so it can't answer yet. Download it or pick an installed model.")
        else:
            self._show_status("")

    def _show_status(self, text: str) -> None:
        self.status.setText(text)
        self.status.setVisible(bool(text))

    def _follow(self, download: ModelDownload) -> None:
        if download not in self._followed:
            self._followed.add(download)
            download.progress.connect(self._on_progress)
            download.done.connect(self._on_done)

    def download(self, name: str | None = None) -> None:
        name = name or self.model_name()
        if not name:
            return
        self._requested = name
        self.download_button.setEnabled(False)
        download = ModelDownload.start(self.host, name)
        self._follow(download)
        self._show_status(download.latest)

    def _on_progress(self, text: str) -> None:
        self._show_status(text)

    def _on_done(self, name: str, error: str) -> None:
        if error:
            self._refresh()
            self._show_status(error)
            return
        selected = name if self._requested == name else self.model_name()
        self._requested = None
        self.set_models([*self.installed, name], selected, self._empty_label, offline=False)
        self.installed_changed.emit()


class AgentEditorDialog(QDialog):
    """Create or edit a custom agent: model, creativity, notes access, persona."""

    def __init__(
        self,
        agent: dict | None,
        installed_models: list[str],
        default_model: str,
        parent: QWidget | None = None,
        can_delete: bool = True,
        host: str = DEFAULT_HOST,
        provider_values: dict | None = None,
    ) -> None:
        super().__init__(parent)
        provider_values = provider_values or {"provider": OLLAMA, "host": host}
        cloud = provider_values.get("provider") == GEMINI
        self.agent = dict(agent or {})
        self.deleted = False
        self.setWindowTitle("Configure agent" if agent else "New agent")
        self.resize(560, 640)

        accent = self.agent.get("accent") or ACCENTS[0]
        self.preview = AgentAvatar(self.agent.get("name", "New agent"), accent, 72)
        self.name_input = QLineEdit(self.agent.get("name", ""))
        self.name_input.setPlaceholderText("e.g. CS 354 Tutor")
        self.name_input.textChanged.connect(self._refresh_preview)

        offline = False
        if not installed_models:  # the app hasn't heard from the model service yet, so ask it now
            try:
                installed_models = fetch_models_by_role(make_client(provider_values))["chat"]
            except OllamaError:
                installed_models, offline = [], True
        saved_model = self.agent.get("model", "")
        if not model_fits(saved_model, provider_values.get("provider", OLLAMA)):
            saved_model = ""  # e.g. a local model while on Gemini: start from the default
        self.model_field = ModelField("chat", host, cloud=cloud)
        self.model_field.set_models(
            installed_models if cloud else models_by_role([{"name": name} for name in installed_models])["chat"],
            saved_model,
            f"Default ({default_model})",
            offline=offline,
        )

        self.creativity = QSlider(Qt.Orientation.Horizontal)
        self.creativity.setRange(0, 150)
        self.creativity.setValue(round(float(self.agent.get("temperature", 0.3)) * 100))
        self.creativity_value = QLabel()
        self.creativity_value.setObjectName("statValue")
        self.creativity.valueChanged.connect(self._refresh_creativity)
        creativity_row = QHBoxLayout()
        creativity_row.addWidget(self.creativity, 1)
        creativity_row.addWidget(self.creativity_value)

        self.notes_mode = QComboBox()
        for key, label in NOTES_MODES.items():
            if key != "off":  # the main chat always reads the selected notes; only the overlay skips them
                self.notes_mode.addItem(label, key)
        self.notes_mode.setCurrentIndex(max(0, self.notes_mode.findData(self.agent.get("notes_mode", "strict"))))
        self.notes_help = QLabel()
        self.notes_help.setObjectName("dialogHelp")
        self.notes_help.setWordWrap(True)
        self.notes_mode.currentIndexChanged.connect(self._refresh_notes_help)

        self.reasoning = QCheckBox("Show reasoning (for thinking models like qwen3 or deepseek-r1)")
        self.reasoning.setChecked(bool(self.agent.get("reasoning")))

        self.accent_group = QButtonGroup(self)
        self.accent_group.setExclusive(True)
        swatches = QHBoxLayout()
        swatches.setSpacing(6)
        for color in ACCENTS:
            swatch = QPushButton()
            swatch.setCheckable(True)
            swatch.setFixedSize(26, 26)
            swatch.setProperty("color", color)
            swatch.setStyleSheet(
                f"QPushButton {{ background: {color}; border: 2px solid {COLORS['bg']}; padding: 0; }}"
                f"QPushButton:checked {{ border: 2px solid {COLORS['white']}; }}"
            )
            swatch.setChecked(color.casefold() == accent.casefold())
            self.accent_group.addButton(swatch)
            swatches.addWidget(swatch)
        if not self.accent_group.checkedButton():
            self.accent_group.buttons()[0].setChecked(True)
        self.accent_group.buttonClicked.connect(lambda _b: self._refresh_preview())
        swatches.addStretch(1)

        self.system_prompt = QPlainTextEdit(self.agent.get("system_prompt", ""))
        self.system_prompt.setPlaceholderText(
            "Who is this agent and how should it answer? e.g. “You are a strict OS TA. Quiz me back after each answer.”"
        )

        form = QFormLayout()
        form.setVerticalSpacing(12)
        form.addRow("Name", self.name_input)
        form.addRow("Model", self.model_field)
        form.addRow("Creativity", creativity_row)
        form.addRow("Lecture notes", self.notes_mode)
        form.addRow("", self.notes_help)
        form.addRow("", self.reasoning)
        form.addRow("Color", swatches)

        top = QHBoxLayout()
        top.addWidget(self.preview)
        intro = QLabel("Agents are saved locally. Each one keeps its own conversations per course.")
        intro.setObjectName("dialogHelp")
        intro.setWordWrap(True)
        top.addWidget(intro, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        if agent and agent.get("id"):
            delete = buttons.addButton("Delete agent", QDialogButtonBox.ButtonRole.DestructiveRole)
            delete.setEnabled(can_delete)
            delete.clicked.connect(self._delete)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.addLayout(top)
        layout.addLayout(form)
        layout.addWidget(QLabel("System prompt"))
        layout.addWidget(self.system_prompt, 1)
        layout.addWidget(buttons)
        self._refresh_creativity()
        self._refresh_notes_help()

    def _accent(self) -> str:
        button = self.accent_group.checkedButton()
        return button.property("color") if button else ACCENTS[0]

    def _refresh_preview(self) -> None:
        self.preview.set_agent(self.name_input.text().strip() or "New agent", self._accent())

    def _refresh_creativity(self) -> None:
        value = self.creativity.value() / 100
        self.creativity_value.setText(f"{value:.2f} {creativity_label(value)}")

    def _refresh_notes_help(self) -> None:
        self.notes_help.setText({
            "strict": "Answers only from your imported lecture pages, with numbered citations. Says so when the notes don't cover it.",
            "blend": "Starts from your lecture pages (cited) and fills gaps with general knowledge, clearly marked.",
            "off": "A normal chatbot. Doesn't search your notes.",
        }[self.notes_mode.currentData()])

    def _accept_if_valid(self) -> None:
        if not self.name_input.text().strip():
            QMessageBox.information(self, "Name needed", "Give this agent a name.")
            return
        self.accept()

    def _delete(self) -> None:
        confirm = QMessageBox.question(
            self, "Delete agent", f"Delete {self.agent.get('name')} and its saved conversations?"
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.deleted = True
            self.accept()

    def agent_data(self) -> dict:
        return {
            "id": self.agent.get("id"),
            "name": self.name_input.text().strip(),
            "model": self.model_field.model_name(),
            "temperature": self.creativity.value() / 100,
            "notes_mode": self.notes_mode.currentData(),
            "reasoning": self.reasoning.isChecked(),
            "accent": self._accent(),
            "system_prompt": self.system_prompt.toPlainText().strip(),
        }


class _ModelTester(QThread):
    """Runs the Gemini model check off the UI thread."""

    done = Signal(object, str)  # results, error

    def __init__(self, values: dict) -> None:
        super().__init__()
        self.values = values

    def run(self) -> None:
        try:
            client = make_client(self.values)
            client.include_unavailable = True  # re-test models an earlier check ruled out
            names = [n for n in fetch_models_by_role(client, timeout=15)["chat"] if n.startswith("gemini")]
            results = check_models(names, timeout=15, api_key=client.api_key, api_root=client.api_root)
            self.done.emit(results, "")
        except OllamaError as error:
            self.done.emit({}, str(error))


class ModelSettingsDialog(QDialog):
    """Choose where the models run (Gemini API or Ollama) and the chat, embedding, and vision models."""

    def __init__(self, values: dict, parent: QWidget | None = None, settings=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Models")
        self.resize(640, 620)
        self._settings = settings
        self._saved = {values.get("provider", GEMINI): dict(values)}
        self.online: bool | None = None
        self._roles: dict = {"chat": [], "vision": [], "embedding": []}

        self.provider_combo = QComboBox()
        for key in (GEMINI, OLLAMA):
            self.provider_combo.addItem(PROVIDER_LABELS[key], key)
        self.provider_combo.setCurrentIndex(max(0, self.provider_combo.findData(values.get("provider", GEMINI))))

        saved_key = get_api_key()
        self.key_input = QLineEdit()
        self.key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_input.setPlaceholderText(f"Saved: {mask_key(saved_key)}" if saved_key else "Paste your Gemini API key")
        key_link = QLabel(f'<a href="{KEY_URL}" style="color: {COLORS["cyan"]};">Get a free key</a>')
        key_link.setOpenExternalLinks(True)
        self.key_row = QWidget()
        key_layout = QHBoxLayout(self.key_row)
        key_layout.setContentsMargins(0, 0, 0, 0)
        key_layout.addWidget(self.key_input, 1)
        key_layout.addWidget(key_link)

        self.host_input = QLineEdit(values.get("host", DEFAULT_HOST))
        self.host_row = QWidget()
        host_layout = QHBoxLayout(self.host_row)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.addWidget(self.host_input, 1)

        host = self.host_input.text().strip()
        self.chat_field = ModelField("chat", host)
        self.embed_field = ModelField("embedding", host)
        self.vision_field = ModelField("vision", host)
        self.fields = (self.chat_field, self.embed_field, self.vision_field)
        for field in self.fields:
            field.installed_changed.connect(self.refresh_models)
        self.handwriting = QCheckBox("Read handwriting on slides when indexing (one extra request per page)")
        self.status = QLabel()
        self.status.setObjectName("onlineStatus")
        self.status.setWordWrap(True)
        refresh = QPushButton("REFRESH")
        refresh.clicked.connect(self.refresh_models)
        self.test_button = QPushButton("TEST MODELS")
        self.test_button.setToolTip("Ask every chat model for a one-word reply, all at once, to see which are "
                                    "answering right now and how fast")
        self.test_button.clicked.connect(self.test_models)
        self.use_best_button = QPushButton("USE RECOMMENDED")
        self.use_best_button.clicked.connect(self.use_recommended)
        self.use_best_button.hide()
        self.test_status = QLabel()
        self.test_status.setObjectName("dialogHelp")
        self.test_status.setWordWrap(True)
        self._recommended: str | None = None
        self._tester: _ModelTester | None = None

        self.help_text = QLabel()
        self.help_text.setObjectName("dialogHelp")
        self.help_text.setWordWrap(True)
        self.help_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        form = QFormLayout()
        form.setVerticalSpacing(12)
        provider_row = QHBoxLayout()
        provider_row.addWidget(self.provider_combo, 1)
        provider_row.addWidget(refresh)
        form.addRow("Models run on", provider_row)
        form.addRow("Gemini API key", self.key_row)
        form.addRow("Ollama server", self.host_row)
        self._key_label = form.labelForField(self.key_row)
        self._host_label = form.labelForField(self.host_row)
        form.addRow("", self.status)
        form.addRow("Default chat model", self.chat_field)
        form.addRow("Embedding model", self.embed_field)
        form.addRow("Vision model", self.vision_field)
        form.addRow("", self.handwriting)
        self.test_row = QWidget()
        test_layout = QHBoxLayout(self.test_row)
        test_layout.setContentsMargins(0, 0, 0, 0)
        test_layout.addWidget(self.test_button)
        test_layout.addWidget(self.use_best_button)
        test_layout.addWidget(self.test_status, 1)
        form.addRow("Model check", self.test_row)
        self._test_label = form.labelForField(self.test_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.addLayout(form)
        layout.addWidget(self.help_text, 1)
        layout.addWidget(buttons)
        self.provider_combo.currentIndexChanged.connect(self._provider_changed)
        self._apply_provider(initial=True)

    # -- provider --------------------------------------------------------------------------
    def provider(self) -> str:
        return self.provider_combo.currentData()

    def _values_for(self, provider: str) -> dict:
        if provider not in self._saved:
            if self._settings is not None:
                from .providers import load_settings

                self._saved[provider] = load_settings(self._settings, provider)
            else:
                from .providers import DEFAULTS

                self._saved[provider] = {"provider": provider, "host": self.host_input.text(), **DEFAULTS[provider]}
        return self._saved[provider]

    def _provider_changed(self) -> None:
        # remember what was picked for the provider we're leaving, then show the other one
        previous = OLLAMA if self.provider() == GEMINI else GEMINI
        if previous in self._saved and self.chat_field.combo.count():
            self._saved[previous].update(self._picked())
        for field in self.fields:
            field.combo.clear()
        self._apply_provider(initial=True)

    def _apply_provider(self, initial: bool) -> None:
        cloud = self.provider() == GEMINI
        for widget in (self.key_row, self._key_label):
            widget.setVisible(cloud)
        for widget in (self.host_row, self._host_label):
            widget.setVisible(not cloud)
        for widget in (self.test_row, self._test_label):
            widget.setVisible(cloud)
        for field in self.fields:
            field.set_cloud(cloud)
        self.handwriting.setChecked(bool(self._values_for(self.provider()).get("handwriting", not cloud)))
        if cloud:
            self.help_text.setText(
                "The models run on Google's servers, so your Mac stays fast. Each question sends the relevant "
                "lecture pages (and, for the screen overlay, the screenshot) to Google. On the free tier Google "
                "may use that content to improve its products and limits how many requests you can make per "
                "minute and per day; a billed key avoids both.\n\n"
                "The lists show the models your key can use. Flash models are fast and fine for studying; Pro "
                "models reason better but are slower and have tighter free limits. Changing the embedding model "
                "re-indexes your pages the next time you press Index."
            )
        else:
            self.help_text.setText(
                "Everything runs on this Mac through Ollama: no API key, and nothing is uploaded, but it needs a "
                "lot of memory and slows other apps while it answers.\n\n"
                "Each list shows the models installed on this Mac that can do that job. Use + GET MORE to "
                "download another. Changing the embedding model re-indexes your pages the next time you press Index."
            )
        self.refresh_models()

    # -- models ----------------------------------------------------------------------------
    def _client_values(self) -> dict:
        values = {**self._values_for(self.provider()), "provider": self.provider(),
                  "host": self.host_input.text().strip() or DEFAULT_HOST}
        typed = self.key_input.text().strip()
        if typed:
            values["api_key"] = typed
        return values

    def refresh_models(self) -> None:
        provider = self.provider()
        values = self._client_values()
        host = values["host"]
        for field in self.fields:
            field.host = host
        try:
            roles = fetch_models_by_role(make_client(values), timeout=8 if provider == GEMINI else 2)
            self.online = True
            count = len({*roles["chat"], *roles["embedding"]})
            where = "AVAILABLE TO THIS KEY" if provider == GEMINI else "INSTALLED"
            self.status.setText(f"◆ ONLINE · {count} MODEL{'S' if count != 1 else ''} {where}")
            self.status.setStyleSheet(f"color: {COLORS['ok']};")
        except OllamaError as error:
            roles = {"chat": [], "vision": [], "embedding": []}
            self.online = False
            self.status.setText(f"◇ {error}")
            self.status.setStyleSheet(f"color: {COLORS['bad']};")
        self._roles = roles
        saved = self._values_for(provider)
        initial = self.chat_field.combo.count() == 0
        chat = saved.get("chat_model", "") if initial else self.chat_field.model_name()
        embed = saved.get("embedding_model", "") if initial else self.embed_field.model_name()
        vision = saved.get("vision_model", "") if initial else self.vision_field.model_name()
        if provider == GEMINI and self.online:  # aliases or retired names: fall back to what the key offers
            if roles["chat"] and chat not in roles["chat"]:
                chat = best_model(roles["chat"], "chat")
            if roles["embedding"] and embed not in roles["embedding"]:
                embed = best_model(roles["embedding"], "embedding")
            if vision and roles["vision"] and vision not in roles["vision"]:
                vision = chat
        offline = not self.online
        self.chat_field.set_models(roles["chat"], chat, offline=offline)
        self.embed_field.set_models(roles["embedding"], embed, offline=offline)
        self.vision_field.set_models(roles["vision"], vision, "None · skip handwriting and screen reading",
                                     offline=offline)

    def chat_models(self) -> list[str]:
        return list(self._roles["chat"])

    # -- model check (Gemini) --------------------------------------------------------------
    def test_models(self) -> None:
        """Probe every chat model in the background; annotate the dropdowns and recommend one."""
        if self._tester is not None:
            return
        values = self._client_values()
        self.test_button.setEnabled(False)
        self.use_best_button.hide()
        self.test_status.setText("Asking every model for a one-word reply…")
        self._tester = _ModelTester(values)
        self._tester.done.connect(self._tested)
        self._tester.start()

    def _tested(self, results: object, error: str) -> None:
        self._tester = None
        self.test_button.setEnabled(True)
        if error:
            self.test_status.setText(f"◇ {error}")
            return
        if self._settings is not None:
            from .providers import save_health

            save_health(self._settings, results)
        working = [n for n, r in results.items() if r["kind"] == "ok"]
        self._recommended = recommend(results)
        summary = f"✓ {len(working)} answering · ✗ {len(results) - len(working)} not"
        if self._recommended:
            summary += f" · recommended: {self._recommended}"
            self.use_best_button.show()
        else:
            summary += ". Nothing answered: Google may be overloaded, or the daily quota is used up."
        self.test_status.setText(summary)
        self.refresh_models()  # re-list: hides models the key can't use and shows ✓/✗ next to the rest

    def done(self, result: int) -> None:  # noqa: D401 - QDialog.done: closing (Save, Cancel, or Esc)
        if self._tester is not None:
            self.test_status.setText("Finishing the model check…")
            self._tester.wait(20000)  # a running QThread must not be destroyed with the dialog
        super().done(result)

    def use_recommended(self) -> None:
        if not self._recommended:
            return
        for field in (self.chat_field, self.vision_field):
            index = field.combo.findData(self._recommended)
            if index >= 0:
                field.combo.setCurrentIndex(index)
        self.test_status.setText(f"{self._recommended} will be the default for chat and screen reading. Press Save.")

    def _picked(self) -> dict:
        return {
            "chat_model": self.chat_field.model_name(),
            "embedding_model": self.embed_field.model_name(),
            "vision_model": self.vision_field.model_name(),
            "handwriting": self.handwriting.isChecked(),
        }

    def values(self) -> dict:
        saved = self._values_for(self.provider())
        picked = self._picked()
        return {
            "provider": self.provider(),
            "host": self.host_input.text().strip(),
            "api_key": self.key_input.text().strip(),
            "chat_model": picked["chat_model"] or saved.get("chat_model") or DEFAULT_CHAT_MODEL,
            "embedding_model": picked["embedding_model"] or saved.get("embedding_model") or DEFAULT_EMBEDDING_MODEL,
            "vision_model": picked["vision_model"],
            "handwriting": picked["handwriting"],
        }


# ---------------------------------------------------------------------------------------------
# Overlay shortcuts
# ---------------------------------------------------------------------------------------------

DEFAULT_SHORTCUTS = {"toggle": "cmd+shift+space", "read": "cmd+shift+return"}

_SHIFTED = {"?": "/", "|": "\\", "<": ",", ">": ".", ":": ";", '"': "'", "~": "`", "_": "-", "+": "=",
            "{": "[", "}": "]", "!": "1", "@": "2", "#": "3", "$": "4", "%": "5", "^": "6", "&": "7",
            "*": "8", "(": "9", ")": "0"}


def shortcut_from_sequence(sequence: QKeySequence, mac: bool = sys.platform == "darwin") -> Shortcut | None:
    """Turn what QKeySequenceEdit recorded into a Shortcut (None if empty or a key we can't bind)."""
    if sequence.isEmpty():
        return None
    combination = sequence[0]
    modifiers = combination.keyboardModifiers()
    names = set()
    # Qt on macOS reports ⌘ as ControlModifier and ⌃ as MetaModifier.
    command_flag, control_flag = ((Qt.KeyboardModifier.ControlModifier, Qt.KeyboardModifier.MetaModifier) if mac
                                  else (Qt.KeyboardModifier.MetaModifier, Qt.KeyboardModifier.ControlModifier))
    if modifiers & command_flag:
        names.add("cmd")
    if modifiers & control_flag:
        names.add("control")
    if modifiers & Qt.KeyboardModifier.AltModifier:
        names.add("option")
    if modifiers & Qt.KeyboardModifier.ShiftModifier:
        names.add("shift")
    key_text = QKeySequence(combination.key()).toString(QKeySequence.SequenceFormat.PortableText).casefold()
    key_text = {"enter": "return", "backtab": "tab"}.get(key_text, _SHIFTED.get(key_text, key_text))
    if key_text not in KEY_CODES:
        return None
    return Shortcut(key_text, frozenset(names))


class ShortcutField(QWidget):
    """Press a key combination to record it; OFF disables it and RESET restores the default."""

    def __init__(self, current: str, default: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.default = default
        self.value = current
        self.recorder = QKeySequenceEdit()
        self.recorder.setMaximumSequenceLength(1)
        self.recorder.editingFinished.connect(self._recorded)
        self.label = QLabel()
        self.label.setMinimumWidth(84)
        self.label.setStyleSheet(f"color: {COLORS['cyan']}; font-size: 15px; font-weight: 700;")
        off = QPushButton("OFF")
        off.clicked.connect(lambda: self.set_value("off"))
        reset = QPushButton("RESET")
        reset.clicked.connect(lambda: self.set_value(self.default))
        self.error = QLabel()
        self.error.setObjectName("dialogHelp")
        self.error.setWordWrap(True)
        self.error.hide()
        row = QHBoxLayout()
        row.addWidget(self.label)
        row.addWidget(self.recorder, 1)
        row.addWidget(off)
        row.addWidget(reset)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(row)
        layout.addWidget(self.error)
        self.set_value(current)

    def set_value(self, text: str) -> None:
        self.value = text
        self.label.setText(shortcut_label(text))
        self.recorder.blockSignals(True)
        self.recorder.clear()
        self.recorder.blockSignals(False)
        self.error.hide()

    def _recorded(self) -> None:
        shortcut = shortcut_from_sequence(self.recorder.keySequence())
        self.recorder.blockSignals(True)
        self.recorder.clear()
        self.recorder.blockSignals(False)
        if shortcut is None:
            self._show_error("That key can't be used as a global shortcut. Try a letter, Space, Return or F-key.")
            return
        try:
            shortcut.validate()
        except ValueError as problem:
            self._show_error(str(problem))
            return
        self.set_value(shortcut.text)

    def _show_error(self, text: str) -> None:
        self.error.setText(text)
        self.error.show()


class ShortcutsDialog(QDialog):
    """Choose the global shortcuts for the screen overlay."""

    def __init__(self, values: dict, failed: Sequence[str] = (), parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Overlay shortcuts")
        self.resize(520, 300)
        self.toggle_field = ShortcutField(values.get("toggle", DEFAULT_SHORTCUTS["toggle"]), DEFAULT_SHORTCUTS["toggle"])
        self.read_field = ShortcutField(values.get("read", DEFAULT_SHORTCUTS["read"]), DEFAULT_SHORTCUTS["read"])
        form = QFormLayout()
        form.setVerticalSpacing(14)
        form.addRow("Show / hide overlay", self.toggle_field)
        form.addRow("Read screen now", self.read_field)
        help_text = QLabel(
            "Click a field and press the keys you want. These work from any app while Cortex is open. "
            "Include ⌘, ⌃ or ⌥. Avoid ⌘Space (Spotlight) and ⌃Space / ⌃⌥Space (macOS input-source switching)."
        )
        help_text.setObjectName("dialogHelp")
        help_text.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.addLayout(form)
        if failed:
            warning = QLabel(
                "Another app already uses " + " and ".join(failed) + ", so Cortex couldn't claim it. Pick another one."
            )
            warning.setWordWrap(True)
            warning.setStyleSheet(f"color: {COLORS['warn']};")
            layout.addWidget(warning)
        layout.addWidget(help_text)
        layout.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _accept_if_valid(self) -> None:
        values = self.values()
        if values["toggle"] != "off" and values["toggle"] == values["read"]:
            QMessageBox.information(self, "Same shortcut twice", "Pick different keys for the two shortcuts.")
            return
        self.accept()

    def values(self) -> dict:
        return {"toggle": self.toggle_field.value, "read": self.read_field.value}
