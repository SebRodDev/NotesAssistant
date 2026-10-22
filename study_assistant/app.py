"""Desktop interface for searching and inspecting lecture PDFs."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .credentials import delete_api_key, get_api_key, save_api_key
from .gemini import EMBEDDING_MODEL, GeminiService
from .pdf_reader import extract_page_texts, render_page
from .storage import Library


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


class ApiKeyDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Gemini API settings")
        self.key_input = QLineEdit()
        self.key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_input.setPlaceholderText("Paste your Gemini API key")
        existing = get_api_key()
        if existing:
            self.key_input.setText(existing)
        note = QLabel(
            "The key is normally stored in macOS Keychain. GEMINI_API_KEY, when set, takes precedence. "
            "Semantic search sends your query to Gemini. "
            "Indexing (including automatic visual-note reading), grounded Q&A, and quiz creation/grading send relevant "
            "PDF page text and/or rendered images. Use an unbilled free-tier project to avoid API charges."
        )
        note.setWordWrap(True)
        form = QFormLayout()
        form.addRow("API key", self.key_input)
        buttons = QDialogButtonBox()
        save_button = buttons.addButton("Save to Keychain", QDialogButtonBox.ButtonRole.AcceptRole)
        clear_button = buttons.addButton("Remove saved key", QDialogButtonBox.ButtonRole.DestructiveRole)
        cancel_button = buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        save_button.clicked.connect(self.save)
        clear_button.clicked.connect(self.clear)
        cancel_button.clicked.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(note)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def save(self) -> None:
        value = self.key_input.text().strip()
        if not value:
            QMessageBox.warning(self, "API key needed", "Paste a Gemini API key or choose Remove saved key.")
            return
        try:
            persisted = save_api_key(value)
        except Exception as error:
            QMessageBox.critical(self, "Could not save key", str(error))
            return
        if not persisted:
            QMessageBox.information(
                self,
                "Keychain unavailable",
                "The key will work until you quit this app. To keep it between launches, set GEMINI_API_KEY before starting the app.",
            )
        elif os.environ.get("GEMINI_API_KEY") and os.environ["GEMINI_API_KEY"] != value:
            QMessageBox.information(
                self,
                "Environment key is active",
                "The Keychain key was saved, but GEMINI_API_KEY is set to a different key and remains active until the app is restarted or that environment variable is changed.",
            )
        self.accept()

    def clear(self) -> None:
        try:
            delete_api_key()
        except Exception as error:
            QMessageBox.critical(self, "Could not remove key", str(error))
            return
        self.key_input.clear()
        if os.environ.get("GEMINI_API_KEY"):
            QMessageBox.information(
                self,
                "Environment key remains active",
                "The saved Keychain key was removed, but GEMINI_API_KEY is still set and remains active.",
            )
        self.accept()


class BackgroundTask(QThread):
    progress = Signal(str)

    def __init__(self, action: Callable[[Callable[[str], None]], object]) -> None:
        super().__init__()
        self.action = action
        self.result: object | None = None
        self.error: str | None = None

    def run(self) -> None:
        try:
            self.result = self.action(self.progress.emit)
        except Exception as error:  # API/SQLite errors are shown in the app rather than killing the UI.
            self.error = str(error)


class ChatMessageWidget(QFrame):
    """A readable chat bubble with source links and optional follow-up actions."""

    source_opened = Signal(object)
    explain_selected = Signal(str)

    def __init__(
        self,
        role: str,
        text: str,
        sources: list[dict] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.role = role
        self.text = text
        self.sources = sources or []
        self.setObjectName("assistantMessage" if role == "assistant" else "userMessage")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)
        heading = QLabel("Study assistant" if role == "assistant" else "You")
        heading.setObjectName("messageHeading")
        layout.addWidget(heading, 0, Qt.AlignmentFlag.AlignLeft)

        self.body = QLabel(text)
        self.body.setTextFormat(Qt.TextFormat.PlainText)
        self.body.setWordWrap(True)
        self.body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.body.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.body.setStyleSheet("background: transparent; border: none; padding: 0;")
        layout.addWidget(self.body)

        self.source_buttons: list[QPushButton] = []
        if self.sources:
            source_heading = QLabel("Sources — select a page to preview it")
            source_heading.setObjectName("sourceHeading")
            layout.addWidget(source_heading)
            for source in self.sources:
                button = QPushButton(self._source_label(source))
                button.setObjectName("sourceLink")
                button.setCursor(Qt.CursorShape.PointingHandCursor)
                button.setToolTip("Open this cited PDF page")
                button.clicked.connect(lambda _checked=False, page=source: self.source_opened.emit(page))
                self.source_buttons.append(button)
                layout.addWidget(button, 0, Qt.AlignmentFlag.AlignLeft)

        if role == "assistant" and self.sources:
            actions = QHBoxLayout()
            copy_button = QPushButton("Copy cited prompt")
            copy_button.clicked.connect(self.copy_prompt)
            explain_button = QPushButton("Explain selected text")
            explain_button.clicked.connect(self.explain_passage)
            actions.addWidget(copy_button)
            actions.addWidget(explain_button)
            actions.addStretch(1)
            layout.addLayout(actions)

    def set_available_width(self, width: int) -> None:
        maximum = max(280, width)
        self.setMinimumWidth(min(500 if self.role == "assistant" else 300, maximum))
        self.setMaximumWidth(maximum)
        for button in self.source_buttons:
            button.setMaximumWidth(maximum - 32)

    @staticmethod
    def _source_label(source: dict) -> str:
        return (
            f"{source['class_name']} · {source['lecture_name']} · "
            f"{Path(source['file_path']).name} · p. {source['page_number']}"
        )

    def copy_prompt(self) -> None:
        selected = self.body.selectedText().strip()
        passage = selected or self.text
        evidence = []
        for source in self.sources:
            label = self._source_label(source)
            text = source.get("text", "").strip()
            visual = source.get("visual_text", "").strip()
            evidence.append(f"[{label}]\n{text or 'No embedded PDF text; consult the original page.'}")
            if visual:
                evidence.append(f"AI-read visual notes (verify against page):\n{visual}")
        prompt = (
            "Explain the selected passage more deeply using only the course evidence below. "
            "Cite each factual claim with its class, lecture, PDF file, and page. If the evidence "
            "does not support a claim, say so instead of guessing.\n\n"
            f"Passage to explain:\n{passage}\n\nCourse evidence:\n" + "\n\n".join(evidence)
        )
        QApplication.clipboard().setText(prompt)
        QMessageBox.information(self, "Copied", "A deeper-explanation prompt with cited evidence is on your clipboard.")

    def explain_passage(self) -> None:
        selected = self.body.selectedText().strip()
        if not selected:
            QMessageBox.information(self, "Select some text", "Highlight part of the answer first, then choose Explain selected text.")
            return
        self.explain_selected.emit(selected)


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


class MainWindow(QMainWindow):
    def __init__(self, library: Library | None = None, settings: QSettings | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Purdue Study Assistant")
        self.resize(1420, 900)
        self.library = library or Library()
        self.settings = settings or QSettings("Purdue", "StudyAssistant")
        self.active_task: BackgroundTask | None = None
        self.task_success_handler: Callable[[object], None] | None = None
        self.task_error_handler: Callable[[str], None] | None = None
        self.closing = False
        self.current_result: dict | None = None
        self.current_scope: tuple[str | None, str | None] = (None, None)
        self.message_widgets: list[ChatMessageWidget] = []
        self._conversation_cache: dict[
            tuple[str | None, str | None], list[tuple[str, str, list[dict]]]
        ] = {}

        self.navigator = QTreeWidget()
        self.navigator.setObjectName("courseNavigator")
        self.navigator.setHeaderLabel("Classes and lectures")
        self.navigator.setMinimumWidth(210)
        self.navigator.currentItemChanged.connect(self.change_scope)

        self.import_button = QPushButton("＋  Add lecture PDFs")
        self.import_button.setObjectName("primaryAction")
        self.import_button.setShortcut(QKeySequence.StandardKey.Open)
        self.import_button.setToolTip("Add a lecture PDF (Command+O on macOS)")
        self.import_button.clicked.connect(self.import_pdf)
        self.index_button = QPushButton("Index PDFs for AI search")
        self.index_button.setToolTip(
            "Read handwriting and diagrams from page images, then create searchable page embeddings. "
            "Page text and images are sent to Gemini."
        )
        self.index_button.clicked.connect(self.index_library)
        self.key_button = QPushButton("AI settings…")
        self.key_button.clicked.connect(self.configure_api_key)
        self.dark_mode_toggle = QCheckBox("Dark mode")
        self.dark_mode_toggle.setObjectName("darkModeToggle")
        self.dark_mode_toggle.setChecked(self.settings.value("appearance/dark_mode", False, type=bool))
        self.dark_mode_toggle.toggled.connect(self.set_dark_mode)
        self.quiz_button = QPushButton("Practice quiz…")
        self.quiz_button.clicked.connect(self.start_quiz)
        self.weak_spots_button = QPushButton("Weak spots")
        self.weak_spots_button.clicked.connect(self.open_weak_spots)
        self.history_button = QPushButton("Question history")
        self.history_button.clicked.connect(self.open_class_history)
        self.api_status = QLabel()
        self.api_status.setWordWrap(True)
        self.api_status.setObjectName("mutedText")
        self.library_status = QLabel()
        self.library_status.setWordWrap(True)
        self.library_status.setObjectName("mutedText")
        self.library_status.setToolTip(
            "AI indexing also checks page images for handwriting and diagram labels."
        )

        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setMinimumWidth(230)
        sidebar.setMaximumWidth(320)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(16, 18, 16, 16)
        sidebar_layout.setSpacing(10)
        brand = QLabel("Purdue Study Assistant")
        brand.setObjectName("brandTitle")
        sidebar_layout.addWidget(brand)
        sidebar_layout.addWidget(QLabel("Your course material, one conversation at a time."))
        sidebar_layout.addSpacing(8)
        sidebar_layout.addWidget(self.import_button)
        courses_heading = QLabel("YOUR COURSES")
        courses_heading.setObjectName("sectionHeading")
        sidebar_layout.addWidget(courses_heading)
        sidebar_layout.addWidget(self.navigator, 1)
        tools_heading = QLabel("STUDY TOOLS")
        tools_heading.setObjectName("sectionHeading")
        sidebar_layout.addWidget(tools_heading)
        sidebar_layout.addWidget(self.quiz_button)
        sidebar_layout.addWidget(self.weak_spots_button)
        sidebar_layout.addWidget(self.history_button)
        sidebar_layout.addSpacing(8)
        sidebar_layout.addWidget(self.library_status)
        sidebar_layout.addWidget(self.index_button)
        sidebar_layout.addWidget(self.key_button)
        sidebar_layout.addWidget(self.dark_mode_toggle)
        self.api_status.setText(self._key_status())
        sidebar_layout.addWidget(self.api_status)

        self.scope_title = QLabel("All materials")
        self.scope_title.setObjectName("workspaceTitle")
        self.scope_subtitle = QLabel("Ask a question or find a page in your PDFs. Answers cite the source pages they use.")
        self.scope_subtitle.setObjectName("mutedText")
        self.preview_toggle = QPushButton("Show page preview")
        self.preview_toggle.setCheckable(True)
        self.preview_toggle.toggled.connect(self.toggle_preview)
        chat_header = QHBoxLayout()
        title_stack = QVBoxLayout()
        title_stack.addWidget(self.scope_title)
        title_stack.addWidget(self.scope_subtitle)
        chat_header.addLayout(title_stack, 1)
        chat_header.addWidget(self.preview_toggle)

        self.chat_scroll = QScrollArea()
        self.chat_scroll.setObjectName("chatScroll")
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chat_body = QWidget()
        self.chat_body.setObjectName("chatBody")
        self.chat_layout = QVBoxLayout(self.chat_body)
        self.chat_layout.setContentsMargins(28, 24, 28, 24)
        self.chat_layout.setSpacing(14)
        self.chat_layout.addStretch(1)
        self.chat_scroll.setWidget(self.chat_body)

        self.query = QLineEdit()
        self.query.setObjectName("composerInput")
        self.query.setPlaceholderText("Ask something about your lectures…")
        self.query.returnPressed.connect(self.send_message)
        self.mode_combo = QComboBox()
        self.mode_combo.setObjectName("messageMode")
        self.mode_combo.addItem("Ask from my PDFs", "ask")
        self.mode_combo.addItem("Find a page", "find")
        self.mode_combo.currentIndexChanged.connect(self.update_composer_mode)
        self.semantic_search = QCheckBox("Search by meaning")
        self.semantic_search.setToolTip("Also use Gemini embeddings to find conceptually related pages.")
        self.semantic_search.setChecked(False)
        self.semantic_search.hide()
        self.send_button = QPushButton("Ask")
        self.send_button.setObjectName("sendButton")
        self.send_button.clicked.connect(self.send_message)
        composer_top = QHBoxLayout()
        composer_top.addWidget(self.mode_combo)
        composer_top.addWidget(self.semantic_search)
        composer_top.addStretch(1)
        composer_row = QHBoxLayout()
        composer_row.addWidget(self.query, 1)
        composer_row.addWidget(self.send_button)
        composer = QFrame()
        composer.setObjectName("composer")
        composer_layout = QVBoxLayout(composer)
        composer_layout.setContentsMargins(14, 10, 14, 10)
        composer_layout.addLayout(composer_top)
        composer_layout.addLayout(composer_row)
        self.composer_help = QLabel()
        self.composer_help.setObjectName("mutedText")
        composer_layout.addWidget(self.composer_help)

        self.preview_title = QLabel("Choose a citation to inspect its source page")
        self.preview_title.setObjectName("previewTitle")
        self.preview_title.setWordWrap(True)
        self.preview = QLabel("PDF page preview")
        self.preview.setObjectName("pdfPreview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(300, 360)
        self.visual_notes_hint = QLabel("Handwriting and visual labels are read automatically when you index PDFs.")
        self.visual_notes_hint.setObjectName("mutedText")
        self.visual_notes_hint.setWordWrap(True)
        self.excerpt = QTextBrowser()
        self.excerpt.setPlaceholderText("Extracted text and AI-read visual notes appear here.")
        self.excerpt.setMinimumHeight(120)
        self.excerpt.setOpenExternalLinks(False)
        preview_panel = QWidget()
        preview_panel.setObjectName("previewPanel")
        preview_panel.setMinimumWidth(280)
        preview_panel.setMaximumWidth(440)
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(12, 10, 12, 10)
        preview_layout.addWidget(self.preview_title)
        preview_layout.addWidget(self.preview, 2)
        preview_layout.addWidget(self.visual_notes_hint)
        preview_layout.addWidget(self.excerpt, 1)
        preview_panel.hide()

        chat_content = QWidget()
        chat_content_layout = QVBoxLayout(chat_content)
        chat_content_layout.setContentsMargins(0, 0, 0, 0)
        chat_content_layout.setSpacing(10)
        chat_content_layout.addLayout(chat_header)
        chat_content_layout.addWidget(self.chat_scroll, 1)
        chat_content_layout.addWidget(composer)
        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.content_splitter.addWidget(chat_content)
        self.content_splitter.addWidget(preview_panel)
        self.content_splitter.setStretchFactor(0, 1)
        self.content_splitter.setStretchFactor(1, 0)
        self.content_splitter.setSizes([1000, 0])
        self.preview_panel = preview_panel

        workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
        workspace_splitter.addWidget(sidebar)
        workspace_splitter.addWidget(self.content_splitter)
        workspace_splitter.setStretchFactor(0, 0)
        workspace_splitter.setStretchFactor(1, 1)
        workspace_splitter.setSizes([260, 1100])

        self.task_controls = [
            self.navigator, self.query, self.mode_combo, self.semantic_search, self.send_button,
            self.import_button, self.index_button, self.key_button, self.quiz_button,
            self.weak_spots_button, self.history_button, self.dark_mode_toggle,
        ]
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(workspace_splitter)
        self.setCentralWidget(root)
        self.set_dark_mode(self.dark_mode_toggle.isChecked(), persist=False)
        saved_class = self.settings.value("study_space/class", "", type=str).strip()
        saved_lecture = self.settings.value("study_space/lecture", "", type=str).strip()
        self.refresh_navigation((saved_class or None, saved_lecture or None))
        self._append_message(
            "assistant",
            "Hi! I can help you study from your lecture PDFs. Choose a course on the left, then ask a question or switch to Find a page to locate relevant slides. Every answer is grounded in retrieved pages and includes citations you can open.",
        )
        self.update_composer_mode()
        self.statusBar().showMessage(self._key_status())

    def set_dark_mode(self, enabled: bool, persist: bool = True) -> None:
        """Apply a coordinated light or dark palette and optionally save the preference."""
        if persist:
            self.settings.setValue("appearance/dark_mode", enabled)
        if enabled:
            colors = {
                "window": "#171a20", "sidebar": "#1d2129", "surface": "#222731",
                "raised": "#282e39", "border": "#363e4b", "text": "#e8ebf0",
                "muted": "#a1a9b7", "accent": "#7aa2f7", "accent_hover": "#91b2fa",
                "accent_soft": "#293750", "user": "#27364d", "input": "#1c2129",
                "hover": "#303846", "pressed": "#394457", "preview": "#171a20",
            }
        else:
            colors = {
                "window": "#f6f7f9", "sidebar": "#eef1f5", "surface": "#ffffff",
                "raised": "#f8f9fb", "border": "#dce1e8", "text": "#202733",
                "muted": "#667085", "accent": "#385eaa", "accent_hover": "#2d4f93",
                "accent_soft": "#f1f5fc", "user": "#edf3fd", "input": "#ffffff",
                "hover": "#f0f3f8", "pressed": "#e7ebf2", "preview": "#f3f5f8",
            }
        c = colors
        self.setStyleSheet(
            f"QMainWindow {{ background: {c['window']}; color: {c['text']}; }}"
            f"QWidget {{ color: {c['text']}; }}"
            f"#sidebar {{ background: {c['sidebar']}; border-right: 1px solid {c['border']}; }}"
            f"#brandTitle {{ font-size: 18px; font-weight: 700; color: {c['text']}; }}"
            f"#workspaceTitle {{ font-size: 22px; font-weight: 650; color: {c['text']}; }}"
            f"#sectionHeading, #sourceHeading {{ color: {c['muted']}; font-size: 10px; font-weight: 700; }}"
            f"#mutedText {{ color: {c['muted']}; font-size: 11px; }}"
            f"#chatScroll, #chatBody {{ background: {c['raised']}; border: none; }}"
            f"#assistantMessage {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 16px; }}"
            f"#userMessage {{ background: {c['user']}; border: 1px solid {c['border']}; border-radius: 16px; }}"
            f"#messageHeading {{ color: {c['text']}; font-weight: 650; }}"
            f"#composer {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 16px; }}"
            f"#composerInput {{ background: transparent; color: {c['text']}; border: none; padding: 8px; font-size: 14px; }}"
            f"#pdfPreview {{ background: {c['preview']}; color: {c['muted']}; border: 1px solid {c['border']}; border-radius: 12px; }}"
            f"#previewPanel {{ background: {c['surface']}; border-left: 1px solid {c['border']}; }}"
            f"#courseNavigator, QTextBrowser, QPlainTextEdit, QLineEdit, QComboBox, QListWidget {{"
            f"background: {c['input']}; color: {c['text']}; border: 1px solid {c['border']}; border-radius: 9px; }}"
            f"QTreeWidget#courseNavigator, QTreeView#courseNavigator {{ background: {c['input']}; color: {c['text']}; }}"
            f"QHeaderView::section {{ background: {c['sidebar']}; color: {c['muted']}; border: none; padding: 5px; }}"
            f"QTreeWidget::item {{ padding: 5px 4px; border-radius: 6px; }}"
            f"QTreeWidget::item:selected, QListWidget::item:selected {{ background: {c['accent_soft']}; color: {c['text']}; }}"
            f"QPushButton {{ background: {c['surface']}; color: {c['text']}; border: 1px solid {c['border']};"
            f" border-radius: 11px; padding: 8px 12px; font-weight: 550; }}"
            f"QPushButton:hover {{ background: {c['hover']}; border-color: {c['accent']}; }}"
            f"QPushButton:pressed {{ background: {c['pressed']}; }}"
            f"QPushButton:disabled {{ color: {c['muted']}; background: {c['sidebar']}; }}"
            f"#sendButton, #primaryAction {{ background: {c['accent']}; color: #ffffff; border: none;"
            f" font-weight: 650; padding: 9px 17px; border-radius: 13px; }}"
            f"#sendButton:hover, #primaryAction:hover {{ background: {c['accent_hover']}; }}"
            f"#sourceLink {{ color: {c['accent']}; text-align: left; border: 1px solid {c['border']};"
            f" background: {c['accent_soft']}; padding: 7px 11px; border-radius: 12px; }}"
            f"#sourceLink:hover {{ border-color: {c['accent']}; }}"
            f"QCheckBox {{ spacing: 8px; }} QCheckBox::indicator {{ width: 15px; height: 15px; }}"
            f"QSplitter::handle {{ background: {c['border']}; }}"
        )

    def _append_message(self, role: str, text: str, sources: list[dict] | None = None) -> ChatMessageWidget:
        message = ChatMessageWidget(role, text, sources, self.chat_body)
        message.set_available_width(min(900 if role == "assistant" else 720, max(300, self.chat_scroll.viewport().width() - 64)))
        message.source_opened.connect(self.open_source_page)
        message.explain_selected.connect(lambda passage, refs=message.sources: self.explain_more(passage, refs))
        alignment = Qt.AlignmentFlag.AlignRight if role == "user" else Qt.AlignmentFlag.AlignLeft
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, message, 0, alignment)
        self.message_widgets.append(message)
        QTimer.singleShot(0, self._refresh_message_layout)
        QTimer.singleShot(0, lambda: self.chat_scroll.verticalScrollBar().setValue(
            self.chat_scroll.verticalScrollBar().maximum()
        ))
        return message

    def _clear_chat(self) -> None:
        for message in self.message_widgets:
            self.chat_layout.removeWidget(message)
            message.deleteLater()
        self.message_widgets.clear()

    def _begin_scope_conversation(self, scope: tuple[str | None, str | None]) -> None:
        cached = self._conversation_cache.get(scope)
        if cached is not None:
            for role, text, sources in cached:
                self._append_message(role, text, sources)
            return

        class_name, lecture_name = scope
        if class_name is not None and lecture_name is None:
            for entry in self.library.class_chat_history(class_name, limit=25):
                self._append_message("user", entry["question"])
                self._append_message("assistant", entry["answer"], entry["sources"])
        if self.message_widgets:
            return
        if lecture_name:
            introduction = (
                f"This conversation is scoped to {class_name} · {lecture_name}. "
                "Ask a question or find a page; answers include citations to the selected lecture."
            )
        elif class_name:
            introduction = (
                f"This conversation is scoped to {class_name}. "
                "Your class questions stay together here, with clickable citations to their source pages."
            )
        else:
            introduction = (
                "Hi! I can help you study from your lecture PDFs. Choose a course on the left, then ask a question "
                "or switch to Find a page. Every answer is grounded in retrieved pages and includes citations you can open."
            )
        self._append_message("assistant", introduction)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt method name
        super().resizeEvent(event)
        if hasattr(self, "message_widgets") and hasattr(self, "chat_scroll"):
            QTimer.singleShot(0, self._refresh_message_layout)

    def _refresh_message_layout(self) -> None:
        available = max(280, self.chat_scroll.viewport().width() - 64)
        for message in self.message_widgets:
            message.set_available_width(min(900 if message.role == "assistant" else 720, available))
            message.body.setMinimumHeight(0)
            message.body.setMinimumHeight(max(0, message.body.heightForWidth(message.body.width())))
            message.body.updateGeometry()
            message.layout().invalidate()
            message.updateGeometry()
        self.chat_layout.invalidate()

    def update_composer_mode(self, *_args) -> None:
        find_mode = self.mode_combo.currentData() == "find"
        self.send_button.setText("Find" if find_mode else "Ask")
        self.query.setPlaceholderText(
            "What topic or phrase should I find in your PDFs?" if find_mode
            else "Ask something about your lectures…"
        )
        self.semantic_search.setVisible(find_mode)
        self.composer_help.setText(
            "Keyword search stays on this Mac. Meaning search uses Gemini and indexed page images."
            if find_mode
            else "Ask uses Gemini with indexed lecture pages. Every factual answer includes clickable citations."
        )

    def send_message(self) -> None:
        if self.mode_combo.currentData() == "find":
            self.search()
        else:
            self.answer_question()

    def toggle_preview(self, visible: bool) -> None:
        self.preview_panel.setVisible(visible)
        self.preview_toggle.setText("Hide page preview" if visible else "Show page preview")
        total_width = max(720, self.content_splitter.width())
        if visible:
            preview_width = min(420, max(300, total_width // 3))
            self.content_splitter.setSizes([total_width - preview_width, preview_width])
        else:
            self.content_splitter.setSizes([total_width, 0])
        QTimer.singleShot(0, self._refresh_message_layout)

    def open_source_page(self, row: dict) -> None:
        self.preview_toggle.setChecked(True)
        self.display_page(row)

    def refresh_navigation(self, select_scope: tuple[str | None, str | None] | None = None) -> None:
        if select_scope is None:
            select_scope = getattr(self, "current_scope", (None, None))
        self.navigator.clear()
        all_item = QTreeWidgetItem(["All materials"])
        all_item.setData(0, Qt.ItemDataRole.UserRole, (None, None))
        self.navigator.addTopLevelItem(all_item)
        classes: dict[str, QTreeWidgetItem] = {}
        for document in _group_lectures(self.library.documents()):
            class_name, lecture_name = document["class_name"], document["lecture_name"]
            parent = classes.get(class_name)
            if parent is None:
                parent = QTreeWidgetItem([class_name])
                parent.setData(0, Qt.ItemDataRole.UserRole, (class_name, None))
                self.navigator.addTopLevelItem(parent)
                classes[class_name] = parent
            page_word = "page" if document["page_count"] == 1 else "pages"
            child = QTreeWidgetItem([f"{lecture_name} · {document['page_count']} {page_word}"])
            child.setData(0, Qt.ItemDataRole.UserRole, (class_name, lecture_name))
            parent.addChild(child)
        self.navigator.expandAll()
        if self.navigator.topLevelItemCount():
            selected_item = all_item
            target_class, target_lecture = select_scope
            if target_class is not None:
                for index in range(self.navigator.topLevelItemCount()):
                    class_item = self.navigator.topLevelItem(index)
                    if class_item.data(0, Qt.ItemDataRole.UserRole) == (target_class, None):
                        selected_item = class_item
                        if target_lecture is not None:
                            for child_index in range(class_item.childCount()):
                                child = class_item.child(child_index)
                                if child.data(0, Qt.ItemDataRole.UserRole) == (target_class, target_lecture):
                                    selected_item = child
                                    break
                        break
            self.navigator.setCurrentItem(selected_item)
        self.update_library_status()

    def update_library_status(self) -> None:
        status = self.library.indexing_status(EMBEDDING_MODEL)
        total = status["total_pages"]
        indexed = status["indexed_pages"]
        if total == 0:
            self.library_status.setText("No PDFs yet. Add a lecture PDF to get started.")
            return
        page_word = "page" if total == 1 else "pages"
        self.library_status.setText(
            f"Library · {total} {page_word}\nAI indexed: {indexed}/{total} · "
            f"visual checks left: {status['pending_visual_pages']}"
        )

    def change_scope(self, current: QTreeWidgetItem | None, _previous: QTreeWidgetItem | None) -> None:
        if current is None:
            return
        new_scope = current.data(0, Qt.ItemDataRole.UserRole)
        if new_scope != self.current_scope:
            if self.message_widgets:
                self._conversation_cache[self.current_scope] = [
                    (message.role, message.text, message.sources)
                    for message in self.message_widgets
                ]
            self._clear_chat()
            self.current_scope = new_scope
            self.settings.setValue("study_space/class", new_scope[0] or "")
            self.settings.setValue("study_space/lecture", new_scope[1] or "")
            self._begin_scope_conversation(new_scope)
        class_name, lecture_name = self.current_scope
        if lecture_name:
            self.scope_title.setText(f"{class_name} · {lecture_name}")
            self.scope_subtitle.setText("This chat is scoped to the selected lecture. Answers include clickable source citations.")
        elif class_name:
            self.scope_title.setText(class_name)
            self.scope_subtitle.setText("This chat is scoped to this class. Answers include clickable source citations.")
        else:
            self.scope_title.setText("All materials")
            self.scope_subtitle.setText("Ask a question or find a page in your PDFs. Answers cite the source pages they use.")
        self.history_button.setEnabled(class_name is not None)

    def open_class_history(self) -> None:
        class_name = self.current_scope[0]
        if not class_name:
            return
        history = self.library.class_chat_history(class_name)
        if not history:
            QMessageBox.information(self, "No saved questions", f"No previous Q&A for {class_name} yet.")
            return
        dialog = ClassHistoryDialog(class_name, history, self)
        dialog.source_opened.connect(self.open_source_page)
        dialog.exec()

    def _key_status(self) -> str:
        return (
            "AI features are ready. Relevant PDF text and images are sent to Gemini when you use them."
            if get_api_key()
            else "Keyword search works locally. Add an API key for AI answers, quizzes, and meaning search."
        )

    def configure_api_key(self) -> None:
        dialog = ApiKeyDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.api_status.setText(self._key_status())
            self.statusBar().showMessage(self._key_status(), 10000)

    def import_pdf(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Choose lecture PDF", str(Path.home()), "PDF files (*.pdf)")
        if not filename:
            return
        selected_class = self.current_scope[0]
        details = ImportDetailsDialog(filename, self, selected_class or "")
        if details.exec() != QDialog.DialogCode.Accepted:
            return
        class_name = details.class_name.text().strip()
        lecture_name = details.lecture_name.text().strip()
        if not class_name or not lecture_name:
            QMessageBox.warning(self, "Missing details", "Enter both a class and lecture name.")
            return
        path = Path(filename)
        database = self.library.path

        def import_pages(progress: Callable[[str], None]) -> dict[str, int | str]:
            progress(f"Reading pages from {path.name}…")
            page_texts = extract_page_texts(path)
            progress(f"Saving {len(page_texts)} page(s) to your local library…")
            library = Library(database)
            try:
                library.import_pdf(class_name, lecture_name, path, page_texts)
            finally:
                library.close()
            return {
                "filename": path.name,
                "page_count": len(page_texts),
                "empty_pages": sum(not text for text in page_texts),
            }

        def imported(result: object) -> None:
            self.refresh_navigation((class_name, lecture_name))
            empty_pages = result["empty_pages"]
            note = (
                f"{empty_pages} page(s) have no embedded text; page images remain available to AI search."
                if empty_pages else "All pages have embedded text."
            )
            self.statusBar().showMessage(
                f"Imported {result['filename']}: {result['page_count']} pages. {note}", 12000,
            )

        self._run_task(
            import_pages,
            imported,
            on_error=lambda error: QMessageBox.critical(self, "Could not import PDF", error),
        )

    def _run_task(
        self,
        action: Callable[[Callable[[str], None]], object],
        on_success: Callable[[object], None],
        on_error: Callable[[str], None] | None = None,
    ) -> bool:
        if self.active_task:
            QMessageBox.information(self, "Still working", "Wait for the current Gemini operation to finish.")
            return False
        task = BackgroundTask(action)
        self.active_task = task
        self.task_success_handler = on_success
        self.task_error_handler = on_error
        for control in self.task_controls:
            control.setEnabled(False)
        task.progress.connect(self.statusBar().showMessage)
        task.finished.connect(self._task_finished)
        task.start()
        return True

    def _task_finished(self) -> None:
        task = self.active_task
        self.active_task = None
        for control in self.task_controls:
            control.setEnabled(True)
        self.history_button.setEnabled(self.current_scope[0] is not None)
        handler = self.task_success_handler
        error_handler = self.task_error_handler
        self.task_success_handler = None
        self.task_error_handler = None
        if task is None:
            return
        task.deleteLater()
        if self.closing:
            return
        self.update_library_status()
        if task.error is not None:
            self.statusBar().showMessage("The operation could not be completed.", 8000)
            if error_handler is not None:
                error_handler(task.error)
            else:
                self._append_message("assistant", f"I couldn't complete that operation.\n\n{task.error}")
        elif handler is not None:
            handler(task.result)

    def index_library(self) -> None:
        if not get_api_key():
            self.configure_api_key()
            if not get_api_key():
                return
        database = self.library.path

        def index(progress: Callable[[str], None]) -> dict[str, int]:
            library = Library(database)
            try:
                pages_to_scan = [dict(row) for row in library.pages_needing_visual_scan()]
                notes_checked = 0
                if pages_to_scan:
                    service = GeminiService()
                    progress(
                        f"Reading handwriting and visual labels on {len(pages_to_scan)} page(s)… "
                        "This may take a while the first time."
                    )
                    notes_checked = service.read_visual_notes_batch(
                        pages_to_scan, library.save_visual_text, progress,
                    )
                pages = [dict(row) for row in library.pages_to_index(EMBEDDING_MODEL)]
                if not pages:
                    return {"indexed": 0, "notes_checked": notes_checked}
                if not pages_to_scan:
                    service = GeminiService()
                progress(f"Creating searchable embeddings for {len(pages)} page(s)…")
                indexed = service.index_pages(pages, library.save_embedding, progress)
                return {"indexed": indexed, "notes_checked": notes_checked}
            finally:
                library.close()

        def indexed(result: object) -> None:
            if result["indexed"] or result["notes_checked"]:
                summary = f"Indexed {result['indexed']} page(s); checked {result['notes_checked']} page(s) for handwriting."
            else:
                summary = "Your PDFs are already indexed and visual notes have already been checked."
            self.statusBar().showMessage(summary, 12000)

        self._run_task(index, indexed)

    def search(self) -> None:
        query = self.query.text().strip()
        if not query:
            return
        class_name, lecture_name = self.current_scope
        if self.semantic_search.isChecked():
            if not get_api_key():
                self.configure_api_key()
                if not get_api_key():
                    return
        self._append_message("user", query)
        self.query.clear()
        find = query

        def display_matches(rows: object) -> None:
            self.show_search_results(rows, find)

        if self.semantic_search.isChecked():
            database = self.library.path

            def semantic_search(progress: Callable[[str], None]) -> list[dict]:
                progress("Searching your PDFs by meaning…")
                embedding = GeminiService().embed_text(find)
                library = Library(database)
                try:
                    return library.search(
                        find, query_embedding=embedding, model=EMBEDDING_MODEL, limit=12,
                        class_name=class_name, lecture_name=lecture_name,
                    )
                finally:
                    library.close()

            self._run_task(semantic_search, display_matches)
            return
        try:
            rows = self.library.search(find, limit=12, class_name=class_name, lecture_name=lecture_name)
        except Exception as error:
            self._append_message("assistant", f"I couldn't search those terms: {error}")
            return
        display_matches(rows)

    def show_search_results(self, rows: object, query: str = "") -> None:
        matches = [dict(row) for row in rows]
        if matches:
            text = f"I found {len(matches)} relevant page(s) for “{query}”. Select a citation below to open that page."
        else:
            text = (
                f"I couldn't find a page matching “{query}”. Try different wording, or turn on Search by meaning "
                "to look for related concepts."
            )
        self._append_message("assistant", text, matches)
        self.statusBar().showMessage(f"Found {len(matches)} matching page(s).", 5000)

    def display_page(self, row: dict) -> None:
        self.current_result = row
        path = Path(row["file_path"])
        page_number = row["page_number"]
        self.preview_title.setText(
            f"{row['class_name']} · {row['lecture_name']} · page {page_number}\n{path.name}"
        )
        body = row["text"].strip() or "No searchable embedded text on this page."
        visual_text = row.get("visual_text", "").strip()
        if visual_text:
            body += "\n\nAI-read handwriting / visual notes (verify against the page):\n" + visual_text
        self.excerpt.setPlainText(body)
        try:
            pixmap = QPixmap()
            pixmap.loadFromData(render_page(path, page_number))
            self.preview.setPixmap(pixmap.scaled(
                self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
        except (OSError, ValueError, RuntimeError) as error:
            self.preview.setText(f"Could not render page: {error}")

    def answer_question(self) -> None:
        question = self.query.text().strip()
        if not question:
            return
        if not get_api_key():
            self.configure_api_key()
            if not get_api_key():
                return
        self._append_message("user", question)
        self.query.clear()
        database = self.library.path
        class_name, lecture_name = self.current_scope

        def answer(progress: Callable[[str], None]) -> dict:
            service = GeminiService()
            progress("Finding relevant pages…")
            query_vector = service.embed_text(question, task="question answering")
            library = Library(database)
            try:
                sources = [dict(row) for row in library.search(
                    question, query_embedding=query_vector, model=EMBEDDING_MODEL, limit=20,
                    class_name=class_name, lecture_name=lecture_name,
                )[:5]]
                for index, source in enumerate(sources, start=1):
                    source["source_id"] = f"S{index}"
                if not sources:
                    result = {
                        "answer": "I couldn't find matching pages in the indexed course material, so I can't answer from your sources yet.",
                        "sources": [],
                    }
                else:
                    progress("Writing an answer from the retrieved pages…")
                    answer_result = service.answer_grounded(question, sources)
                    used_ids = set(answer_result["used_source_ids"])
                    visible_sources = [source for source in sources if not answer_result["answerable"] or source["source_id"] in used_ids]
                    result = {"answer": answer_result["answer"], "sources": visible_sources}
                if class_name:
                    library.save_class_chat(class_name, question, result["answer"], result["sources"])
                return result
            finally:
                library.close()

        def show_answer(result: object) -> None:
            sources = result["sources"]
            self._append_message("assistant", result["answer"], sources)

        self._run_task(answer, show_answer)

    def explain_more(self, passage: str, sources: list[dict]) -> None:
        if not sources:
            return
        self._append_message("user", f"Can you explain this more deeply?\n\n{passage}")

        def explain(progress: Callable[[str], None]) -> dict:
            progress("Preparing a deeper explanation from the cited pages…")
            service = GeminiService()
            question = f"Explain this selected answer passage in greater depth, using only the supplied course sources:\n{passage}"
            result = service.answer_grounded(question, sources)
            used_ids = set(result["used_source_ids"])
            visible_sources = [source for source in sources if not result["answerable"] or source["source_id"] in used_ids]
            return {"answer": result["answer"], "sources": visible_sources}

        def show_explanation(result: object) -> None:
            self._append_message("assistant", result["answer"], result["sources"])

        self._run_task(explain, show_explanation)

    def start_quiz(self) -> None:
        if not get_api_key():
            self.configure_api_key()
            if not get_api_key():
                return
        documents = [dict(row) for row in self.library.documents()]
        if not documents:
            QMessageBox.information(self, "No lectures yet", "Import a PDF before building a quiz.")
            return
        setup = QuizSetupDialog(documents, self)
        if setup.exec() == QDialog.DialogCode.Accepted:
            self.generate_quiz(setup.scopes, setup.question_count.value(), setup.question_type.currentData())

    def generate_quiz(self, scopes: list[tuple[str, str]], count: int, mode: str, focus_topic: str | None = None) -> None:
        if not get_api_key():
            self.configure_api_key()
            if not get_api_key():
                return
        database = self.library.path

        def generate(progress: Callable[[str], None]) -> dict:
            library = Library(database)
            try:
                progress("Selecting source pages from the chosen lectures…")
                sources = library.pages_for_scopes(scopes, limit=16)
                if not sources:
                    return {"questions": [], "sources": []}
                for index, source in enumerate(sources, start=1):
                    source["source_id"] = f"S{index}"
                progress("Writing practice questions and answer rubrics…")
                generated = GeminiService().generate_quiz(count, mode, sources, focus_topic=focus_topic)
                saved = library.create_quiz_session(scopes, generated)
                return {"questions": saved, "sources": sources}
            finally:
                library.close()

        def show_quiz(result: object) -> None:
            if not result["questions"]:
                QMessageBox.information(self, "No source pages", "The selected lectures have no pages available to quiz.")
                return
            dialog = QuizRunDialog(result["questions"], self)
            dialog.answer_submitted.connect(lambda question, answer: self.grade_quiz_answer(dialog, question, answer))
            dialog.exec()
            self.statusBar().showMessage("Quiz finished. Weak-topic scores are updated in SQLite.", 10000)

        self._run_task(generate, show_quiz)

    @staticmethod
    def _citation_labels(question: dict, sources: list[dict]) -> str:
        labels = []
        source_map = {source["source_id"]: source for source in sources}
        for source_id in question["source_ids"]:
            source = source_map[source_id]
            labels.append(
                f"{source['class_name']} · {source['lecture_name']} · "
                f"{Path(source['file_path']).name} · p. {source['page_number']}"
            )
        return "; ".join(dict.fromkeys(labels))

    def grade_quiz_answer(self, dialog: QuizRunDialog, question: dict, answer: str) -> None:
        sources = [question["source_map"][source_id] for source_id in question["source_ids"]]
        citations = self._citation_labels(question, sources)
        if question["question_type"] == "multiple_choice":
            correct = answer == question["correct_answer"]
            score = 1.0 if correct else 0.0
            feedback = "Your choice is correct." if correct else f"That choice is incorrect. The cited explanation below shows the key idea."
            self.library.record_quiz_attempt(question["id"], answer, score, correct, feedback)
            dialog.show_grade({
                "score": score,
                "is_correct": correct,
                "feedback": feedback,
                "explanation_citations": f"[{citations}]",
            })
            return

        database = self.library.path

        def grade(progress: Callable[[str], None]) -> dict:
            progress("Grading against the answer rubric and cited lecture pages…")
            result = GeminiService().grade_short_answer(question, answer, sources)
            feedback = result["feedback"]
            library = Library(database)
            try:
                library.record_quiz_attempt(question["id"], answer, result["score"], result["is_correct"], feedback)
            finally:
                library.close()
            return {
                **result,
                "explanation_citations": f"[{citations}]",
            }

        dialog.set_grading(True)
        if not self._run_task(grade, dialog.show_grade, on_error=dialog.grading_failed):
            dialog.grading_failed()

    def open_weak_spots(self) -> None:
        dialog = WeakSpotsDialog(self.library.topic_performance(), self)
        dialog.review_requested.connect(self.generate_topic_review)
        dialog.exec()

    def generate_topic_review(self, topic: str) -> None:
        if not get_api_key():
            self.configure_api_key()
            if not get_api_key():
                return
        database = self.library.path

        def retrieve_and_generate(progress: Callable[[str], None]) -> dict:
            service = GeminiService()
            progress(f"Finding source pages for {topic}…")
            vector = service.embed_text(topic)
            library = Library(database)
            try:
                sources = [dict(row) for row in library.search(
                    topic, query_embedding=vector, model=EMBEDDING_MODEL, limit=16
                )]
                if not sources:
                    return {"questions": [], "sources": []}
                for index, source in enumerate(sources, start=1):
                    source["source_id"] = f"S{index}"
                scopes = sorted({(source["class_name"], source["lecture_name"]) for source in sources})
                progress(f"Building a targeted review quiz for {topic}…")
                generated = service.generate_quiz(3, "mixed", sources, focus_topic=topic)
                saved = library.create_quiz_session(scopes, generated)
                return {"questions": saved, "sources": sources}
            finally:
                library.close()

        def show_review(result: object) -> None:
            if not result["questions"]:
                QMessageBox.information(self, "No matching pages", f"I couldn't find course pages for {topic}.")
                return
            dialog = QuizRunDialog(result["questions"], self)
            dialog.answer_submitted.connect(lambda question, answer: self.grade_quiz_answer(dialog, question, answer))
            dialog.exec()

        self._run_task(retrieve_and_generate, show_review)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt method name
        self.closing = True
        if self.active_task and self.active_task.isRunning():
            self.statusBar().showMessage("Waiting for the current Gemini request to finish before closing…")
            self.active_task.wait()
        self.library.close()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
