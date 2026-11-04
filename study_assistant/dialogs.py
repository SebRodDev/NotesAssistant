"""Dialogs: PDF import, quizzes, history, agent editor, and local model settings."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
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

from .ollama import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VISION_MODEL,
    NOTES_MODES,
    OllamaClient,
    OllamaError,
)
from .theme import ACCENTS, creativity_label
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

def is_embedding_model(name: str) -> bool:
    lowered = name.casefold()
    return any(key in lowered for key in ("embed", "bge-", "minilm", "arctic-embed"))


class AgentEditorDialog(QDialog):
    """Create or edit a custom agent: model, creativity, notes access, persona."""

    def __init__(
        self,
        agent: dict | None,
        installed_models: list[str],
        default_model: str,
        parent: QWidget | None = None,
        can_delete: bool = True,
    ) -> None:
        super().__init__(parent)
        self.agent = dict(agent or {})
        self.deleted = False
        self.setWindowTitle("Configure agent" if agent else "New agent")
        self.resize(560, 640)

        accent = self.agent.get("accent") or ACCENTS[0]
        self.preview = AgentAvatar(self.agent.get("name", "New agent"), accent, 72)
        self.name_input = QLineEdit(self.agent.get("name", ""))
        self.name_input.setPlaceholderText("e.g. CS 354 Tutor")
        self.name_input.textChanged.connect(self._refresh_preview)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItem(f"Default ({default_model})", "")
        for name in installed_models:
            if not is_embedding_model(name):
                self.model_combo.addItem(name, name)
        current_model = self.agent.get("model", "")
        index = self.model_combo.findData(current_model)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        elif current_model:
            self.model_combo.setEditText(current_model)

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
                f"QPushButton {{ background: {color}; border: 2px solid #07070a; padding: 0; }}"
                "QPushButton:checked { border: 2px solid #f4f4f8; }"
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
        form.addRow("Model", self.model_combo)
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
        model = self.model_combo.currentData()
        typed = self.model_combo.currentText().strip()
        if self.model_combo.findText(typed) < 0:  # a model name typed by hand
            model = typed
        return {
            "id": self.agent.get("id"),
            "name": self.name_input.text().strip(),
            "model": model or "",
            "temperature": self.creativity.value() / 100,
            "notes_mode": self.notes_mode.currentData(),
            "reasoning": self.reasoning.isChecked(),
            "accent": self._accent(),
            "system_prompt": self.system_prompt.toPlainText().strip(),
        }


class ModelSettingsDialog(QDialog):
    """Pick the Ollama server and the default chat, embedding, and vision models."""

    def __init__(self, values: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Local models")
        self.resize(600, 520)
        self.host_input = QLineEdit(values.get("host", ""))
        self.chat_combo = QComboBox()
        self.embed_combo = QComboBox()
        self.vision_combo = QComboBox()
        for combo in (self.chat_combo, self.embed_combo, self.vision_combo):
            combo.setEditable(True)
        self._values = values
        self.status = QLabel()
        self.status.setObjectName("onlineStatus")
        self.status.setWordWrap(True)
        refresh = QPushButton("REFRESH")
        refresh.clicked.connect(self.refresh_models)

        help_text = QLabel(
            "Everything runs on this Mac through Ollama — no API keys, nothing uploaded.\n\n"
            "Recommended starting set (run in Terminal):\n"
            f"  ollama pull {DEFAULT_CHAT_MODEL}          chat & quizzes\n"
            f"  ollama pull {DEFAULT_EMBEDDING_MODEL}   search by meaning\n"
            f"  ollama pull {DEFAULT_VISION_MODEL}      handwriting (optional)\n\n"
            "On a 16 GB Mac, 7–9B models are the sweet spot. With 8 GB, try qwen3:4b and gemma3:4b. "
            "Changing the embedding model re-indexes your pages the next time you press Index."
        )
        help_text.setObjectName("dialogHelp")
        help_text.setWordWrap(True)
        help_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        form = QFormLayout()
        form.setVerticalSpacing(12)
        host_row = QHBoxLayout()
        host_row.addWidget(self.host_input, 1)
        host_row.addWidget(refresh)
        form.addRow("Ollama server", host_row)
        form.addRow("", self.status)
        form.addRow("Default chat model", self.chat_combo)
        form.addRow("Embedding model", self.embed_combo)
        form.addRow("Vision model", self.vision_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.addLayout(form)
        layout.addWidget(help_text, 1)
        layout.addWidget(buttons)
        self.refresh_models()

    def refresh_models(self) -> None:
        host = self.host_input.text().strip()
        try:
            client = OllamaClient(host)
            names = sorted(model["name"] for model in client.list_models(timeout=2))
            self.status.setText(f"◆ ONLINE · {len(names)} MODEL{'S' if len(names) != 1 else ''} INSTALLED")
            self.status.setStyleSheet("color: #52e3a4;")
        except OllamaError as error:
            names = []
            self.status.setText(f"◇ OFFLINE · {error}")
            self.status.setStyleSheet("color: #ff5d73;")
        chats = [n for n in names if not is_embedding_model(n)]
        embeds = [n for n in names if is_embedding_model(n)]
        self._fill(self.chat_combo, chats, self.chat_combo.currentText() or self._values.get("chat_model", DEFAULT_CHAT_MODEL))
        self._fill(self.embed_combo, embeds, self.embed_combo.currentText() or self._values.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
        vision_current = self.vision_combo.currentText() if self.vision_combo.count() else (
            self._values.get("vision_model") or "None")
        self._fill(self.vision_combo, ["None"] + chats, vision_current)

    @staticmethod
    def _fill(combo: QComboBox, names: list[str], current: str) -> None:
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(names)
        if current and combo.findText(current) < 0:
            combo.addItem(current)
        combo.setCurrentText(current)
        combo.blockSignals(False)

    def values(self) -> dict:
        vision = self.vision_combo.currentText().strip()
        return {
            "host": self.host_input.text().strip(),
            "chat_model": self.chat_combo.currentText().strip() or DEFAULT_CHAT_MODEL,
            "embedding_model": self.embed_combo.currentText().strip() or DEFAULT_EMBEDDING_MODEL,
            "vision_model": "" if vision.casefold() in {"", "none"} else vision,
        }
