"""Cortex — an offline, Ollama-powered chat interface for studying your lecture notes."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QThread, QTime, QTimer, Signal
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTabWidget,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .dialogs import (
    AgentEditorDialog,
    BatchImportDetailsDialog,
    ClassHistoryDialog,
    ImportDetailsDialog,
    ModelSettingsDialog,
    QuizRunDialog,
    QuizSetupDialog,
    WeakSpotsDialog,
    _group_lectures,
)
from .ollama import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_HOST,
    DEFAULT_VISION_MODEL,
    OllamaClient,
    OllamaError,
    OllamaUnavailable,
    StudyModels,
    describe_runtime,
)
from .pdf_reader import extract_page_texts, render_page
from .storage import Library
from .theme import APP_NAME, APP_TAGLINE, COLORS, creativity_label, stylesheet
from .widgets import AgentAvatar, AgentCard, ChatMessageWidget, ComposerInput, MeterBar

CONTEXT_WINDOW = 16384
RETRIEVED_PAGES = 6


# ---------------------------------------------------------------------------------------------
# Background work
# ---------------------------------------------------------------------------------------------

class TaskReporter:
    """Passed to background actions: call it for status text, or use token/think/event."""

    def __init__(self, task: "BackgroundTask") -> None:
        self._task = task

    def __call__(self, message: str) -> None:
        self._task.progress.emit(message)

    def token(self, text: str) -> None:
        self._task.chunk.emit(text)

    def think(self, text: str) -> None:
        self._task.thinking.emit(text)

    def event(self, payload: dict) -> None:
        self._task.event.emit(payload)


class BackgroundTask(QThread):
    progress = Signal(str)
    chunk = Signal(str)
    thinking = Signal(str)
    event = Signal(object)

    def __init__(self, action: Callable[[TaskReporter], object]) -> None:
        super().__init__()
        self.action = action
        self.result: object | None = None
        self.error: str | None = None
        self.exception: Exception | None = None

    def run(self) -> None:
        try:
            self.result = self.action(TaskReporter(self))
        except Exception as error:  # shown in the app rather than killing the UI
            self.exception = error
            self.error = str(error)


class OllamaMonitor(QThread):
    """Polls the local Ollama server so the UI can show online state, models, and GPU use."""

    status = Signal(object)

    def __init__(self, host: str, interval: float = 4.0) -> None:
        super().__init__()
        self.host = host
        self.interval = interval
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        while not self._stop:
            client = OllamaClient(self.host)
            try:
                models = client.list_models(timeout=2)
                running = client.running_models(timeout=2)
                self.status.emit({"online": True, "models": models, "running": running})
            except OllamaError as error:
                self.status.emit({"online": False, "models": [], "running": [], "error": str(error)})
            waited = 0.0
            while waited < self.interval and not self._stop:
                time.sleep(0.1)
                waited += 0.1


# ---------------------------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------------------------

def _mono_label(text: str, name: str = "monoMuted") -> QLabel:
    label = QLabel(text)
    label.setObjectName(name)
    return label


class MainWindow(QMainWindow):
    def __init__(
        self,
        library: Library | None = None,
        settings: QSettings | None = None,
        start_monitor: bool = True,
    ) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1440, 900)
        self.setMinimumSize(980, 600)
        self.library = library or Library()
        self.settings = settings or QSettings("Purdue", "StudyAssistant")
        self.active_task: BackgroundTask | None = None
        self.task_success_handler: Callable[[object], None] | None = None
        self.task_error_handler: Callable[[str], None] | None = None
        self.closing = False
        self.current_result: dict | None = None
        self.current_scope: tuple[str | None, str | None] = (None, None)
        self.message_widgets: list[ChatMessageWidget] = []
        self._conversation_cache: dict[tuple, list[dict]] = {}
        self._streaming_message: ChatMessageWidget | None = None
        self._pending_tokens: list[str] = []
        self.ollama_online: bool | None = None
        self.installed_models: list[str] = []
        self.running_models: list[dict] = []
        self.monitor: OllamaMonitor | None = None

        agents = self.library.agents()
        saved_agent = self.settings.value("agents/current", 0, type=int)
        self.current_agent = next((a for a in agents if a["id"] == saved_agent), agents[0])

        self._build_sidebar()
        self._build_center()
        self._build_right_panel()

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self.sidebar_scroll)
        root_layout.addWidget(self.center_panel, 1)
        root_layout.addWidget(self.right_panel)
        self.setCentralWidget(root)

        self.task_controls = [
            self.navigator, self.query, self.ask_mode_button, self.find_mode_button, self.semantic_search,
            self.send_button, self.import_button, self.index_button, self.quiz_button, self.weak_spots_button,
            self.history_button, self.new_agent_button, self.config_button, self.clear_button,
            self.models_button, self.agent_host,
        ]

        self._token_timer = QTimer(self)
        self._token_timer.setInterval(45)
        self._token_timer.timeout.connect(self._flush_tokens)
        self._clock = QTimer(self)
        self._clock.timeout.connect(lambda: self.clock_label.setText(QTime.currentTime().toString("HH:mm:ss")))
        self._clock.start(1000)
        self.clock_label.setText(QTime.currentTime().toString("HH:mm:ss"))

        self.refresh_agents()
        self._apply_agent_to_ui()
        saved_class = self.settings.value("study_space/class", "", type=str).strip()
        saved_lecture = self.settings.value("study_space/lecture", "", type=str).strip()
        self.refresh_navigation((saved_class or None, saved_lecture or None))
        if not self.message_widgets:
            self._begin_conversation()
        self.set_mode("ask")
        self._set_online_status(None)
        if start_monitor:
            self._start_monitor()
        QShortcut(QKeySequence("Ctrl+K"), self, activated=lambda: self.query.setFocus())

    # -- settings ------------------------------------------------------------------------------
    def model_settings(self) -> dict:
        return {
            "host": self.settings.value("models/host", DEFAULT_HOST, type=str) or DEFAULT_HOST,
            "chat_model": self.settings.value("models/chat", DEFAULT_CHAT_MODEL, type=str) or DEFAULT_CHAT_MODEL,
            "embedding_model": self.settings.value("models/embedding", DEFAULT_EMBEDDING_MODEL, type=str)
            or DEFAULT_EMBEDDING_MODEL,
            "vision_model": self.settings.value("models/vision", DEFAULT_VISION_MODEL, type=str),
        }

    def _models(self) -> StudyModels:
        values = self.model_settings()
        return StudyModels(
            OllamaClient(values["host"]),
            chat_model=values["chat_model"],
            embedding_model=values["embedding_model"],
            vision_model=values["vision_model"] or None,
        )

    def agent_model(self, agent: dict | None = None) -> str:
        agent = agent or self.current_agent
        return agent.get("model") or self.model_settings()["chat_model"]

    # -- layout: sidebar -------------------------------------------------------------------------
    def _build_sidebar(self) -> None:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(14, 18, 14, 12)
        layout.setSpacing(8)

        top = QHBoxLayout()
        rule = QFrame()
        rule.setObjectName("brandRule")
        rule.setFixedSize(22, 2)
        top.addWidget(rule, 0, Qt.AlignmentFlag.AlignVCenter)
        top.addStretch(1)
        self.gear_button = QPushButton("✳")
        self.gear_button.setObjectName("ghostButton")
        self.gear_button.setToolTip("Local model settings")
        self.gear_button.clicked.connect(self.configure_models)
        top.addWidget(self.gear_button)
        layout.addLayout(top)
        brand = QLabel(APP_NAME.upper())
        brand.setObjectName("brandTitle")
        layout.addWidget(brand)
        layout.addWidget(_mono_label(APP_TAGLINE, "brandTag"))
        self.online_label = _mono_label("", "onlineStatus")
        layout.addWidget(self.online_label)
        layout.addSpacing(12)

        agents_row = QHBoxLayout()
        agents_row.addWidget(_mono_label("AGENTS", "sectionHeading"))
        agents_row.addStretch(1)
        self.new_agent_button = QPushButton("+ NEW")
        self.new_agent_button.clicked.connect(self.create_agent)
        agents_row.addWidget(self.new_agent_button)
        layout.addLayout(agents_row)
        self.agent_host = QWidget()
        self.agent_list = QVBoxLayout(self.agent_host)
        self.agent_list.setContentsMargins(0, 0, 0, 0)
        self.agent_list.setSpacing(4)
        layout.addWidget(self.agent_host)
        self.agent_cards: list[AgentCard] = []
        layout.addSpacing(8)

        # Courses tab
        self.navigator = QTreeWidget()
        self.navigator.setObjectName("courseNavigator")
        self.navigator.setHeaderHidden(True)
        self.navigator.setMinimumHeight(170)
        self.navigator.setIndentation(14)
        self.navigator.currentItemChanged.connect(self.change_scope)
        self.import_button = QPushButton("+ ADD LECTURE PDFS")
        self.import_button.setShortcut(QKeySequence.StandardKey.Open)
        self.import_button.setToolTip("Add lecture PDFs (Command+O)")
        self.import_button.clicked.connect(self.import_pdf)
        self.library_status = _mono_label("", "monoMuted")
        self.library_status.setWordWrap(True)
        self.index_button = QPushButton("⟳ INDEX FOR SEARCH")
        self.index_button.setToolTip(
            "Embed every page with your local embedding model (and read handwriting if a vision model is set). "
            "Nothing leaves this Mac."
        )
        self.index_button.clicked.connect(self.index_library)
        courses = QWidget()
        courses_layout = QVBoxLayout(courses)
        courses_layout.setContentsMargins(0, 10, 0, 0)
        courses_layout.setSpacing(8)
        courses_layout.addWidget(self.import_button)
        courses_layout.addWidget(self.navigator, 1)
        courses_layout.addWidget(self.library_status)
        courses_layout.addWidget(self.index_button)

        # Study tab
        self.quiz_button = QPushButton("PRACTICE QUIZ…")
        self.quiz_button.clicked.connect(self.start_quiz)
        self.weak_spots_button = QPushButton("WEAK SPOTS")
        self.weak_spots_button.clicked.connect(self.open_weak_spots)
        self.history_button = QPushButton("QUESTION HISTORY")
        self.history_button.clicked.connect(self.open_class_history)
        study = QWidget()
        study_layout = QVBoxLayout(study)
        study_layout.setContentsMargins(0, 10, 0, 0)
        study_layout.setSpacing(8)
        for button in (self.quiz_button, self.weak_spots_button, self.history_button):
            study_layout.addWidget(button)
        study_help = QLabel(
            "Quizzes are written and graded by your default local model from the lectures you pick. "
            "Weak spots ranks topics by recent accuracy."
        )
        study_help.setObjectName("dialogHelp")
        study_help.setWordWrap(True)
        study_layout.addWidget(study_help)
        study_layout.addStretch(1)

        self.side_tabs = QTabWidget()
        self.side_tabs.addTab(courses, "COURSES")
        self.side_tabs.addTab(study, "STUDY")
        layout.addWidget(self.side_tabs, 1)

        self.models_button = QPushButton("⬡  MODELS")
        self.models_button.clicked.connect(self.configure_models)
        layout.addWidget(self.models_button)
        footer = _mono_label("◆ 100% LOCAL · NO CLOUD", "footerNote")
        footer.setToolTip(str(self.library.path))
        layout.addWidget(footer)

        self.sidebar_scroll = QScrollArea()
        self.sidebar_scroll.setObjectName("sidebarScroll")
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.sidebar_scroll.setFixedWidth(272)
        self.sidebar_scroll.setWidget(sidebar)
        sidebar.setMinimumHeight(720)
        sidebar.setMaximumWidth(272)

    # -- layout: center -----------------------------------------------------------------------
    def _stat(self, row: QHBoxLayout, label: str) -> QLabel:
        row.addWidget(_mono_label(label, "statLabel"))
        value = _mono_label("—", "statValue")
        row.addWidget(value)
        row.addSpacing(14)
        return value

    def _build_center(self) -> None:
        self.center_panel = QWidget()
        self.center_panel.setObjectName("centerPanel")
        center = QVBoxLayout(self.center_panel)
        center.setContentsMargins(0, 0, 0, 0)
        center.setSpacing(0)

        header = QFrame()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(22, 14, 22, 14)
        header_layout.setSpacing(14)
        self.header_avatar = AgentAvatar("", "#9b87f5", 50)
        header_layout.addWidget(self.header_avatar)
        info = QVBoxLayout()
        info.setSpacing(5)
        title_row = QHBoxLayout()
        title_row.setSpacing(12)
        self.agent_title = _mono_label("", "agentTitle")
        title_row.addWidget(self.agent_title)
        self.scope_title = _mono_label("ALL MATERIALS", "scopeChip")
        title_row.addWidget(self.scope_title, 0, Qt.AlignmentFlag.AlignVCenter)
        title_row.addStretch(1)
        info.addLayout(title_row)
        row1 = QHBoxLayout()
        row1.setSpacing(5)
        self.stat_model = self._stat(row1, "MODEL")
        self.stat_creativity = self._stat(row1, "CREATIVITY")
        row1.addWidget(_mono_label("SPEED", "statLabel"))
        self.stat_speed = _mono_label("—", "statValue")
        row1.addWidget(self.stat_speed)
        self.speed_meter = MeterBar(44)
        row1.addWidget(self.speed_meter, 0, Qt.AlignmentFlag.AlignVCenter)
        row1.addStretch(1)
        row2 = QHBoxLayout()
        row2.setSpacing(5)
        self.stat_runtime = self._stat(row2, "RUNS ON")
        row2.addWidget(_mono_label("CONTEXT", "statLabel"))
        self.stat_context = _mono_label("—", "statValue")
        row2.addWidget(self.stat_context)
        self.context_meter = MeterBar(44)
        row2.addWidget(self.context_meter, 0, Qt.AlignmentFlag.AlignVCenter)
        row2.addSpacing(14)
        self.stat_notes = self._stat(row2, "NOTES")
        row2.addStretch(1)
        info.addLayout(row1)
        info.addLayout(row2)
        header_layout.addLayout(info, 1)
        self.panel_button = QPushButton("▯ PANEL")
        self.panel_button.setCheckable(True)
        self.panel_button.setChecked(True)
        self.panel_button.toggled.connect(self.toggle_right_panel)
        self.config_button = QPushButton("⚙ CONFIG")
        self.config_button.clicked.connect(self.edit_agent)
        self.clear_button = QPushButton("✕")
        self.clear_button.setToolTip("Clear this conversation")
        self.clear_button.clicked.connect(self.clear_conversation)
        for button in (self.panel_button, self.config_button, self.clear_button):
            header_layout.addWidget(button, 0, Qt.AlignmentFlag.AlignTop)
        center.addWidget(header)

        self.chat_scroll = QScrollArea()
        self.chat_scroll.setObjectName("chatScroll")
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chat_body = QWidget()
        self.chat_body.setObjectName("chatBody")
        self.chat_layout = QVBoxLayout(self.chat_body)
        self.chat_layout.setContentsMargins(48, 26, 48, 26)
        self.chat_layout.setSpacing(22)
        self.chat_layout.addStretch(1)
        self.chat_scroll.setWidget(self.chat_body)
        center.addWidget(self.chat_scroll, 1)

        self.composer = QFrame()
        self.composer.setObjectName("composer")
        composer_layout = QVBoxLayout(self.composer)
        composer_layout.setContentsMargins(14, 10, 12, 10)
        composer_layout.setSpacing(6)
        self.query = ComposerInput()
        self.query.submitted.connect(self.send_message)
        self.query.focus_changed.connect(self._composer_focus)
        composer_layout.addWidget(self.query)
        controls = QHBoxLayout()
        controls.setSpacing(6)
        self.ask_mode_button = QPushButton("ASK")
        self.find_mode_button = QPushButton("FIND PAGE")
        self.mode_group = QButtonGroup(self)
        for mode, button in (("ask", self.ask_mode_button), ("find", self.find_mode_button)):
            button.setObjectName("modeButton")
            button.setCheckable(True)
            button.clicked.connect(lambda _c=False, m=mode: self.set_mode(m))
            self.mode_group.addButton(button)
            controls.addWidget(button)
        self.semantic_search = QCheckBox("BY MEANING")
        self.semantic_search.setObjectName("hintText")
        self.semantic_search.setToolTip("Also rank pages by meaning with your local embedding model.")
        self.semantic_search.setChecked(True)
        controls.addWidget(self.semantic_search)
        controls.addSpacing(8)
        self.composer_help = _mono_label("ENTER SENDS · SHIFT+ENTER NEW LINE", "hintText")
        controls.addWidget(self.composer_help)
        controls.addStretch(1)
        self.send_button = QPushButton("SEND ▸")
        self.send_button.setObjectName("sendButton")
        self.send_button.clicked.connect(self.send_message)
        controls.addWidget(self.send_button)
        composer_layout.addLayout(controls)
        composer_wrap = QVBoxLayout()
        composer_wrap.setContentsMargins(32, 8, 32, 18)
        composer_wrap.addWidget(self.composer)
        center.addLayout(composer_wrap)

    # -- layout: right panel -------------------------------------------------------------------
    def _build_right_panel(self) -> None:
        self.right_panel = QWidget()
        self.right_panel.setObjectName("rightPanel")
        self.right_panel.setFixedWidth(330)
        outer = QVBoxLayout(self.right_panel)
        outer.setContentsMargins(0, 0, 0, 0)
        self.right_stack = QStackedWidget()
        outer.addWidget(self.right_stack)

        link = QWidget()
        layout = QVBoxLayout(link)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(8)
        top = QHBoxLayout()
        top.addWidget(_mono_label("// NEURAL LINK", "panelSubject"))
        top.addStretch(1)
        self.clock_label = _mono_label("", "panelSubject")
        top.addWidget(self.clock_label)
        layout.addLayout(top)
        layout.addSpacing(6)
        self.panel_avatar = AgentAvatar("", "#9b87f5", 270, ring=True)
        layout.addWidget(self.panel_avatar, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addSpacing(4)
        self.panel_subject = _mono_label("", "panelSubject")
        self.panel_name = _mono_label("", "panelName")
        self.panel_state = _mono_label("◆ IDLE", "monoMuted")
        for widget in (self.panel_subject, self.panel_name, self.panel_state):
            widget.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            layout.addWidget(widget)
        dots = _mono_label("·" * 34, "monoMuted")
        dots.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(dots)
        self.panel_prompt = QLabel()
        self.panel_prompt.setObjectName("panelPrompt")
        self.panel_prompt.setWordWrap(True)
        self.panel_prompt.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.panel_prompt)
        layout.addStretch(1)
        self.right_stack.addWidget(link)

        # Page preview (opened from citations)
        page = QWidget()
        self.preview_panel = page
        preview_layout = QVBoxLayout(page)
        preview_layout.setContentsMargins(16, 14, 16, 14)
        preview_layout.setSpacing(8)
        back = QPushButton("◂ BACK TO LINK")
        back.setObjectName("ghostButton")
        back.clicked.connect(lambda: self.right_stack.setCurrentIndex(0))
        preview_layout.addWidget(back, 0, Qt.AlignmentFlag.AlignLeft)
        self.preview_title = _mono_label("", "monoMuted")
        self.preview_title.setWordWrap(True)
        preview_layout.addWidget(self.preview_title)
        self.preview = QLabel("PDF page preview")
        self.preview.setObjectName("pdfPreview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(260)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        preview_layout.addWidget(self.preview, 3)
        self.excerpt = QTextBrowser()
        self.excerpt.setPlaceholderText("Extracted text and machine-read handwriting appear here.")
        preview_layout.addWidget(self.excerpt, 2)
        self.right_stack.addWidget(page)

    # -- appearance & status ---------------------------------------------------------------------
    def _apply_agent_to_ui(self) -> None:
        agent = self.current_agent
        self.setStyleSheet(stylesheet(agent["accent"]))
        name = agent["name"]
        self.agent_title.setText(name.upper())
        self.header_avatar.set_agent(name, agent["accent"])
        self.panel_avatar.set_agent(name, agent["accent"])
        self.panel_name.setText(name.upper())
        self.panel_subject.setText(f"SUBJECT // ID-{agent['id'] * 7919 % 0xFFFFF:05X}")
        self.panel_prompt.setText(agent["system_prompt"] or "No system prompt set.")
        self.stat_model.setText(self.agent_model())
        temperature = float(agent["temperature"])
        self.stat_creativity.setText(f"{temperature:.2f} {creativity_label(temperature)}")
        self.stat_notes.setText({"strict": "STRICT · CITED", "blend": "NOTES + GENERAL", "off": "OFF"}[agent["notes_mode"]])
        self._update_runtime_stat()
        for card in self.agent_cards:
            card.set_active(card.agent_id == agent["id"])
        if hasattr(self, "find_mode_button"):
            self.set_mode(self.current_mode())

    def _set_state(self, text: str) -> None:
        self.panel_state.setText(f"◆ {text}")
        busy = text not in {"IDLE", "OFFLINE"}
        self.panel_avatar.set_busy(busy)

    def _composer_focus(self, focused: bool) -> None:
        self.composer.setProperty("focused", focused)
        self.composer.style().unpolish(self.composer)
        self.composer.style().polish(self.composer)

    def _start_monitor(self) -> None:
        if self.monitor:
            self.monitor.stop()
            self.monitor.wait(2000)
        self.monitor = OllamaMonitor(self.model_settings()["host"])
        self.monitor.status.connect(self._on_ollama_status)
        self.monitor.start()

    def _on_ollama_status(self, status: dict) -> None:
        if self.closing:
            return
        installed = sorted(model["name"] for model in status["models"])
        changed = installed != self.installed_models
        self.installed_models = installed
        self.running_models = status["running"]
        self._set_online_status(status["online"])
        self._update_runtime_stat()
        if changed and not self.active_task:
            self.refresh_agents()

    def _set_online_status(self, online: bool | None) -> None:
        self.ollama_online = online
        if online is None:
            self.online_label.setText("◇ CONNECTING TO OLLAMA…")
            self.online_label.setStyleSheet(f"color: {COLORS['muted']};")
        elif online:
            count = len(self.installed_models)
            self.online_label.setText(f"◆ ONLINE · {count} MODEL{'S' if count != 1 else ''}")
            self.online_label.setStyleSheet(f"color: {COLORS['ok']};")
            if self.panel_state.text() == "◆ OFFLINE":
                self._set_state("IDLE")
        else:
            self.online_label.setText("◇ OFFLINE · OPEN THE OLLAMA APP")
            self.online_label.setStyleSheet(f"color: {COLORS['bad']};")
            if not self.active_task:
                self._set_state("OFFLINE")

    def _update_runtime_stat(self) -> None:
        model = self.agent_model()
        runtime = describe_runtime(self.running_models, model)
        if runtime["loaded"]:
            self.stat_runtime.setText(f"{runtime['where']} · {runtime['memory_gb']:.1f} GB")
        elif self.installed_models and model not in self.installed_models and f"{model}:latest" not in self.installed_models:
            self.stat_runtime.setText("NOT INSTALLED")
        else:
            self.stat_runtime.setText("LOADS ON FIRST USE" if self.ollama_online else "—")

    # -- agents ------------------------------------------------------------------------------
    def refresh_agents(self) -> None:
        for card in self.agent_cards:
            self.agent_list.removeWidget(card)
            card.deleteLater()
        self.agent_cards = []
        for agent in self.library.agents():
            card = AgentCard(agent, self.agent_model(agent))
            card.clicked.connect(self.select_agent)
            card.set_active(agent["id"] == self.current_agent["id"])
            self.agent_list.addWidget(card)
            self.agent_cards.append(card)

    def _conversation_key(self) -> tuple:
        return (self.current_agent["id"], self.current_scope)

    def _cache_conversation(self) -> None:
        if self.message_widgets:
            self._conversation_cache[self._conversation_key()] = [
                {"role": m.role, "text": m.text, "sources": m.sources, "kind": getattr(m, "kind", "chat")}
                for m in self.message_widgets
            ]

    def select_agent(self, agent_id: int) -> None:
        if agent_id == self.current_agent["id"] or self.active_task:
            return
        agent = self.library.agent(agent_id)
        if not agent:
            return
        self._cache_conversation()
        self._clear_chat()
        self.current_agent = agent
        self.settings.setValue("agents/current", agent_id)
        self._apply_agent_to_ui()
        self._begin_conversation()

    def create_agent(self) -> None:
        dialog = AgentEditorDialog(None, self.installed_models, self.model_settings()["chat_model"], self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            agent_id = self.library.save_agent(dialog.agent_data())
            self.refresh_agents()
            self.select_agent(agent_id)

    def edit_agent(self) -> None:
        can_delete = len(self.library.agents()) > 1
        dialog = AgentEditorDialog(
            self.current_agent, self.installed_models, self.model_settings()["chat_model"], self, can_delete
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if dialog.deleted:
            deleted_id = self.current_agent["id"]
            self.library.delete_agent(deleted_id)
            for key in [k for k in self._conversation_cache if k[0] == deleted_id]:
                del self._conversation_cache[key]
            self._clear_chat()
            self.current_agent = self.library.agents()[0]
            self.settings.setValue("agents/current", self.current_agent["id"])
            self.refresh_agents()
            self._apply_agent_to_ui()
            self._begin_conversation()
            return
        self.library.save_agent(dialog.agent_data())
        self.current_agent = self.library.agent(self.current_agent["id"])
        self.refresh_agents()
        self._apply_agent_to_ui()

    def configure_models(self) -> None:
        dialog = ModelSettingsDialog(self.model_settings(), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        self.settings.setValue("models/host", values["host"] or DEFAULT_HOST)
        self.settings.setValue("models/chat", values["chat_model"])
        self.settings.setValue("models/embedding", values["embedding_model"])
        self.settings.setValue("models/vision", values["vision_model"])
        self.refresh_agents()
        self._apply_agent_to_ui()
        self.update_library_status()
        if self.monitor:
            self._start_monitor()

    # -- chat plumbing ---------------------------------------------------------------------------
    def _append_message(
        self, role: str, text: str, sources: list[dict] | None = None, *,
        kind: str = "chat", retrieval: str = "", streaming: bool = False,
    ) -> ChatMessageWidget:
        agent = self.current_agent
        message = ChatMessageWidget(
            role, text, sources, self.chat_body, agent_name=agent["name"], accent=agent["accent"],
            retrieval=retrieval, streaming=streaming,
        )
        message.kind = kind
        if role == "assistant" and kind == "intro":
            message.action_host.hide()
        elif role == "assistant" and kind in {"search", "error"}:
            message.action_buttons["regenerate"].hide()
            message.action_buttons["explain"].hide()
        message.source_opened.connect(self.open_source_page)
        message.explain_selected.connect(lambda passage, m=message: self.explain_more(passage, m.sources))
        message.regenerate_requested.connect(lambda m=message: self.regenerate(m))
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, message)
        self.message_widgets.append(message)
        QTimer.singleShot(0, self._refresh_message_layout)
        QTimer.singleShot(0, self._scroll_to_bottom)
        return message

    def _scroll_to_bottom(self) -> None:
        bar = self.chat_scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _clear_chat(self) -> None:
        for message in self.message_widgets:
            self.chat_layout.removeWidget(message)
            message.hide()
            message.deleteLater()
        self.message_widgets.clear()

    def _intro_text(self) -> str:
        agent = self.current_agent
        class_name, lecture_name = self.current_scope
        scope = f"{class_name} · {lecture_name}" if lecture_name else class_name or "all of your lecture notes"
        if agent["notes_mode"] == "off":
            return (f"Hi, I'm **{agent['name']}**. I'm in free-chat mode, so I'm not reading your notes. "
                    "Pick a notes agent to ask about lectures.")
        if agent["notes_mode"] == "blend":
            return (f"Hi, I'm **{agent['name']}**, linked to **{scope}**. I'll start from your notes and cite the "
                    "pages, then fill gaps with general knowledge (clearly marked).")
        return (f"Hi, I'm **{agent['name']}**, linked to **{scope}**. Ask me anything from your lectures. I answer "
                "only from your notes, with numbered citations you can open.")

    def _begin_conversation(self) -> None:
        cached = self._conversation_cache.get(self._conversation_key())
        if cached is not None:
            for entry in cached:
                self._append_message(entry["role"], entry["text"], entry["sources"], kind=entry["kind"])
            return
        class_name, lecture_name = self.current_scope
        if class_name is not None and lecture_name is None:
            for entry in self.library.class_chat_history(class_name, limit=25, agent_id=self.current_agent["id"]):
                self._append_message("user", entry["question"])
                self._append_message("assistant", entry["answer"], entry["sources"])
        if not self.message_widgets:
            self._append_message("assistant", self._intro_text(), kind="intro")

    def _history_for_model(self) -> list[tuple[str, str]]:
        return [(m.role, m.text) for m in self.message_widgets if getattr(m, "kind", "chat") == "chat" and m.text]

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if hasattr(self, "message_widgets") and hasattr(self, "chat_scroll"):
            QTimer.singleShot(0, self._refresh_message_layout)

    def _refresh_message_layout(self) -> None:
        available = max(280, self.chat_scroll.viewport().width() - 96)
        for message in self.message_widgets:
            message.set_available_width(min(920, available))
            message.updateGeometry()
        self.chat_layout.invalidate()

    def set_mode(self, mode: str) -> None:
        find_mode = mode == "find"
        self.ask_mode_button.setChecked(not find_mode)
        self.find_mode_button.setChecked(find_mode)
        self.semantic_search.setVisible(find_mode)
        self.send_button.setText("FIND ▸" if find_mode else "SEND ▸")
        self.query.setPlaceholderText(
            "Topic or phrase to find in your PDFs…" if find_mode
            else f"Message {self.current_agent['name']}…"
        )

    def current_mode(self) -> str:
        return "find" if self.find_mode_button.isChecked() else "ask"

    def send_message(self) -> None:
        if self.active_task:
            return
        if self.current_mode() == "find":
            self.search()
        else:
            self.answer_question()

    def toggle_right_panel(self, visible: bool) -> None:
        self.right_panel.setVisible(visible)
        QTimer.singleShot(0, self._refresh_message_layout)

    def open_source_page(self, row: dict) -> None:
        self.panel_button.setChecked(True)
        self.right_stack.setCurrentIndex(1)
        self.display_page(row)

    # -- courses -------------------------------------------------------------------------------
    def refresh_navigation(self, select_scope: tuple[str | None, str | None] | None = None) -> None:
        if select_scope is None:
            select_scope = getattr(self, "current_scope", (None, None))
        self.navigator.blockSignals(True)
        self.navigator.clear()
        all_item = QTreeWidgetItem(["◇ All materials"])
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
        selected_item = all_item
        target_class, target_lecture = select_scope
        if target_class is not None and target_class in classes:
            class_item = classes[target_class]
            selected_item = class_item
            for index in range(class_item.childCount()):
                child = class_item.child(index)
                if child.data(0, Qt.ItemDataRole.UserRole) == (target_class, target_lecture):
                    selected_item = child
                    break
        self.navigator.blockSignals(False)
        self.navigator.setCurrentItem(selected_item)
        self.change_scope(selected_item, None)
        self.update_library_status()

    def update_library_status(self) -> None:
        values = self.model_settings()
        status = self.library.indexing_status(values["embedding_model"])
        total = status["total_pages"]
        if total == 0:
            self.library_status.setText("NO PDFS YET · ADD A LECTURE TO START")
            return
        lines = [f"{total} PAGES · {status['indexed_pages']}/{total} INDEXED"]
        if status["pending_visual_pages"] and values["vision_model"]:
            lines.append(f"HANDWRITING UNREAD: {status['pending_visual_pages']}")
        self.library_status.setText("\n".join(lines))

    def change_scope(self, current: QTreeWidgetItem | None, _previous: QTreeWidgetItem | None) -> None:
        if current is None:
            return
        new_scope = current.data(0, Qt.ItemDataRole.UserRole)
        if new_scope != self.current_scope:
            self._cache_conversation()
            self._clear_chat()
            self.current_scope = new_scope
            self.settings.setValue("study_space/class", new_scope[0] or "")
            self.settings.setValue("study_space/lecture", new_scope[1] or "")
            self._begin_conversation()
        class_name, lecture_name = self.current_scope
        if lecture_name:
            self.scope_title.setText(f"{class_name} · {lecture_name}".upper())
        elif class_name:
            self.scope_title.setText(class_name.upper())
        else:
            self.scope_title.setText("ALL MATERIALS")
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

    def clear_conversation(self) -> None:
        class_name, lecture_name = self.current_scope
        if class_name and not lecture_name:
            answer = QMessageBox.question(
                self, "Clear conversation",
                f"Clear {self.current_agent['name']}'s saved conversation for {class_name}?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            self.library.clear_class_chat(class_name, self.current_agent["id"])
        self._conversation_cache.pop(self._conversation_key(), None)
        self._clear_chat()
        self._begin_conversation()

    # -- importing -----------------------------------------------------------------------------
    def import_pdf(self) -> None:
        filenames, _ = QFileDialog.getOpenFileNames(self, "Choose lecture PDFs", str(Path.home()), "PDF files (*.pdf)")
        if not filenames:
            return
        selected_class = self.current_scope[0]
        if len(filenames) == 1:
            details = ImportDetailsDialog(filenames[0], self, selected_class or "")
            if details.exec() != QDialog.DialogCode.Accepted:
                return
            specs = [(Path(filenames[0]), details.class_name.text().strip(), details.lecture_name.text().strip())]
            if not specs[0][1] or not specs[0][2]:
                QMessageBox.warning(self, "Missing details", "Enter both a class and lecture name.")
                return
        else:
            details = BatchImportDetailsDialog(filenames, self, selected_class or "")
            if details.exec() != QDialog.DialogCode.Accepted:
                return
            specs = details.import_specs()
        self._import_pdf_specs(specs)

    def _import_pdf_specs(self, specs: list[tuple[Path, str, str]]) -> None:
        database = self.library.path

        def import_pages(progress: TaskReporter) -> dict:
            imported, failures = [], []
            library = Library(database)
            try:
                for index, (path, class_name, lecture_name) in enumerate(specs, start=1):
                    progress(f"Reading PDF {index} of {len(specs)}: {path.name}…")
                    try:
                        page_texts = extract_page_texts(path)
                        library.import_pdf(class_name, lecture_name, path, page_texts)
                        imported.append({
                            "filename": path.name, "page_count": len(page_texts),
                            "empty_pages": sum(not text for text in page_texts),
                            "class_name": class_name, "lecture_name": lecture_name,
                        })
                    except Exception as error:
                        failures.append(f"{path.name}: {error}")
            finally:
                library.close()
            return {"imported": imported, "failures": failures}

        def imported(result: object) -> None:
            files = result["imported"]
            if files:
                last = files[-1]
                self.refresh_navigation((last["class_name"], last["lecture_name"]))
                pages = sum(item["page_count"] for item in files)
                self.statusBar().showMessage(
                    f"Imported {len(files)} PDF(s), {pages} page(s). Press INDEX FOR SEARCH to enable meaning search.",
                    15000,
                )
            if result["failures"]:
                QMessageBox.warning(
                    self, "Some PDFs could not be imported",
                    "The other PDFs were imported successfully.\n\n" + "\n".join(result["failures"]),
                )
            elif not files:
                QMessageBox.warning(self, "No PDFs imported", "None of the selected PDFs could be imported.")

        self._run_task(import_pages, imported,
                       on_error=lambda error: QMessageBox.critical(self, "Could not import PDF", error))

    # -- task runner ---------------------------------------------------------------------------
    def _run_task(
        self,
        action: Callable[[TaskReporter], object],
        on_success: Callable[[object], None],
        on_error: Callable[[str], None] | None = None,
        state: str = "WORKING",
    ) -> bool:
        if self.active_task:
            QMessageBox.information(self, "Still working", "Wait for the current operation to finish.")
            return False
        task = BackgroundTask(action)
        self.active_task = task
        self.task_success_handler = on_success
        self.task_error_handler = on_error
        for control in self.task_controls:
            control.setEnabled(False)
        task.progress.connect(self.statusBar().showMessage)
        task.chunk.connect(self._queue_token)
        task.thinking.connect(self._on_thinking)
        task.event.connect(self._on_task_event)
        task.finished.connect(self._task_finished)
        self._set_state(state)
        task.start()
        return True

    def _task_finished(self) -> None:
        task = self.active_task
        self.active_task = None
        self._flush_tokens()
        self._token_timer.stop()
        for control in self.task_controls:
            control.setEnabled(True)
        self.history_button.setEnabled(self.current_scope[0] is not None)
        handler, error_handler = self.task_success_handler, self.task_error_handler
        self.task_success_handler = self.task_error_handler = None
        self._set_state("OFFLINE" if self.ollama_online is False else "IDLE")
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
            elif isinstance(task.exception, FileNotFoundError):
                missing = getattr(task.exception, "filename", None)
                self._append_message(
                    "assistant",
                    f"I couldn't open the original PDF at {missing}. The imported text is still saved, but the page "
                    "image is unavailable. Check that the PDF is still there, then re-import it if you moved or "
                    "renamed it.",
                    kind="error",
                )
            else:
                self._append_message("assistant", f"I couldn't complete that.\n\n{task.error}", kind="error")
        elif handler is not None:
            handler(task.result)
        self.query.setFocus()

    def _queue_token(self, token: str) -> None:
        if self._streaming_message is None:
            return
        if not self._pending_tokens and self.panel_state.text() != "◆ STREAMING":
            self._set_state("STREAMING")
        self._pending_tokens.append(token)
        if not self._token_timer.isActive():
            self._token_timer.start()

    def _flush_tokens(self) -> None:
        if self._streaming_message is not None and self._pending_tokens:
            self._streaming_message.append_token("".join(self._pending_tokens))
            self._pending_tokens.clear()
            bar = self.chat_scroll.verticalScrollBar()
            if bar.maximum() - bar.value() < 160:
                QTimer.singleShot(0, self._scroll_to_bottom)

    def _on_thinking(self, token: str) -> None:
        if self._streaming_message is not None:
            if self.panel_state.text() != "◆ REASONING":
                self._set_state("REASONING")
            self._streaming_message.append_thinking(token)

    def _on_task_event(self, payload: dict) -> None:
        if payload.get("state"):
            self._set_state(payload["state"])
        if payload.get("retrieval") and self._streaming_message is not None:
            self._streaming_message.set_retrieval(payload["retrieval"])

    # -- indexing & search ---------------------------------------------------------------------
    def index_library(self) -> None:
        database = self.library.path
        models = self._models()

        def index(progress: TaskReporter) -> dict:
            library = Library(database)
            try:
                notes_checked, vision_note = 0, ""
                pending_visual = [dict(row) for row in library.pages_needing_visual_scan()]
                if pending_visual and models.vision_model:
                    if models.vision_ready():
                        progress(f"Reading handwriting on {len(pending_visual)} page(s) with {models.vision_model}…")
                        notes_checked = models.read_visual_notes(pending_visual, library.save_visual_text, progress)
                    else:
                        vision_note = (f" Handwriting skipped: run `ollama pull {models.vision_model}` to enable it.")
                pages = [dict(row) for row in library.pages_to_index(models.embedding_model)]
                indexed = 0
                if pages:
                    progress(f"Embedding {len(pages)} page(s) with {models.embedding_model}…")
                    indexed = models.index_pages(pages, library.save_embedding, progress)
                return {"indexed": indexed, "notes_checked": notes_checked, "vision_note": vision_note}
            finally:
                library.close()

        def indexed(result: object) -> None:
            if result["indexed"] or result["notes_checked"]:
                summary = f"Indexed {result['indexed']} page(s); read handwriting on {result['notes_checked']}."
            else:
                summary = "Everything is already indexed."
            self.statusBar().showMessage(summary + result["vision_note"], 15000)

        self._run_task(index, indexed, state="INDEXING")

    def _retrieve(self, models: StudyModels, query: str, progress: TaskReporter, limit: int,
                  class_name: str | None, lecture_name: str | None, semantic: bool = True) -> tuple[list[dict], str]:
        """Hybrid keyword + vector retrieval. Falls back to keywords if embeddings are unavailable."""
        vector, note = None, ""
        if semantic:
            try:
                vector = models.embed_query(query)
            except OllamaUnavailable:
                raise
            except OllamaError:
                note = f" · KEYWORDS ONLY (pull {models.embedding_model})"
        library = Library(self.library.path)
        try:
            rows = [dict(row) for row in library.search(
                query, query_embedding=vector, model=models.embedding_model, limit=max(limit, 20),
                class_name=class_name, lecture_name=lecture_name, match_any=True,
            )[:limit]]
        finally:
            library.close()
        return rows, note

    def search(self) -> None:
        query = self.query.text().strip()
        if not query:
            return
        class_name, lecture_name = self.current_scope
        self._append_message("user", query, kind="search")
        self.query.clear()
        semantic = self.semantic_search.isChecked()

        def display(rows: object) -> None:
            self.show_search_results(rows, query)

        if semantic:
            models = self._models()

            def run(progress: TaskReporter) -> list[dict]:
                progress("Searching your PDFs by meaning…")
                rows, _note = self._retrieve(models, query, progress, 12, class_name, lecture_name)
                return rows

            def fallback(error: str) -> None:
                rows = self.library.search(query, limit=12, class_name=class_name, lecture_name=lecture_name)
                self.show_search_results(rows, query, note="Ollama is offline, so this used keyword search only.")

            self._run_task(run, display, on_error=fallback, state="SEARCHING")
            return
        try:
            rows = self.library.search(query, limit=12, class_name=class_name, lecture_name=lecture_name)
        except Exception as error:
            self._append_message("assistant", f"I couldn't search those terms: {error}", kind="error")
            return
        display(rows)

    def show_search_results(self, rows: object, query: str = "", note: str = "") -> None:
        matches = [dict(row) for row in rows]
        if matches:
            text = f"Found **{len(matches)}** page(s) for “{query}”. Open one below."
        else:
            text = f"No page matched “{query}”. Try other wording, or turn on BY MEANING."
        if note:
            text += f"\n\n*{note}*"
        self._append_message("assistant", text, matches, kind="search",
                             retrieval=f"SEARCHED “{query}” · {len(matches)} HITS")
        self.statusBar().showMessage(f"Found {len(matches)} matching page(s).", 5000)

    def display_page(self, row: dict) -> None:
        self.current_result = row
        path = Path(row["file_path"])
        page_number = row["page_number"]
        self.preview_title.setText(f"{row['class_name']} · {row['lecture_name']} · P.{page_number}\n{path.name}".upper())
        body = (row.get("text") or "").strip() or "No embedded text on this page."
        visual_text = (row.get("visual_text") or "").strip()
        if visual_text:
            body += "\n\nMachine-read handwriting / visual notes (verify against the page):\n" + visual_text
        self.excerpt.setPlainText(body)
        try:
            pixmap = QPixmap()
            pixmap.loadFromData(render_page(path, page_number))
            target = self.preview.size() if self.preview.width() > 50 else self.right_panel.size()
            self.preview.setPixmap(pixmap.scaled(
                target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
            ))
        except (OSError, ValueError, RuntimeError) as error:
            self.preview.setText(f"Could not render page: {error}")

    # -- asking --------------------------------------------------------------------------------
    def answer_question(self, question: str | None = None, *, add_user: bool = True,
                        fixed_sources: list[dict] | None = None) -> None:
        question = (question if question is not None else self.query.text()).strip()
        if not question:
            return
        agent = dict(self.current_agent)
        history = self._history_for_model()
        if add_user:
            self._append_message("user", question)
            self.query.clear()
        message = self._append_message("assistant", "", streaming=True)
        self._streaming_message = message
        self._pending_tokens.clear()
        models = self._models()
        model = self.agent_model(agent)
        class_name, lecture_name = self.current_scope
        database = self.library.path

        def answer(progress: TaskReporter) -> dict:
            sources: list[dict] = []
            note = ""
            if agent["notes_mode"] != "off":
                if fixed_sources is not None:
                    sources = [dict(s) for s in fixed_sources]
                else:
                    progress.event({"state": "SEARCHING"})
                    progress("Searching your notes…")
                    sources, note = self._retrieve(models, question, progress, RETRIEVED_PAGES,
                                                   class_name, lecture_name)
                for index, source in enumerate(sources, start=1):
                    source["source_id"] = f"S{index}"
                progress.event({"retrieval": f"SEARCHED NOTES · {len(sources)} PAGE{'S' if len(sources) != 1 else ''}{note}"})
            progress.event({"state": "THINKING"})
            progress(f"{model} is writing…")
            result = models.stream_answer(
                question, sources, model=model, persona=agent["system_prompt"], notes_mode=agent["notes_mode"],
                temperature=float(agent["temperature"]), num_ctx=CONTEXT_WINDOW, reasoning=bool(agent["reasoning"]),
                history=history, on_token=progress.token, on_thinking=progress.think,
            )
            if class_name and fixed_sources is None:
                library = Library(database)
                try:
                    library.save_class_chat(class_name, question, result["answer"], result["sources"], agent["id"])
                finally:
                    library.close()
            return result

        def show(result: object) -> None:
            self._streaming_message = None
            message.finish(result["answer"], result["sources"])
            if not result.get("answerable", True):
                message.kind = "notfound"
            speed = result.get("tokens_per_second", 0.0)
            self.stat_speed.setText(f"{speed:.0f} t/s" if speed else "—")
            self.speed_meter.set_value(speed / 100)
            context = result.get("context_used", 0.0)
            self.stat_context.setText(f"{context:.0%} of {CONTEXT_WINDOW // 1024}k")
            self.context_meter.set_value(context)
            self.statusBar().showMessage(f"{model} · {speed:.1f} tokens/s", 8000)
            QTimer.singleShot(0, self._refresh_message_layout)

        def failed(error: str) -> None:
            self._streaming_message = None
            message.kind = "error"
            message.finish(f"⚠ {error}", [])

        if not self._run_task(answer, show, on_error=failed, state="THINKING"):
            self._streaming_message = None
            message.finish("Another operation is still running.", [])

    def regenerate(self, message: ChatMessageWidget) -> None:
        if self.active_task or message not in self.message_widgets:
            return
        index = self.message_widgets.index(message)
        question = next((m.text for m in reversed(self.message_widgets[:index]) if m.role == "user"), None)
        if not question:
            return
        for widget in self.message_widgets[index:]:
            self.chat_layout.removeWidget(widget)
            widget.hide()
            widget.deleteLater()
        del self.message_widgets[index:]
        # drop the matching user turn from model history, then re-ask without re-adding it
        user_widget = self.message_widgets.pop() if self.message_widgets and self.message_widgets[-1].role == "user" else None
        if user_widget is not None:
            self.chat_layout.removeWidget(user_widget)
            user_widget.hide()
            user_widget.deleteLater()
        self.answer_question(question)

    def explain_more(self, passage: str, sources: list[dict]) -> None:
        if self.active_task:
            return
        self._append_message("user", f"Can you explain this more deeply?\n\n{passage}")
        self.answer_question(
            f"Explain this passage in more depth, step by step:\n{passage}", add_user=False,
            fixed_sources=sources or None,
        )

    # -- quizzes -------------------------------------------------------------------------------
    def start_quiz(self) -> None:
        documents = [dict(row) for row in self.library.documents()]
        if not documents:
            QMessageBox.information(self, "No lectures yet", "Import a PDF before building a quiz.")
            return
        setup = QuizSetupDialog(documents, self)
        if setup.exec() == QDialog.DialogCode.Accepted:
            self.generate_quiz(setup.scopes, setup.question_count.value(), setup.question_type.currentData())

    def generate_quiz(self, scopes: list[tuple[str, str]], count: int, mode: str, focus_topic: str | None = None) -> None:
        database = self.library.path
        models = self._models()

        def generate(progress: TaskReporter) -> dict:
            library = Library(database)
            try:
                progress("Selecting source pages from the chosen lectures…")
                sources = library.pages_for_scopes(scopes, limit=12)
                if not sources:
                    return {"questions": [], "sources": []}
                for index, source in enumerate(sources, start=1):
                    source["source_id"] = f"S{index}"
                progress(f"{models.chat_model} is writing practice questions…")
                generated = models.generate_quiz(count, mode, sources, focus_topic=focus_topic)
                return {"questions": library.create_quiz_session(scopes, generated), "sources": sources}
            finally:
                library.close()

        self._run_task(generate, self._show_quiz, state="WRITING QUIZ")

    def _show_quiz(self, result: object) -> None:
        if not result["questions"]:
            QMessageBox.information(self, "No source pages", "The selected lectures have no pages available to quiz.")
            return
        dialog = QuizRunDialog(result["questions"], self)
        dialog.answer_submitted.connect(lambda question, answer: self.grade_quiz_answer(dialog, question, answer))
        dialog.exec()
        self.statusBar().showMessage("Quiz finished. Weak-topic scores are updated.", 10000)

    @staticmethod
    def _citation_labels(question: dict, sources: list[dict]) -> str:
        source_map = {source["source_id"]: source for source in sources}
        labels = [
            f"{s['class_name']} · {s['lecture_name']} · {Path(s['file_path']).name} · p. {s['page_number']}"
            for s in (source_map[sid] for sid in question["source_ids"])
        ]
        return "; ".join(dict.fromkeys(labels))

    def grade_quiz_answer(self, dialog: QuizRunDialog, question: dict, answer: str) -> None:
        sources = [question["source_map"][sid] for sid in question["source_ids"]]
        citations = self._citation_labels(question, sources)
        if question["question_type"] == "multiple_choice":
            correct = answer == question["correct_answer"]
            score = 1.0 if correct else 0.0
            feedback = "Your choice is correct." if correct else "That choice is incorrect. The cited explanation below shows the key idea."
            self.library.record_quiz_attempt(question["id"], answer, score, correct, feedback)
            dialog.show_grade({"score": score, "is_correct": correct, "feedback": feedback,
                               "explanation_citations": f"[{citations}]"})
            return
        database = self.library.path
        models = self._models()

        def grade(progress: TaskReporter) -> dict:
            progress("Grading against the rubric and cited lecture pages…")
            result = models.grade_short_answer(question, answer, sources)
            library = Library(database)
            try:
                library.record_quiz_attempt(question["id"], answer, result["score"], result["is_correct"], result["feedback"])
            finally:
                library.close()
            return {**result, "explanation_citations": f"[{citations}]"}

        dialog.set_grading(True)
        if not self._run_task(grade, dialog.show_grade, on_error=dialog.grading_failed, state="GRADING"):
            dialog.grading_failed()

    def open_weak_spots(self) -> None:
        dialog = WeakSpotsDialog(self.library.topic_performance(), self)
        dialog.review_requested.connect(self.generate_topic_review)
        dialog.exec()

    def generate_topic_review(self, topic: str) -> None:
        database = self.library.path
        models = self._models()

        def retrieve_and_generate(progress: TaskReporter) -> dict:
            progress(f"Finding source pages for {topic}…")
            sources, _ = self._retrieve(models, topic, progress, 12, None, None)
            if not sources:
                return {"questions": [], "sources": []}
            for index, source in enumerate(sources, start=1):
                source["source_id"] = f"S{index}"
            scopes = sorted({(s["class_name"], s["lecture_name"]) for s in sources})
            progress(f"Building a targeted review quiz for {topic}…")
            generated = models.generate_quiz(3, "mixed", sources, focus_topic=topic)
            library = Library(database)
            try:
                return {"questions": library.create_quiz_session(scopes, generated), "sources": sources}
            finally:
                library.close()

        self._run_task(retrieve_and_generate, self._show_quiz, state="WRITING QUIZ")

    # -- shutdown ------------------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802
        self.closing = True
        if self.monitor:
            self.monitor.stop()
            self.monitor.wait(3000)
        if self.active_task and self.active_task.isRunning():
            self.statusBar().showMessage("Waiting for the current local-model request to finish…")
            self.active_task.wait()
        self.library.close()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
