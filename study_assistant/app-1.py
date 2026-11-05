"""Cortex — a chat interface for studying your lecture notes, on the Gemini API or a local Ollama."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QBuffer, QIODevice, QPoint, QSettings, Qt, QThread, QTime, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFontMetrics, QGuiApplication, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
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
    QTabBar,
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
    DEFAULT_SHORTCUTS,
    ImportDetailsDialog,
    ModelSettingsDialog,
    QuizRunDialog,
    QuizSetupDialog,
    ShortcutsDialog,
    WeakSpotsDialog,
    _group_lectures,
)
from .hardware import is_cloud_model, is_installed, models_by_role
from .macos import (
    SCREEN_SETTINGS_URL,
    HotkeyManager,
    ScreenCaptureError,
    Shortcut,
    capture_interactive,
    capture_region,
    request_screen_capture,
    screen_capture_allowed,
    shortcut_label,
)
from .gemini import save_api_key
from .ollama import (
    DEFAULT_HOST,
    AnswerStopped,
    ChatResult,
    OllamaError,
    OllamaUnavailable,
    StudyModels,
    describe_runtime,
    finalize_answer,
)
from .overlay import MODES, ScreenOverlay, build_screen_prompt
from .providers import (
    EMBED_BATCH,
    GEMINI,
    POLL_SECONDS,
    best_model,
    get_api_key,
    load_health,
    load_settings,
    make_client,
    model_fits,
    save_settings,
)
from .pdf_reader import extract_page_texts, render_page
from .storage import Library
from .theme import ACCENTS, APP_NAME, APP_TAGLINE, COLORS, creativity_label, mono_font, stylesheet
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
        self.streamed: list[str] = []  # every token so far, so a stopped answer keeps what it wrote

    def __call__(self, message: str) -> None:
        self._task.progress.emit(message)

    @property
    def stopped(self) -> bool:
        return self._task.stop_event.is_set()

    @property
    def stop_event(self) -> threading.Event:
        return self._task.stop_event

    def watch(self, models: "StudyModels", notices: bool = True) -> None:
        """Route the client's retry and wait notices here, and let Stop cut its waits short."""
        client = models.client
        if hasattr(client, "on_status"):
            client.on_status = (lambda text: (self(text), self.event({"notice": text}))) if notices else self
            client.stop_event = self.stop_event

    def token(self, text: str) -> None:
        if self.stopped:
            raise AnswerStopped  # unwinds the HTTP stream, which closes the connection to Ollama
        self.streamed.append(text)
        self._task.chunk.emit(text)

    def think(self, text: str) -> None:
        if self.stopped:
            raise AnswerStopped
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
        self.stop_event = threading.Event()
        self.result: object | None = None
        self.error: str | None = None
        self.exception: Exception | None = None

    def run(self) -> None:
        try:
            self.result = self.action(TaskReporter(self))
        except Exception as error:  # shown in the app rather than killing the UI
            self.exception = error
            self.error = str(error)


class ModelMonitor(QThread):
    """Polls the model backend (Gemini or Ollama) so the UI can show online state, models, and GPU use."""

    status = Signal(object)

    def __init__(self, values: dict) -> None:
        super().__init__()
        self.values = dict(values)
        self.interval = POLL_SECONDS[values["provider"]]
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        client = make_client(self.values)  # one client, so each model's capabilities are looked up once
        while not self._stop:
            try:
                models = client.list_models(timeout=5)
                running = client.running_models(timeout=2)
                for model in models:
                    if not is_cloud_model(model["name"]):
                        model["capabilities"] = sorted(client.capabilities(model["name"]))
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


class _EvenTabBar(QTabBar):
    """Tabs that split the full width evenly, so their labels sit centred under the sidebar."""

    def tabSizeHint(self, index: int):  # noqa: N802
        hint = super().tabSizeHint(index)
        parent = self.parentWidget()
        if parent is not None and self.count():
            hint.setWidth(parent.width() // self.count())
        return hint

    def minimumTabSizeHint(self, index: int):  # noqa: N802
        return super().tabSizeHint(index)


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
        self.chat_models: list[str] = []  # installed, local, able to answer questions
        self.running_models: list[dict] = []
        self.monitor: ModelMonitor | None = None
        self._auto_indexed = False
        self._auto_index_running = False
        self._send_after_task = False

        load_health(self.settings)  # which Gemini models answered the last model check
        QTimer.singleShot(1500, self._warn_missing_renderers)
        agents = self.library.agents()
        saved_agent = self.settings.value("agents/current", 0, type=int)
        self.current_agent = next((a for a in agents if a["id"] == saved_agent), agents[0])

        self._build_sidebar()
        self._build_center()
        self._build_right_panel()
        self.overlay = ScreenOverlay(settings=self.settings)
        self.overlay.read_requested.connect(self.read_screen)
        self.overlay.stop_requested.connect(self.stop_screen_answer)
        self.overlay.settings_requested.connect(self.configure_shortcuts)
        self.overlay.source_opened.connect(self._open_source_from_overlay)
        self.screen_task: BackgroundTask | None = None
        self.hotkeys: HotkeyManager | None = None
        self.failed_shortcuts: list[str] = []

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
            self.history_button, self.new_agent_button, self.config_button, self.clear_button, self.model_picker,
            self.models_button, self.agent_host,
        ]

        self._wait_timer = QTimer(self)  # counts up "Waiting for <model> · 12s" until the first words arrive
        self._wait_timer.setInterval(1000)
        self._wait_timer.timeout.connect(self._refresh_waiting)
        self._wait_started = 0.0
        self._wait_notice = ""
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
            self.hotkeys = HotkeyManager()
        self.apply_shortcuts()
        QShortcut(QKeySequence("Ctrl+K"), self, activated=lambda: self.query.setFocus())

    # -- settings ------------------------------------------------------------------------------
    def model_settings(self) -> dict:
        """The active provider (Gemini by default) and its models."""
        return load_settings(self.settings)

    @property
    def provider(self) -> str:
        return self.model_settings()["provider"]

    def _models(self) -> StudyModels:
        values = self.model_settings()
        return StudyModels(
            make_client(values),
            chat_model=values["chat_model"],
            embedding_model=values["embedding_model"],
            vision_model=values["vision_model"] or None,
        )

    def agent_model(self, agent: dict | None = None) -> str:
        """The agent's own model when it belongs to the active provider, else the default chat model."""
        agent = agent or self.current_agent
        values = self.model_settings()
        model = agent.get("model") or ""
        return model if model and model_fits(model, values["provider"]) else values["chat_model"]

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
        self.gear_button.setToolTip("Model settings")
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
            "Embed every page with your embedding model so questions find pages by meaning "
            "(and read handwriting if that's turned on under ⬡ MODELS)."
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
            "Quizzes are written and graded by your default chat model from the lectures you pick. "
            "Weak spots ranks topics by recent accuracy."
        )
        study_help.setObjectName("dialogHelp")
        study_help.setWordWrap(True)
        study_layout.addWidget(study_help)
        study_layout.addStretch(1)

        self.side_tabs = QTabWidget()
        self.side_tabs.setTabBar(_EvenTabBar())
        self.side_tabs.addTab(courses, "COURSES")
        self.side_tabs.addTab(study, "STUDY")
        self.side_tabs.tabBar().setElideMode(Qt.TextElideMode.ElideNone)  # never "CO…"
        self.side_tabs.tabBar().setDrawBase(False)
        layout.addWidget(self.side_tabs, 1)

        self.models_button = QPushButton("⬡  MODELS")
        self.models_button.clicked.connect(self.configure_models)
        layout.addWidget(self.models_button)
        self.footer_label = _mono_label("", "footerNote")
        self.footer_label.setToolTip(str(self.library.path))
        layout.addWidget(self.footer_label)

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
        row.addSpacing(10)
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
        header_layout.setSpacing(8)
        self.header_avatar = AgentAvatar("", ACCENTS[0], 50)
        header_layout.addWidget(self.header_avatar)
        info = QVBoxLayout()
        info.setSpacing(5)
        title_row = QHBoxLayout()
        title_row.setSpacing(12)
        self.agent_title = _mono_label("", "agentTitle")
        title_row.addWidget(self.agent_title)
        self.scope_title = _mono_label("ALL MATERIALS", "scopeChip")
        self.scope_title.setMaximumWidth(280)  # long lecture names clip instead of crowding the header
        # Both may shrink (clipping their text) so a long agent, course, or model name can't push the
        # header wider than the window and squash the stats underneath.
        self.agent_title.setMinimumWidth(40)
        self.scope_title.setMinimumWidth(40)
        title_row.addWidget(self.scope_title, 0, Qt.AlignmentFlag.AlignVCenter)
        title_row.addSpacing(10)
        title_row.addWidget(_mono_label("MODEL", "statLabel"))
        self.model_picker = QComboBox()
        self.model_picker.setObjectName("modelPicker")
        self.model_picker.setEditable(False)
        self.model_picker.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.model_picker.setToolTip("The model this agent answers with. Only installed models are listed.")
        self.model_picker.activated.connect(self._model_picked)
        title_row.addWidget(self.model_picker, 0, Qt.AlignmentFlag.AlignVCenter)
        title_row.addStretch(1)
        info.addLayout(title_row)
        row1 = QHBoxLayout()
        row1.setSpacing(5)
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
        row2.addSpacing(10)
        self.stat_notes = self._stat(row2, "NOTES")
        row2.addStretch(1)
        info.addLayout(row1)
        info.addLayout(row2)
        header_layout.addLayout(info, 1)
        self.overlay_button = QPushButton("◉ OVERLAY")
        self.overlay_button.setToolTip("Float a see-through panel over your screen that reads it")
        self.overlay_button.clicked.connect(self.toggle_overlay)
        self.config_button = QPushButton("⚙ CONFIG")
        self.config_button.clicked.connect(self.edit_agent)
        self.clear_button = QPushButton("✕")
        self.clear_button.setToolTip("Clear this conversation")
        self.clear_button.clicked.connect(self.clear_conversation)
        for button in (self.overlay_button, self.config_button, self.clear_button):
            button.setObjectName("headerButton")
            header_layout.addWidget(button, 0, Qt.AlignmentFlag.AlignTop)
        # The agent's live status: the ring spins while it works; click for the agent card.
        header_layout.addSpacing(6)
        self._build_link_card()
        header_layout.addWidget(self.panel_avatar, 0, Qt.AlignmentFlag.AlignTop)
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
        self.semantic_search.setToolTip("Also rank pages by meaning with your embedding model.")
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
        self.composer_wrap = QVBoxLayout()
        self.composer_wrap.setContentsMargins(32, 8, 32, 18)
        self.composer_wrap.addWidget(self.composer)
        center.addLayout(self.composer_wrap)

    # -- layout: right panel -------------------------------------------------------------------
    def _build_link_card(self) -> None:
        """The header's status badge and the card it opens: who the agent is, what it's doing, its persona."""
        self.panel_avatar = AgentAvatar("", ACCENTS[0], 44, ring=True)
        self.panel_avatar.setCursor(Qt.CursorShape.PointingHandCursor)
        self.panel_avatar.clicked.connect(self.show_link_card)
        card = QFrame(self, Qt.WindowType.Popup)
        card.setObjectName("linkCard")
        card.setFixedWidth(340)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 14, 18, 16)
        layout.setSpacing(6)
        top = QHBoxLayout()
        top.addWidget(_mono_label("// NEURAL LINK", "panelSubject"))
        top.addStretch(1)
        self.clock_label = _mono_label("", "panelSubject")
        top.addWidget(self.clock_label)
        layout.addLayout(top)
        layout.addSpacing(4)
        identity = QHBoxLayout()
        identity.setSpacing(14)
        self.card_avatar = AgentAvatar("", ACCENTS[0], 84, ring=True)
        identity.addWidget(self.card_avatar, 0, Qt.AlignmentFlag.AlignTop)
        names = QVBoxLayout()
        names.setSpacing(4)
        self.panel_subject = _mono_label("", "panelSubject")
        self.panel_name = _mono_label("", "panelName")
        self.panel_state = _mono_label("◆ IDLE", "monoMuted")
        self.card_model = _mono_label("", "monoMuted")
        for widget in (self.panel_subject, self.panel_name, self.panel_state, self.card_model):
            names.addWidget(widget)
        names.addStretch(1)
        identity.addLayout(names, 1)
        layout.addLayout(identity)
        layout.addWidget(_mono_label("·" * 40, "monoMuted"))
        self.panel_prompt = QLabel()
        self.panel_prompt.setObjectName("panelPrompt")
        self.panel_prompt.setWordWrap(True)
        layout.addWidget(self.panel_prompt)
        edit = QPushButton("⚙ EDIT AGENT")
        edit.setObjectName("ghostButton")
        edit.clicked.connect(lambda: (card.hide(), self.edit_agent()))
        layout.addWidget(edit, 0, Qt.AlignmentFlag.AlignLeft)
        self.link_card = card

    def show_link_card(self) -> None:
        self.card_model.setText(f"MODEL {self.model_settings()['chat_model']}"[:44])
        self.link_card.adjustSize()
        corner = self.panel_avatar.mapToGlobal(QPoint(self.panel_avatar.width(), self.panel_avatar.height() + 6))
        self.link_card.move(corner.x() - self.link_card.width(), corner.y())
        self.link_card.show()

    def _build_right_panel(self) -> None:
        self.right_panel = QWidget()
        self.right_panel.setObjectName("rightPanel")
        self.right_panel.setFixedWidth(330)
        outer = QVBoxLayout(self.right_panel)
        outer.setContentsMargins(0, 0, 0, 0)
        self.right_stack = QStackedWidget()
        outer.addWidget(self.right_stack)

        # The side panel only appears to preview a cited page; the agent lives in the header badge.
        self.right_stack.addWidget(QWidget())
        self.right_panel.hide()

        # Page preview (opened from citations)
        page = QWidget()
        self.preview_panel = page
        preview_layout = QVBoxLayout(page)
        preview_layout.setContentsMargins(16, 14, 16, 14)
        preview_layout.setSpacing(8)
        back = QPushButton("✕ CLOSE PREVIEW")
        back.setObjectName("ghostButton")
        back.clicked.connect(lambda: self.toggle_right_panel(False))
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
        self.overlay.setStyleSheet(stylesheet(agent["accent"]) + self.overlay.extra_css)
        self.overlay.set_agent(agent["name"], agent["accent"])
        self._update_overlay_model()
        name = agent["name"]
        self.agent_title.setText(name.upper())
        self.header_avatar.set_agent(name, agent["accent"])
        self.panel_avatar.set_agent(name, agent["accent"])
        self.card_avatar.set_agent(name, agent["accent"])
        self.panel_name.setText(name.upper())
        self.panel_subject.setText(f"SUBJECT // ID-{agent['id'] * 7919 % 0xFFFFF:05X}")
        self.panel_prompt.setText(agent["system_prompt"] or "No system prompt set.")
        self.refresh_model_picker()
        temperature = float(agent["temperature"])
        self.stat_creativity.setText(f"{temperature:.2f} {creativity_label(temperature)}")
        self.stat_notes.setText("STRICT · CITED" if agent["notes_mode"] == "strict" else "NOTES + GENERAL")
        self._update_runtime_stat()
        for card in self.agent_cards:
            card.set_active(card.agent_id == agent["id"])
        if hasattr(self, "find_mode_button"):
            self.set_mode(self.current_mode())

    def _set_state(self, text: str) -> None:
        self.panel_state.setText(f"◆ {text}")
        busy = text not in {"IDLE", "OFFLINE"}
        for avatar in (self.panel_avatar, self.card_avatar):
            avatar.set_busy(busy)
            avatar.set_offline(text == "OFFLINE")
        self.panel_avatar.setToolTip(f"{self.current_agent['name']} · {text.lower()}  (click for details)")

    def _composer_focus(self, focused: bool) -> None:
        self.composer.setProperty("focused", focused)
        self.composer.style().unpolish(self.composer)
        self.composer.style().polish(self.composer)

    def _start_monitor(self) -> None:
        if self.monitor:
            self.monitor.stop()
            self.monitor.wait(2000)
        self.monitor = ModelMonitor(self.model_settings())
        self.monitor.status.connect(self._on_ollama_status)
        self.monitor.start()

    def _on_ollama_status(self, status: dict) -> None:
        if self.closing:
            return
        installed = sorted(model["name"] for model in status["models"])
        roles = models_by_role(status["models"])
        chat_models = roles["chat"]
        self.vision_models = roles["vision"] if status["online"] else None
        changed = installed != self.installed_models or chat_models != self.chat_models
        self.installed_models = installed
        self.chat_models = chat_models
        self.running_models = status["running"]
        if status["online"] and self.provider == GEMINI:
            self._settle_gemini_defaults(roles)
        self._set_online_status(status["online"])
        self._update_runtime_stat()
        self._update_overlay_model()
        if changed and not self.active_task:
            self.refresh_agents()
            self.refresh_model_picker()
        if status["online"]:
            self._auto_index()

    def _auto_index(self) -> None:
        """Once per launch, index pages the current embedding model hasn't seen (e.g. after switching to
        Gemini), so meaning search works without remembering to press the button. Skips handwriting,
        which costs a request per page; that stays on the button."""
        if self._auto_indexed or self.active_task or self.closing:
            return
        status = self.library.indexing_status(self.model_settings()["embedding_model"])
        if status["total_pages"] and status["indexed_pages"] < status["total_pages"]:
            self._auto_indexed = True
            self.index_library(automatic=True)
    def _settle_gemini_defaults(self, roles: dict) -> None:
        """If a default Gemini model isn't offered to this key, switch to the closest one that is."""
        values = self.model_settings()
        changes = {}
        if roles["chat"] and values["chat_model"] not in roles["chat"]:
            changes["chat_model"] = best_model(roles["chat"], "chat")
        if roles["embedding"] and values["embedding_model"] not in roles["embedding"]:
            changes["embedding_model"] = best_model(roles["embedding"], "embedding")
        if values["vision_model"] and roles["vision"] and values["vision_model"] not in roles["vision"]:
            changes["vision_model"] = changes.get("chat_model") or best_model(roles["vision"], "chat")
        if changes:
            save_settings(self.settings, {**values, **changes})
            self._apply_agent_to_ui()
            self.update_library_status()

    def _set_online_status(self, online: bool | None) -> None:
        self.ollama_online = online
        provider = self.provider
        name = "GEMINI" if provider == GEMINI else "OLLAMA"
        self.footer_label.setText("◆ CLOUD · GEMINI API" if provider == GEMINI else "◆ 100% LOCAL · NO CLOUD")
        if online is None:
            self.online_label.setText(f"◇ CONNECTING TO {name}…")
            self.online_label.setStyleSheet(f"color: {COLORS['muted']};")
        elif online:
            count = len(self.chat_models)
            self.online_label.setText(f"◆ {name} ONLINE · {count} MODEL{'S' if count != 1 else ''}")
            self.online_label.setStyleSheet(f"color: {COLORS['ok']};")
            if self.panel_state.text() == "◆ OFFLINE":
                self._set_state("IDLE")
        else:
            if provider == GEMINI and not get_api_key():
                self.online_label.setText("◇ ADD A GEMINI API KEY · ⬡ MODELS")
            elif provider == GEMINI:
                self.online_label.setText("◇ GEMINI UNREACHABLE · CHECK KEY / WI-FI")
            else:
                self.online_label.setText("◇ OFFLINE · OPEN THE OLLAMA APP")
            self.online_label.setStyleSheet(f"color: {COLORS['bad']};")
            if not self.active_task:
                self._set_state("OFFLINE")

    def _update_runtime_stat(self) -> None:
        if self.provider == GEMINI:
            self.stat_runtime.setText("CLOUD")
            return
        model = self.agent_model()
        runtime = describe_runtime(self.running_models, model)
        if runtime["loaded"]:
            self.stat_runtime.setText(f"{runtime['where']} · {runtime['memory_gb']:.1f} GB")
        elif self.installed_models and model not in self.installed_models and f"{model}:latest" not in self.installed_models:
            self.stat_runtime.setText("NOT INSTALLED")
        else:
            self.stat_runtime.setText("LOADS ON FIRST USE" if self.ollama_online else "—")

    # -- model picker ------------------------------------------------------------------------
    def refresh_model_picker(self) -> None:
        """List the installed chat models; the agent's own pick is selected."""
        values = self.model_settings()
        agent_model = self.current_agent.get("model") or ""
        if not model_fits(agent_model, values["provider"]):
            agent_model = ""  # e.g. an Ollama model while on Gemini: use the default
        default = values["chat_model"]
        picker = self.model_picker
        picker.blockSignals(True)
        picker.clear()
        picker.addItem(f"★ {default}", "")  # ★ = the default model, which follows ⬡ MODELS
        for name in self.chat_models:
            picker.addItem(name, name)
        if agent_model and not is_installed(agent_model, self.chat_models) and self.ollama_online:
            picker.addItem(f"{agent_model} · {'NOT AVAILABLE' if values['provider'] == GEMINI else 'NOT INSTALLED'}",
                           agent_model)
        elif agent_model and picker.findData(agent_model) < 0:
            picker.addItem(agent_model, agent_model)  # Ollama hasn't reported yet; keep showing the saved pick
        index = picker.findData(agent_model)
        if index < 0 and agent_model:
            index = picker.findData(f"{agent_model}:latest")
        picker.setCurrentIndex(max(0, index))
        picker.blockSignals(False)
        # The stylesheet's mono font is wider than the default metrics Qt sizes with, so size explicitly.
        self._fit_model_picker()

    def _fit_model_picker(self) -> None:
        """Size the closed dropdown to the current pick (capped, so long names can't push the header
        stats off screen); the open list is as wide as its longest entry."""
        picker = self.model_picker
        metrics = QFontMetrics(mono_font(8, bold=True, spacing=0.5))
        widest = max(metrics.horizontalAdvance(picker.itemText(i)) for i in range(picker.count()))
        current = metrics.horizontalAdvance(picker.currentText())
        picker.setFixedWidth(min(current + 44, 220))
        picker.view().setMinimumWidth(widest + 44)
        picker.updateGeometry()

    def _model_picked(self, index: int) -> None:
        self._fit_model_picker()
        model = self.model_picker.itemData(index) or ""
        if model == (self.current_agent.get("model") or ""):
            return
        self.library.save_agent({**self.current_agent, "model": model})
        self.current_agent = self.library.agent(self.current_agent["id"])
        self.refresh_agents()
        self._update_runtime_stat()
        self.statusBar().showMessage(f"{self.current_agent['name']} now answers with {self.agent_model()}.", 6000)

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
        values = self.model_settings()
        dialog = AgentEditorDialog(None, self.chat_models, values["chat_model"], self, host=values["host"],
                                   provider_values=values)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            agent_id = self.library.save_agent(dialog.agent_data())
            self.refresh_agents()
            self.select_agent(agent_id)

    def edit_agent(self) -> None:
        can_delete = len(self.library.agents()) > 1
        values = self.model_settings()
        dialog = AgentEditorDialog(
            self.current_agent, self.chat_models, values["chat_model"], self, can_delete, values["host"],
            provider_values=values,
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
        dialog = ModelSettingsDialog(self.model_settings(), self, settings=self.settings)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        if values.pop("api_key", ""):
            if not save_api_key(dialog.key_input.text()):
                QMessageBox.information(self, "Key not saved to Keychain",
                                        "The key works for now, but macOS Keychain didn't accept it, so you'll "
                                        "need to enter it again next time.")
        save_settings(self.settings, {**values, "host": values.get("host") or DEFAULT_HOST})
        self._auto_indexed = False  # a new embedding model means pages need indexing again
        self.chat_models, self.installed_models = dialog.chat_models(), []
        self._set_online_status(dialog.online)
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
            if kind == "search":
                message.sources_toggle.setChecked(True)  # the pages are the result here: start open
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
        if lecture_name:
            scope = f"every slide of **{class_name} · {lecture_name}**"
        elif class_name:
            scope = f"every lecture in **{class_name}**"
        else:
            scope = "**all of your lecture notes**"
        if agent["notes_mode"] != "strict":
            return (f"Hi, I'm **{agent['name']}**. I read {scope} for each question, cite the pages I use, and "
                    "fill gaps with general knowledge (clearly marked).")
        return (f"Hi, I'm **{agent['name']}**. I read {scope} for each question and answer only from them, with "
                "numbered citations you can open.")

    def _begin_conversation(self) -> None:
        cached = self._conversation_cache.get(self._conversation_key())
        if cached is not None:
            for entry in cached:
                self._append_message(entry["role"], entry["text"], entry["sources"], kind=entry["kind"])
            return
        # Each selection keeps its own conversation: a course's chat doesn't pull in its lectures' chats.
        class_name, lecture_name = self.current_scope
        for entry in self.library.class_chat_history(class_name or "", limit=25, agent_id=self.current_agent["id"],
                                                     lecture_name=lecture_name or ""):
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

    CHAT_COLUMN = 920  # widest the conversation gets; it's centred in the space it has

    def _refresh_message_layout(self) -> None:
        # Centre the conversation by padding both sides equally (aligned items would lose word-wrap heights).
        viewport = self.chat_scroll.viewport().width()
        column = min(self.CHAT_COLUMN, max(280, viewport - 96))
        side = max(48, (viewport - column) // 2)
        self.chat_layout.setContentsMargins(side, 26, side, 26)
        for message in self.message_widgets:
            message.set_available_width(column)
            message.updateGeometry()
        # The message box lines up with the messages above it.
        composer_side = max(32, (self.center_panel.width() - column) // 2)
        self.composer_wrap.setContentsMargins(composer_side, 8, composer_side, 18)
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
            if self.send_button.text() == "■ STOP":
                self.active_task.stop_event.set()
            elif self._auto_index_running and self.query.text().strip():
                self._send_after_task = True
                self.statusBar().showMessage("Indexing your pages for search first; your question sends in a moment…")
            return
        if self.current_mode() == "find":
            self.search()
        else:
            self.answer_question()

    def toggle_right_panel(self, visible: bool) -> None:
        self.right_panel.setVisible(visible)
        QTimer.singleShot(0, self._refresh_message_layout)

    def open_source_page(self, row: dict) -> None:
        self.toggle_right_panel(True)
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

    def _warn_missing_renderers(self) -> None:
        """Math and code still show without these packages, but only as plain text."""
        import importlib.util

        missing = [name for name, module in (("matplotlib", "matplotlib"), ("Pygments", "pygments"))
                   if importlib.util.find_spec(module) is None]
        if missing and not self.closing:
            self.statusBar().showMessage(
                f"{' and '.join(missing)} not installed, so math and code show as plain text. "
                "Run `python -m pip install -e .` in Cortex's folder, then restart.", 0)

    def update_library_status(self) -> None:
        """Sidebar summary of the active context: what the selected lecture / course / library holds and
        whether it all goes to the model with each question. The search index only matters when a
        selection is too big to send whole, so it's mentioned only then."""
        values = self.model_settings()
        model = values["embedding_model"]
        status = self.library.indexing_status(model)
        total = status["total_pages"]
        if total == 0:
            self.library_status.setText("NO PDFS YET · ADD A LECTURE TO START")
            self.index_button.setText("⟳ INDEX FOR SEARCH")
            return
        class_name, lecture_name = getattr(self, "current_scope", (None, None)) or (None, None)
        scope = self.library.scope_status(class_name, lecture_name, model)
        name = (lecture_name or class_name or "WHOLE LIBRARY").upper()
        pages = scope["pages"]
        lines = [f"CONTEXT: {name}",
                 f"{pages} PAGE{'S' if pages != 1 else ''}"
                 + (f" · {scope['lectures']} LECTURES" if not lecture_name and scope["lectures"] > 1 else "")]
        if pages:
            readable = [p for p in self.library.scope_pages(class_name, lecture_name)
                        if (p.get("text") or p.get("visual_text") or "").strip()]
            budget = self.SCOPE_CHARS.get(values["provider"], self.SCOPE_CHARS_DEFAULT)
            if sum(self._page_weight(p) for p in readable) <= budget:
                lines.append("ALL SENT WITH EACH QUESTION")
            else:
                lines.append("TOO BIG TO SEND WHOLE · BEST MATCHES SENT")
                if scope["indexed"] < pages:
                    lines.append(f"SEARCH READY FOR {scope['indexed']}/{pages}")
        pending = total - status["indexed_pages"]
        self.index_button.setText(f"⟳ INDEX {pending} NEW PAGE{'S' if pending != 1 else ''}" if pending
                                  else "⟳ INDEX FOR SEARCH")
        if status["pending_visual_pages"] and values["vision_model"] and values["handwriting"]:
            lines.append(f"HANDWRITING UNREAD: {status['pending_visual_pages']}")
        self.library_status.setText("\n".join(lines))
        self.library_status.setToolTip(
            f"Your library: {total} pages, {status['indexed_pages']} indexed for search.\n"
            "Every question is answered from the selection above. When it fits, all of it is sent; "
            "the search index is only used to pick the best pages from selections too big to send whole.")

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
            self.update_library_status()
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
        saved = self.library.class_chat_history(class_name or "", limit=1, agent_id=self.current_agent["id"],
                                                lecture_name=lecture_name or "")
        if saved:
            where = lecture_name or class_name or "all materials"
            answer = QMessageBox.question(
                self, "Clear conversation",
                f"Clear {self.current_agent['name']}'s saved conversation for {where}? "
                "Conversations for other lectures and courses are kept.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            self.library.clear_class_chat(class_name or "", self.current_agent["id"], lecture_name=lecture_name or "")
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
        stoppable: bool = False,
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
        if stoppable:  # the Send button becomes Stop while an answer streams
            self.send_button.setEnabled(True)
            self.send_button.setText("■ STOP")
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
        self.set_mode(self.current_mode())  # restores the Send label after Stop
        self.history_button.setEnabled(self.current_scope[0] is not None)
        handler, error_handler = self.task_success_handler, self.task_error_handler
        self.task_success_handler = self.task_error_handler = None
        self._auto_index_running = False
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
        if self._send_after_task and not self.closing:  # a question typed during background indexing
            self._send_after_task = False
            QTimer.singleShot(0, self.send_message)

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
        if payload.get("notice"):
            self._wait_notice = payload["notice"]
            self._refresh_waiting()
        if payload.get("retrieval") and self._streaming_message is not None:
            self._streaming_message.set_retrieval(payload["retrieval"])

    def _start_waiting(self, notice: str) -> None:
        self._wait_started = time.monotonic()
        self._wait_notice = notice
        self._refresh_waiting()
        self._wait_timer.start()

    def _refresh_waiting(self) -> None:
        message = self._streaming_message
        if message is None or message.text:
            self._wait_timer.stop()  # the answer has started (or finished): nothing to count
            return
        elapsed = int(time.monotonic() - self._wait_started)
        notice = self._wait_notice
        message.set_waiting(notice if notice[-1:] == "s" and notice[-2:-1].isdigit() else f"{notice} · {elapsed}s")

    # -- indexing & search ---------------------------------------------------------------------
    def index_library(self, automatic: bool = False) -> None:
        database = self.library.path
        models = self._models()
        values = self.model_settings()
        # Handwriting: off by default on Gemini (one request per page), never in the automatic pass.
        read_handwriting = bool(values["handwriting"]) and not automatic
        batch = EMBED_BATCH[values["provider"]]
        pull_hint = ("pick a vision model under ⬡ MODELS" if values["provider"] == GEMINI
                     else f"run `ollama pull {models.vision_model}`")

        def index(progress: TaskReporter) -> dict:
            progress.watch(models, notices=False)
            library = Library(database)
            try:
                notes_checked, vision_note = 0, ""
                pending_visual = [dict(row) for row in library.pages_needing_visual_scan()]
                if pending_visual and models.vision_model and read_handwriting:
                    if models.vision_ready():
                        progress(f"Reading handwriting on {len(pending_visual)} page(s) with {models.vision_model}…")
                        notes_checked = models.read_visual_notes(pending_visual, library.save_visual_text, progress)
                    else:
                        vision_note = f" Handwriting skipped: {pull_hint} to enable it."
                pages = [dict(row) for row in library.pages_to_index(models.embedding_model)]
                indexed = 0
                if pages:
                    progress(f"Embedding {len(pages)} page(s) with {models.embedding_model}…")
                    indexed = models.index_pages(pages, library.save_embedding, progress, batch_size=batch)
                return {"indexed": indexed, "notes_checked": notes_checked, "vision_note": vision_note}
            finally:
                library.close()

        def indexed(result: object) -> None:
            if result["indexed"] or result["notes_checked"]:
                summary = f"Indexed {result['indexed']} page(s); read handwriting on {result['notes_checked']}."
            else:
                summary = "Everything is already indexed."
            if automatic:
                summary = f"Indexed {result['indexed']} page(s) for search in the background."
            self.statusBar().showMessage(summary + result["vision_note"], 15000)

        def failed(error: str) -> None:
            if automatic:  # quiet: the button still works, and the next launch tries again
                self.statusBar().showMessage(f"Background indexing didn't finish: {error}", 15000)
            else:
                self._append_message("assistant", f"I couldn't complete that.\n\n{error}", kind="error")

        if self._run_task(index, indexed, on_error=failed, state="INDEXING") and automatic:
            self._auto_index_running = True
            for control in (self.query, self.send_button, self.ask_mode_button, self.find_mode_button):
                control.setEnabled(True)  # keep typing: a question sent now waits for indexing

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

    # How much slide text one question may carry. Gemini reads a whole course at once (hundreds of
    # pages); a local model's 16k-token window only fits a lecture or two.
    SCOPE_CHARS = {GEMINI: 600_000}
    SCOPE_CHARS_DEFAULT = 30_000

    def _context_window(self) -> int:
        """Tokens the model can take: Gemini Flash reads about a million; local models get 16k here."""
        return 1_048_576 if self.provider == GEMINI else CONTEXT_WINDOW

    @staticmethod
    def _page_weight(page: dict) -> int:
        """Characters a page adds to the prompt (pages are trimmed the same way when sent)."""
        return min(len(page.get("text") or ""), 3500) + min(len(page.get("visual_text") or ""), 2000) + 80

    def _gather_sources(self, models: StudyModels, question: str, progress: TaskReporter,
                        class_name: str | None, lecture_name: str | None, provider: str) -> tuple[list[dict], str]:
        """Every page in the selected scope: the lecture, the whole course, or (nothing selected) the library.

        If that's more than the model can take, the most relevant pages fill the space, kept in teaching
        order. Returns (pages, retrieval label)."""
        library = Library(self.library.path)
        try:
            stored = library.scope_status(class_name, lecture_name, models.embedding_model)
            pages = [p for p in library.scope_pages(class_name, lecture_name)
                     if (p.get("text") or p.get("visual_text") or "").strip()]
        finally:
            library.close()
        if stored["documents"] == 0 or not pages:
            return [], ""  # nothing readable here: no need to ask any model
        where = (f"{lecture_name}" if lecture_name else (f"{class_name}" if class_name else "your library")).upper()
        lectures = len({(p["class_name"], p["lecture_name"]) for p in pages})
        span = f" · {lectures} LECTURES" if lectures > 1 else ""
        budget = self.SCOPE_CHARS.get(provider, self.SCOPE_CHARS_DEFAULT)
        if sum(self._page_weight(p) for p in pages) <= budget:
            progress(f"Reading all {len(pages)} pages of {where.title() if lecture_name or class_name else 'your notes'}…")
            return pages, f"READ ALL {len(pages)} PAGE{'S' if len(pages) != 1 else ''} OF {where}{span}"
        # Too much to send at once: rank by relevance, fill the budget, then put back in teaching order.
        progress("Picking the most relevant pages…")
        ranked, note = self._retrieve(models, question, progress, min(len(pages), 500), class_name, lecture_name)
        order = {page["id"]: index for index, page in enumerate(pages)}
        by_id = {page["id"]: page for page in pages}
        chosen, used = [], 0
        ranked_ids = [r.get("page_id", r.get("id")) for r in ranked]  # search rows call it page_id
        for page in [by_id[i] for i in ranked_ids if i in by_id] + pages:  # relevant first, then the rest
            weight = self._page_weight(page)
            if page["id"] in {c["id"] for c in chosen} or used + weight > budget:
                continue
            chosen.append(page)
            used += weight
        chosen.sort(key=lambda page: order[page["id"]])
        return chosen, f"READ {len(chosen)} MOST RELEVANT OF {len(pages)} PAGES IN {where}{note}"

    def _explain_no_pages(self, class_name: str | None, lecture_name: str | None, models: StudyModels,
                          pages_read: int) -> str:
        """Why a notes-only answer came up empty, in terms of what's actually in the library."""
        library = Library(self.library.path)
        try:
            status = library.scope_status(class_name, lecture_name, models.embedding_model)
            everything = library.scope_status(None, None, models.embedding_model)
        finally:
            library.close()
        scope = f"**{class_name} · {lecture_name}**" if lecture_name else (f"**{class_name}**" if class_name
                                                                          else "your library")
        if everything["documents"] == 0:
            return ("Your library is empty: no lecture PDFs have been added yet. Use **+ ADD LECTURE PDFS** in "
                    "the Courses tab, then ask again.")
        if status["documents"] == 0:
            return f"There are no PDFs in {scope} yet. Add some with **+ ADD LECTURE PDFS**, or pick another course."
        stored = (f"{scope} has {status['pages']} saved page{'s' if status['pages'] != 1 else ''}"
                  + (f" across {status['lectures']} lectures" if status["lectures"] > 1 else ""))
        if status["text_pages"] == 0:
            return (f"{stored}, but none of them has selectable text (the PDF is probably scanned or image-only), "
                    "so there's nothing I can read yet. Turn on *Read handwriting on slides* under **⬡ MODELS** "
                    "and press **⟳ INDEX FOR SEARCH** so the pages are read from their images.")
        if pages_read:
            return (f"I read {pages_read} page{'s' if pages_read != 1 else ''} from {scope}, but they don't cover "
                    "that. Try naming a topic from the lecture, pick a broader scope, or switch this agent to "
                    "*Notes + general knowledge*.")
        unindexed = status["pages"] - status["indexed"]
        if unindexed:
            return (f"{stored}, but no page matched your words, and {unindexed} of them aren't indexed for "
                    f"search by meaning with {models.embedding_model} yet (needed once after switching models). "
                    "Press **⟳ INDEX FOR SEARCH**, then ask again. Or select a single lecture in the sidebar and "
                    "I'll read all of it.")
        return (f"{stored}, but none of them matched that question. Try naming a topic from the lecture, or "
                "select a single lecture in the sidebar and I'll read all of it.")

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
                self.show_search_results(rows, query, note="The model service is unreachable, so this used keyword search only.")

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
        provider = self.provider
        notes_mode = "strict" if agent["notes_mode"] == "strict" else "blend"  # never "off" in the main chat
        context_window = self._context_window()
        # Short follow-ups ("and how is it fixed?") have little to search on; borrow the previous question.
        previous = next((text for role, text in reversed(history) if role == "user"), "")
        search_query = f"{previous} {question}" if previous and len(question.split()) < 10 else question

        def answer(progress: TaskReporter) -> dict:
            progress.watch(models)
            sources: list[dict] = []
            # The main chat always answers from the selected notes (the overlay never does).
            if fixed_sources is not None:
                sources = [dict(s) for s in fixed_sources]
                label = f"USING {len(sources)} PAGE{'S' if len(sources) != 1 else ''} FROM THE ANSWER"
            else:
                progress.event({"state": "SEARCHING"})
                sources, label = self._gather_sources(models, search_query, progress, class_name,
                                                      lecture_name, provider)
            for index, source in enumerate(sources, start=1):
                source["source_id"] = f"S{index}"
            progress.event({"retrieval": label or "NO READABLE PAGES IN THIS SCOPE"})
            if not sources and fixed_sources is None and notes_mode == "strict":
                # Nothing to answer from: say exactly why instead of asking the model.
                return {"answer": self._explain_no_pages(class_name, lecture_name, models, 0),
                        "sources": [], "answerable": False, "tokens_per_second": 0.0,
                        "context_used": 0.0, "model": model}
            progress.event({"state": "THINKING", "notice": f"Waiting for {model}"})
            progress(f"{model} is writing…")
            try:
                result = models.stream_answer(
                    question, sources, model=model, persona=agent["system_prompt"], notes_mode=notes_mode,
                    temperature=float(agent["temperature"]), num_ctx=context_window,
                    reasoning=bool(agent["reasoning"]), history=history,
                    on_token=progress.token, on_thinking=progress.think,
                )
            except AnswerStopped:
                # Keep what was written; "blend" keeps real citations without the not-found check.
                mode = "blend"
                partial = finalize_answer(ChatResult(content="".join(progress.streamed)), sources, mode,
                                          context_window)
                return {**partial, "answer": (partial["answer"] + "\n\n*Stopped.*").strip()}
            if not result.get("answerable", True):  # read pages, but they don't cover it: say what was read
                result["answer"] = self._explain_no_pages(class_name, lecture_name, models, len(sources))
            if fixed_sources is None:  # saved under exactly this selection: lecture, course, or everything
                library = Library(database)
                try:
                    library.save_class_chat(class_name or "", question, result["answer"], result["sources"],
                                            agent["id"], lecture_name=lecture_name or "")
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
            window = self._context_window()
            text = f"{context:.0%} of " + ("1M" if window >= 1_000_000 else f"{window // 1024}k")
            prompt, cached = result.get("prompt_tokens", 0), result.get("cached_tokens", 0)
            if prompt and cached:  # Gemini reused the slides from its cache instead of re-reading them
                text += f" · {min(1.0, cached / prompt):.0%} CACHED"
            self.stat_context.setText(text)
            self.context_meter.set_value(context)
            answered = result.get("model") or model
            note = f" ({model} was overloaded, so {answered} answered)" if answered != model else ""
            self.statusBar().showMessage(f"{answered} · {speed:.1f} tokens/s{note}", 12000)
            QTimer.singleShot(0, self._refresh_message_layout)

        def failed(error: str) -> None:
            self._streaming_message = None
            message.kind = "error"
            message.finish(f"⚠ {error}", [])

        self._start_waiting("Reading your notes" if fixed_sources is None else f"Waiting for {model}")
        if not self._run_task(answer, show, on_error=failed, state="THINKING", stoppable=True):
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
            progress.watch(models, notices=False)
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
            progress.watch(models, notices=False)
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
            progress.watch(models, notices=False)
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

    # -- screen overlay ------------------------------------------------------------------------
    def shortcut_settings(self) -> dict:
        return {key: self.settings.value(f"shortcuts/{key}", default, type=str) or default
                for key, default in DEFAULT_SHORTCUTS.items()}

    def apply_shortcuts(self) -> None:
        """(Re)register the global shortcuts and show them in the overlay. Remembers any another app owns."""
        values = self.shortcut_settings()
        actions = {
            "toggle": lambda: QTimer.singleShot(0, self.toggle_overlay),  # deferred: Carbon calls in mid-event
            "read": lambda: QTimer.singleShot(0, self.read_screen_now),
        }
        self.failed_shortcuts = []
        if self.hotkeys is not None:
            self.hotkeys.clear()
            for key, action in actions.items():
                try:
                    shortcut = Shortcut.parse(values[key])
                except ValueError:
                    shortcut = Shortcut.parse(DEFAULT_SHORTCUTS[key])
                if shortcut is not None and not self.hotkeys.register(shortcut, action) and sys.platform == "darwin":
                    self.failed_shortcuts.append(shortcut.label)
        labels = {key: shortcut_label(values[key]) for key in values}
        self.overlay.set_shortcuts(labels["toggle"], labels["read"])
        tip = "Float a see-through panel over your screen that reads it"
        self.overlay_button.setToolTip(f"{tip} ({labels['toggle']})" if labels["toggle"] != "OFF" else tip)
        if self.failed_shortcuts:
            self.statusBar().showMessage(
                f"{' and '.join(self.failed_shortcuts)} is already used by another app. "
                "Choose another in the overlay's ⚙.", 15000)

    def configure_shortcuts(self) -> None:
        dialog = ShortcutsDialog(self.shortcut_settings(), self.failed_shortcuts, self.overlay)
        dialog.setStyleSheet(stylesheet(self.current_agent["accent"]))
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        for key, value in dialog.values().items():
            self.settings.setValue(f"shortcuts/{key}", value)
        self.apply_shortcuts()

    def toggle_overlay(self) -> None:
        self.overlay.toggle()

    def _update_overlay_model(self) -> None:
        """Show the vision model in the overlay, flagged when the provider says it isn't available."""
        model = self.model_settings()["vision_model"]
        installed = getattr(self, "vision_models", None)
        if model and installed is not None and not is_installed(model, installed):
            missing = "NOT AVAILABLE" if self.provider == GEMINI else "NOT INSTALLED"
            self.overlay.set_model(f"{model} · {missing}")
        else:
            self.overlay.set_model(model)

    def read_screen_now(self) -> None:
        """The global "read screen" shortcut: answer in the overlay's current mode, whether or not it's open."""
        self.read_screen("", self.overlay.mode(), False)

    def _open_source_from_overlay(self, row: dict) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()
        self.open_source_page(row)

    def read_screen(self, question: str = "", mode: str = "solve", pick_area: bool = False) -> None:
        """Capture the screen (or an area the user drags out) without the overlay in it, then answer."""
        if self.screen_task is not None:
            return
        if not screen_capture_allowed():
            request_screen_capture()
            self.overlay.show_overlay()
            message = self.overlay.add_message(
                "assistant",
                "Cortex needs **Screen Recording** permission to read your screen. Turn it on in System "
                "Settings → Privacy & Security → Screen Recording (for Cortex, or for Terminal if you start it "
                "with `cortex`), then quit and reopen the app.",
            )
            message.kind = "error"
            QDesktopServices.openUrl(QUrl(SCREEN_SETTINGS_URL))
            return
        self.settings.setValue("overlay/mode", mode)
        was_visible = self.overlay.isVisible()
        center = self.overlay.geometry().center() if was_visible else self.cursor().pos()
        screen = QGuiApplication.screenAt(center) or QGuiApplication.primaryScreen()
        area = screen.geometry()
        self.overlay.set_busy(True, "PICK AN AREA…" if pick_area else "CAPTURING…")
        self.overlay.hide()  # keep the panel itself out of the screenshot
        if pick_area:
            self._pick_area_then_answer(question, mode)
        else:
            QTimer.singleShot(250, lambda: self._capture_and_answer(question, mode, area))

    def _pick_area_then_answer(self, question: str, mode: str) -> None:
        picker = BackgroundTask(lambda _progress: capture_interactive())
        self.screen_task = picker

        def picked() -> None:
            self.screen_task = None
            picker.deleteLater()
            if picker.error is not None:
                self._screen_failed(picker.error)
            elif picker.result is None:  # Esc in the picker
                self.overlay.show_overlay()
                self.overlay.set_busy(False)
            else:
                self._answer_screen(question, mode, self._shrink_png(picker.result))

        picker.finished.connect(picked)
        picker.start()

    @staticmethod
    def _shrink_png(png: bytes, max_side: int = 1600) -> bytes:
        """Downscale big Retina captures: vision models resize anyway, and smaller images answer faster."""
        image = QImage.fromData(png)
        if image.isNull() or max(image.width(), image.height()) <= max_side:
            return png
        image = image.scaled(max_side, max_side, Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.SmoothTransformation)
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        image.save(buffer, "PNG")
        return bytes(buffer.data())

    def _screen_failed(self, error: str) -> None:
        self.overlay.show_overlay()
        self.overlay.set_busy(False)
        self.overlay.add_message("assistant", f"⚠ I couldn't capture the screen: {error}").kind = "error"

    def _capture_and_answer(self, question: str, mode: str, area) -> None:
        try:
            image = self._shrink_png(capture_region(area.x(), area.y(), area.width(), area.height()))
        except ScreenCaptureError as error:
            self._screen_failed(str(error))
            return
        self._answer_screen(question, mode, image)

    def _answer_screen(self, question: str, mode: str, image: bytes) -> None:
        self.overlay.show_overlay()
        history = self.overlay.history()
        label = MODES.get(mode, MODES["solve"])[0]
        self.overlay.add_message("user", question or f"◉ {label.title()} what's on my screen")
        message = self.overlay.add_message("assistant", "", streaming=True)
        agent = dict(self.current_agent)
        models = self._models()
        class_name, lecture_name = self.current_scope
        prompt = build_screen_prompt(mode, question)

        def work(progress: TaskReporter) -> dict:
            progress.watch(models, notices=False)
            sources: list[dict] = []  # the overlay only looks at the screen: no lecture notes
            progress(f"{models.vision_model or 'The vision model'} is reading your screen…")
            try:
                return models.stream_screen_answer(
                    prompt, image, sources, persona=agent["system_prompt"], num_ctx=CONTEXT_WINDOW,
                    history=history, on_token=progress.token,
                )
            except AnswerStopped:
                partial = finalize_answer(ChatResult(content="".join(progress.streamed)), sources,
                                          "blend" if sources else "off", CONTEXT_WINDOW)
                return {**partial, "answer": (partial["answer"] + "\n\n*Stopped.*").strip()}

        task = BackgroundTask(work)
        self.screen_task = task
        task.progress.connect(lambda text: self.overlay.set_busy(True, text.upper()[:44], stoppable=True))
        task.chunk.connect(message.append_token)

        def finished() -> None:
            self.screen_task = None
            self.overlay.set_busy(False)
            task.deleteLater()
            if task.error is not None:
                message.kind = "error"
                message.finish(f"⚠ {task.error}\n\nPick a vision model under ⬡ MODELS if you haven't yet.", [])
            else:
                message.finish(task.result["answer"], task.result["sources"])
            QTimer.singleShot(0, self.overlay.scroll_to_bottom)

        task.finished.connect(finished)
        self.overlay.set_busy(True, "READING SCREEN…", stoppable=True)
        task.start()

    def stop_screen_answer(self) -> None:
        if self.screen_task is not None:
            self.screen_task.stop_event.set()

    # -- shutdown ------------------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802
        self.closing = True
        if self.hotkeys:
            self.hotkeys.close()
        self.overlay.close()
        if self.screen_task and self.screen_task.isRunning():
            self.screen_task.stop_event.set()
            self.screen_task.wait(3000)
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
