"""A translucent, always-on-top panel that reads the screen and works through what's on it.

It's an ordinary window: it shows in screen recordings and screen sharing like any other app.
"""

from __future__ import annotations

from PySide6.QtCore import QByteArray, QPoint, QRectF, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QKeyEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizeGrip,
    QVBoxLayout,
    QWidget,
)

from .macos import float_above_everything
from .theme import COLORS
from .widgets import ChatMessageWidget, ComposerInput

# How the overlay should help. HINT is for practising: it nudges without giving the answer away.
MODES: dict[str, tuple[str, str]] = {
    "solve": (
        "SOLVE",
        "Explain what the problem on my screen is asking, then work through a full solution step by step. "
        "For coding problems give the approach, the code, and its time and space complexity.",
    ),
    "hint": (
        "HINT",
        "Give me one hint for the problem on my screen: the key idea or the next step, without the full "
        "solution or final answer. Keep it to a few sentences so I can keep working on it myself.",
    ),
    "explain": (
        "EXPLAIN",
        "Explain the concepts shown on my screen clearly, with a small example. Don't solve any exercise "
        "for me unless I ask.",
    ),
}
DEFAULT_SCREEN_QUESTION = MODES["solve"][1]
PANEL_OPACITY = 0.82  # see-through enough to show what's behind, solid enough to read on any background


def build_screen_prompt(mode: str, question: str) -> str:
    """The instruction sent with the screenshot: the mode's default, or the typed question in that mode."""
    label, default = MODES.get(mode, MODES["solve"])
    if not question:
        return default
    if mode == "hint":
        return f"{question}\n\n(Answer with a hint only: no full solution or final answer.)"
    return question


class _DragHandle(QFrame):
    """The title bar: drag it to move the frameless overlay."""

    def __init__(self, window: QWidget) -> None:
        super().__init__(window)
        self._window = window
        self._offset: QPoint | None = None
        self.setCursor(Qt.CursorShape.SizeAllCursor)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._offset = event.globalPosition().toPoint() - self._window.frameGeometry().topLeft()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._offset is not None:
            self._window.move(event.globalPosition().toPoint() - self._offset)

    def mouseReleaseEvent(self, _event) -> None:  # noqa: N802
        if self._offset is not None:
            self._offset = None
            self._window.save_state()


class ScreenOverlay(QWidget):
    """Floating, see-through study panel. The main window does the capturing and answering."""

    read_requested = Signal(str, str, bool)  # question ("" = mode default), mode, pick an area?
    stop_requested = Signal()
    settings_requested = Signal()
    source_opened = Signal(object)

    def __init__(self, parent: QWidget | None = None, settings: QSettings | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint)
        self.setObjectName("screenOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle("Cortex overlay")
        self.setMinimumSize(360, 320)
        self.resize(440, 580)
        self.settings = settings
        self._panel_alpha = PANEL_OPACITY
        self._accent = QColor(COLORS["cyan"])
        self._placed = False
        self.agent_name = "Assistant"
        self.messages: list[ChatMessageWidget] = []
        self.busy = False
        self.toggle_label = self.read_label = ""
        self.model_label = ""
        self._save_timer = QTimer(self)  # created first: the slider's setRange already emits valueChanged
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(400)
        self._save_timer.timeout.connect(self.save_state)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(8)

        handle = _DragHandle(self)
        head = QHBoxLayout(handle)
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(4)
        handle.setToolTip("Drag to move")
        head.addStretch(1)  # the empty space is the drag area: no title, to keep the panel uncluttered
        self.settings_button = QPushButton("⚙")
        self.settings_button.setObjectName("ghostButton")
        self.settings_button.setToolTip("Overlay shortcuts")
        self.settings_button.clicked.connect(self.settings_requested.emit)
        head.addWidget(self.settings_button)
        self.clear_button = QPushButton("CLEAR")
        self.clear_button.setObjectName("ghostButton")
        self.clear_button.clicked.connect(self.clear)
        head.addWidget(self.clear_button)
        self.close_button = QPushButton("✕")
        self.close_button.setObjectName("ghostButton")
        self.close_button.clicked.connect(self.hide)
        head.addWidget(self.close_button)
        outer.addWidget(handle)

        self.hint = QLabel()
        self.hint.setObjectName("overlayHint")
        self.hint.setWordWrap(True)
        outer.addWidget(self.hint)

        self.scroll = QScrollArea()
        self.scroll.setObjectName("overlayScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("overlayBody")
        self.body_layout = QVBoxLayout(body)
        self.body_layout.setContentsMargins(0, 0, 4, 0)
        self.body_layout.setSpacing(14)
        self.body_layout.addStretch(1)
        self.scroll.setWidget(body)
        outer.addWidget(self.scroll, 1)

        modes = QHBoxLayout()
        modes.setSpacing(4)
        self.mode_group = QButtonGroup(self)
        self.mode_buttons: dict[str, QPushButton] = {}
        for key, (label, _prompt) in MODES.items():
            button = QPushButton(label)
            button.setObjectName("modeButton")
            button.setCheckable(True)
            button.setToolTip(_prompt)
            self.mode_group.addButton(button)
            self.mode_buttons[key] = button
            modes.addWidget(button)
        self.mode_buttons["solve"].setChecked(True)
        modes.addStretch(1)
        self.status = QLabel("")
        self.status.setObjectName("overlayHint")
        modes.addWidget(self.status)
        outer.addLayout(modes)

        self.input = ComposerInput()
        self.input.setPlaceholderText("Optional: ask something specific about it")
        self.input.submitted.connect(lambda: self._submit(False))
        outer.addWidget(self.input)
        actions = QHBoxLayout()
        actions.setSpacing(6)
        self.area_button = QPushButton("⬚ AREA")
        self.area_button.setToolTip("Drag out the part of the screen to read (Esc cancels)")
        self.area_button.clicked.connect(lambda: self._submit(True))
        actions.addWidget(self.area_button)
        actions.addStretch(1)
        self.read_button = QPushButton("◉ READ SCREEN")
        self.read_button.setObjectName("sendButton")
        self.read_button.clicked.connect(self._read_or_stop)
        actions.addWidget(self.read_button)
        grip = QSizeGrip(self)
        actions.addWidget(grip, 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)
        outer.addLayout(actions)

        self.extra_css = f"""
            #overlayHint {{ color: {COLORS['muted']}; font-size: 11px; }}
            #overlayScroll, #overlayBody {{ background: transparent; border: none; }}
            #composerInput {{ background: rgba(0, 0, 0, 0.35); border: 1px solid {COLORS['border_hi']}; }}
        """
        self.setStyleSheet(self.extra_css)
        self._restoring = True
        self.restore_state()
        self._restoring = False
        self.set_shortcuts("", "")

    # -- persistence ------------------------------------------------------------------------
    def restore_state(self) -> None:
        if self.settings is not None:
            geometry = self.settings.value("overlay/geometry")
            if isinstance(geometry, QByteArray) and not geometry.isEmpty() and self.restoreGeometry(geometry):
                self._placed = self._on_a_screen()
            mode = self.settings.value("overlay/mode", "solve", type=str)
            if mode in self.mode_buttons:
                self.mode_buttons[mode].setChecked(True)

    def save_state(self) -> None:
        if self.settings is None or getattr(self, "_restoring", True):
            return
        self.settings.setValue("overlay/geometry", self.saveGeometry())
        self.settings.setValue("overlay/mode", self.mode())

    def _on_a_screen(self) -> bool:
        """A saved spot can sit on a monitor that's no longer plugged in."""
        center = self.frameGeometry().center()
        return any(screen.availableGeometry().contains(center) for screen in QGuiApplication.screens())

    def moveEvent(self, event) -> None:  # noqa: N802
        super().moveEvent(event)
        self._save_timer.start()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._save_timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802
        self.save_state()
        super().hideEvent(event)

    # -- appearance ---------------------------------------------------------------------------
    def set_agent(self, name: str, _accent: str = "") -> None:
        self.agent_name = name  # shown on the answers it writes

    def set_model(self, model: str) -> None:
        self.model_label = model
        if not self.busy:
            self.status.setText(f"VISION · {model}" if model else "NO VISION MODEL")

    def set_shortcuts(self, toggle_label: str, read_label: str) -> None:
        """Show the current global shortcuts in the hint and tooltips ("" or "OFF" when unset)."""
        self.toggle_label, self.read_label = toggle_label, read_label
        parts = ["Put a problem on screen, then press READ SCREEN, or AREA to pick just part of it."]
        if toggle_label and toggle_label != "OFF":
            parts.append(f"{toggle_label} shows or hides this panel from any app.")
        if read_label and read_label != "OFF":
            parts.append(f"{read_label} reads the screen straight away.")
        self.hint.setText(" ".join(parts))
        self.close_button.setToolTip(f"Hide ({toggle_label} brings it back)" if toggle_label not in ("", "OFF")
                                     else "Hide")
        self.read_button.setToolTip(f"Read the whole screen ({read_label})" if read_label not in ("", "OFF")
                                    else "Read the whole screen")

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(rect, 6, 6)
        fill = QColor(COLORS["bg"])
        fill.setAlphaF(self._panel_alpha)
        painter.fillPath(path, fill)
        border = QColor(self._accent)
        border.setAlphaF(0.55)
        painter.setPen(QPen(border, 1))
        painter.drawPath(path)
        painter.end()

    def show_overlay(self) -> None:
        if not self._placed:
            screen = QGuiApplication.screenAt(self.pos()) or QGuiApplication.primaryScreen()
            if screen is not None:
                area = screen.availableGeometry()
                self.move(area.right() - self.width() - 24, area.top() + 48)
            self._placed = True
        self.show()
        self.raise_()
        self.activateWindow()
        float_above_everything(int(self.winId()))
        self.input.setFocus()

    show_near_top_right = show_overlay  # older name

    def toggle(self) -> None:
        if self.isVisible():
            self.hide()
        else:
            self.show_overlay()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            if self.busy:
                self.stop_requested.emit()
            else:
                self.hide()
            return
        super().keyPressEvent(event)

    # -- conversation -------------------------------------------------------------------------
    def mode(self) -> str:
        return next((key for key, button in self.mode_buttons.items() if button.isChecked()), "solve")

    def _read_or_stop(self) -> None:
        if self.busy:
            self.stop_requested.emit()
        else:
            self._submit(False)

    def _submit(self, pick_area: bool) -> None:
        if self.busy:
            return
        question = self.input.text().strip()
        self.input.clear()
        self.read_requested.emit(question, self.mode(), pick_area)

    def set_busy(self, busy: bool, status: str = "", stoppable: bool = False) -> None:
        self.busy = busy
        self.area_button.setEnabled(not busy)
        self.read_button.setEnabled(not busy or stoppable)
        self.read_button.setText("■ STOP" if busy and stoppable else "◉ READ SCREEN")
        if busy:
            self.status.setText(status)
        else:
            self.set_model(self.model_label)

    def add_message(self, role: str, text: str, *, streaming: bool = False) -> ChatMessageWidget:
        self.hint.hide()
        message = ChatMessageWidget(role, text, None, self, agent_name=self.agent_name,
                                    accent=self._accent.name(), streaming=streaming)
        if role == "assistant":
            message.action_buttons["regenerate"].hide()
            message.action_buttons["explain"].hide()
        message.source_opened.connect(self.source_opened.emit)
        self.body_layout.insertWidget(self.body_layout.count() - 1, message)
        self.messages.append(message)
        QTimer.singleShot(0, self.scroll_to_bottom)
        return message

    def scroll_to_bottom(self) -> None:
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def history(self) -> list[tuple[str, str]]:
        return [(m.role, m.text) for m in self.messages if m.text and getattr(m, "kind", "chat") == "chat"]

    def clear(self) -> None:
        for message in self.messages:
            self.body_layout.removeWidget(message)
            message.hide()
            message.deleteLater()
        self.messages.clear()
        self.hint.show()
