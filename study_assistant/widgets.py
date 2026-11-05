"""Custom widgets: generated agent portraits, agent cards, chat messages, composer."""

from __future__ import annotations

import hashlib
import math
import random

from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QKeyEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from .ollama import source_label
from .rich_text import COPY_LINK, code_blocks, render_markdown
from .theme import ACCENTS, COLORS, mono_family


# ---------------------------------------------------------------------------------------------
# Generated portrait
# ---------------------------------------------------------------------------------------------

def _seed(name: str) -> int:
    return int(hashlib.sha1(name.encode("utf-8")).hexdigest()[:8], 16)


def _inside_bust(x: float, y: float, style: int) -> float:
    """Return >0 inside a stylised head-and-shoulders silhouette (coords in [-1, 1])."""
    head = 1 - ((x / 0.34) ** 2 + ((y + 0.22) / 0.44) ** 2)
    neck = 1.0 if abs(x) < 0.15 and 0.12 < y < 0.45 else -1.0
    shoulders = 1 - ((x / 0.88) ** 2 + ((y - 1.02) / 0.58) ** 2)
    value = max(head, neck * 0.3, shoulders if y < 0.98 else -1)
    if style == 0:  # hood / headdress
        hood = 1.0 if abs(x) < 0.44 and -0.72 < y < 0.42 and (abs(x) > 0.3 or y < -0.5) else -1.0
        value = max(value, hood * 0.5)
    elif style == 1 and -0.32 < y < -0.18 and abs(x) < 0.3:  # visor gap
        value = -1.0
    return value


class AgentAvatar(QWidget):
    """A dot-matrix bust generated from the agent's name, with an animated link ring."""

    clicked = Signal()

    def __init__(self, name: str = "", accent: str = ACCENTS[0], size: int = 40, ring: bool = False,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._offline = False
        self._name = name
        self._accent = QColor(accent)
        self._ring = ring
        self._phase = 0.0
        self._busy = False
        self._dots: list[tuple[float, float, float]] = []
        self.setFixedSize(size, size)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._build()

    def set_agent(self, name: str, accent: str) -> None:
        self._name = name
        self._accent = QColor(accent)
        self._build()
        self.update()

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        if busy and self._ring:
            self._timer.start(40)
        else:
            self._timer.stop()
            self.update()

    def set_offline(self, offline: bool) -> None:
        self._offline = offline
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def _tick(self) -> None:
        self._phase = (self._phase + 0.06) % (2 * math.pi)
        self.update()

    def _build(self) -> None:
        seed = _seed(self._name or "agent")
        rng = random.Random(seed)
        style = seed % 3
        steps = max(14, min(64, self.width() // 3))
        dots = []
        for row in range(steps):
            for col in range(steps):
                x = -1 + 2 * (col + 0.5) / steps
                y = -1 + 2 * (row + 0.5) / steps
                inside = _inside_bust(x, y, style)
                if inside <= 0:
                    continue
                rim = max(0.0, 1 - inside * 3.2)  # brighter near the edge
                light = 0.55 + 0.45 * max(0.0, -x)  # key light from the left
                brightness = min(1.0, 0.18 + 0.55 * rim + 0.35 * light * rng.random())
                dots.append((x, y, brightness))
        self._dots = dots

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        center = QPointF(w / 2, w / 2)
        compact = self._ring and w < 120  # a small status badge: thinner ring, no tick marks
        if self._offline:
            painter.setOpacity(0.35)
        scale = w * (0.30 if compact else 0.36 if self._ring else 0.46)
        dot = max(1.0, w / 110)
        painter.setPen(Qt.PenStyle.NoPen)
        for x, y, brightness in self._dots:
            color = QColor(self._accent)
            shimmer = 0.85 + 0.15 * math.sin(self._phase * 2 + y * 7) if self._busy else 1.0
            color.setAlphaF(max(0.08, min(1.0, brightness * shimmer)))
            painter.setBrush(color)
            painter.drawEllipse(QPointF(center.x() + x * scale, center.y() + y * scale * 0.98), dot, dot)
        if compact:
            radius = w * 0.44
            faint = QColor(self._accent)
            faint.setAlphaF(0.25)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(faint, 1))
            painter.drawEllipse(center, radius, radius)
            painter.setPen(QPen(self._accent, 2 if self._busy else 1.5))
            rect = QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2)
            start = int((90 + math.degrees(self._phase) * 3) * 16)
            painter.drawArc(rect, start, int((-110 if self._busy else -300) * 16))
        elif self._ring:
            radius = w * 0.46
            faint = QColor(self._accent)
            faint.setAlphaF(0.18)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(faint, 1))
            painter.drawEllipse(center, radius + 6, radius + 6)
            painter.setPen(QPen(self._accent, 2))
            rect = QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2)
            start = int((90 + math.degrees(self._phase) * 2) * 16)
            painter.drawArc(rect, start, int(-300 * 16))
            tick = QColor(COLORS["muted"])
            painter.setPen(QPen(tick, 1))
            for index in range(0, 360, 15):
                angle = math.radians(index)
                inner, outer = radius + 10, radius + (14 if index % 90 == 0 else 12)
                painter.drawLine(
                    QPointF(center.x() + math.cos(angle) * inner, center.y() + math.sin(angle) * inner),
                    QPointF(center.x() + math.cos(angle) * outer, center.y() + math.sin(angle) * outer),
                )
        else:
            frame = QColor(COLORS["border_hi"])
            painter.setPen(QPen(frame, 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            # corner brackets, like a viewfinder
            length = w * 0.18
            for cx, cy, dx, dy in ((0, 0, 1, 1), (w - 1, 0, -1, 1), (0, w - 1, 1, -1), (w - 1, w - 1, -1, -1)):
                painter.drawLine(QPointF(cx, cy), QPointF(cx + dx * length, cy))
                painter.drawLine(QPointF(cx, cy), QPointF(cx, cy + dy * length))
        painter.end()


# ---------------------------------------------------------------------------------------------
# Sidebar agent card
# ---------------------------------------------------------------------------------------------

def mode_tags(agent: dict) -> list[str]:
    tags = {"strict": ["NOTES", "CITED"], "blend": ["NOTES", "GENERAL"], "off": ["CHAT"]}[agent["notes_mode"]]
    if agent.get("reasoning"):
        tags.append("REASON")
    return tags


class AgentCard(QFrame):
    clicked = Signal(int)

    def __init__(self, agent: dict, model_label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.agent_id = int(agent["id"])
        self.setObjectName("agentCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setProperty("active", False)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        self.avatar = AgentAvatar(agent["name"], agent["accent"], 40)
        layout.addWidget(self.avatar, 0, Qt.AlignmentFlag.AlignTop)
        text = QVBoxLayout()
        text.setSpacing(2)
        name = QLabel(agent["name"])
        name.setObjectName("agentName")
        model = QLabel(model_label)
        model.setObjectName("agentModel")
        tags = QHBoxLayout()
        tags.setSpacing(4)
        for tag in mode_tags(agent):
            chip = QLabel(tag)
            chip.setObjectName("tag")
            tags.addWidget(chip)
        tags.addStretch(1)
        text.addWidget(name)
        text.addWidget(model)
        text.addSpacing(3)
        text.addLayout(tags)
        layout.addLayout(text, 1)

    def set_active(self, active: bool) -> None:
        self.setProperty("active", active)
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.agent_id)
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------------------------
# Chat message
# ---------------------------------------------------------------------------------------------

class ChatMessageWidget(QFrame):
    """One chat turn. Assistant turns can stream tokens and reasoning, then show citations."""

    source_opened = Signal(object)
    explain_selected = Signal(str)
    regenerate_requested = Signal()

    def __init__(
        self,
        role: str,
        text: str,
        sources: list[dict] | None = None,
        parent: QWidget | None = None,
        *,
        agent_name: str = "Assistant",
        accent: str = ACCENTS[0],
        retrieval: str = "",
        streaming: bool = False,
    ) -> None:
        super().__init__(parent)
        self.role = role
        self.text = text
        self.sources = sources or []
        self.thinking = ""
        self.streaming = streaming
        self.accent = accent
        self.setObjectName("assistantMessage" if role == "assistant" else "userMessage")
        # Preferred, not Minimum: with Minimum the height estimated for some other width became a floor,
        # and those overestimates stacked up into blank space below reopened conversations.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 4, 0, 4)
        outer.setSpacing(8)

        heading_row = QHBoxLayout()
        heading_row.setSpacing(8)
        if role == "assistant":
            self.mini_avatar = AgentAvatar(agent_name, accent, 18)
            heading_row.addWidget(self.mini_avatar)
        heading = QLabel(agent_name.upper() if role == "assistant" else "YOU")
        heading.setObjectName("messageHeading")
        heading_row.addWidget(heading)
        rule = QFrame()
        rule.setObjectName("messageRule")
        rule.setFixedHeight(1)
        heading_row.addWidget(rule, 1, Qt.AlignmentFlag.AlignVCenter)
        outer.addLayout(heading_row)

        self.retrieval_label = QLabel()
        self.retrieval_label.setObjectName("retrievalLine")
        self.retrieval_label.setVisible(False)
        outer.addWidget(self.retrieval_label)
        self.set_retrieval(retrieval)

        self.thinking_button = QPushButton()
        self.thinking_button.setObjectName("chipButton")
        self.thinking_button.setCheckable(True)
        self.thinking_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.thinking_button.hide()
        self.thinking_label = QLabel()
        self.thinking_label.setObjectName("thinkingText")
        self.thinking_label.setWordWrap(True)
        self.thinking_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.thinking_label.hide()
        self.thinking_button.toggled.connect(self.thinking_label.setVisible)
        outer.addWidget(self.thinking_button, 0, Qt.AlignmentFlag.AlignLeft)
        outer.addWidget(self.thinking_label)

        self.body = QLabel()
        self.body.setObjectName("messageBody")
        self.body.setWordWrap(True)
        self.body.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        self.body.setOpenExternalLinks(False)
        self.body.linkActivated.connect(self._link_clicked)
        self._copied_block: int | None = None
        self.body.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        if role == "user":
            bubble = QFrame()
            bubble.setObjectName("userBubble")
            bubble_layout = QVBoxLayout(bubble)
            bubble_layout.setContentsMargins(14, 10, 14, 10)
            bubble_layout.addWidget(self.body)
            self.body.setTextFormat(Qt.TextFormat.PlainText)
            outer.addWidget(bubble)
        else:
            self.body.setTextFormat(Qt.TextFormat.RichText)  # Markdown → HTML with rendered math and code
            outer.addWidget(self.body)
        self._render_body()

        # The first source shows; the rest fold away under one toggle, so a heavily cited answer
        # doesn't end in a wall of links.
        self.first_source_host = QVBoxLayout()
        self.first_source_host.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(self.first_source_host)
        self.sources_toggle = QPushButton()
        self.sources_toggle.setObjectName("sourcesToggle")
        self.sources_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.sources_toggle.setCheckable(True)
        self.sources_toggle.toggled.connect(self._show_sources)
        outer.addWidget(self.sources_toggle, 0, Qt.AlignmentFlag.AlignLeft)
        self.sources_panel = QWidget()
        self.sources_host = QVBoxLayout(self.sources_panel)
        self.sources_host.setContentsMargins(0, 2, 0, 0)
        self.sources_host.setSpacing(4)
        self.sources_panel.setVisible(False)
        outer.addWidget(self.sources_panel)
        self.source_buttons: list[QPushButton] = []

        self.actions = QHBoxLayout()
        self.actions.setSpacing(2)
        self.action_buttons: dict[str, QPushButton] = {}
        if role == "assistant":
            for key, label, slot in (("copy", "COPY", self.copy_text),
                                     ("explain", "EXPLAIN SELECTED", self.explain_passage),
                                     ("regenerate", "↻ REGENERATE", self.regenerate_requested.emit)):
                button = QPushButton(label)
                self.action_buttons[key] = button
                button.setObjectName("ghostButton")
                button.setCursor(Qt.CursorShape.PointingHandCursor)
                button.clicked.connect(slot)
                self.actions.addWidget(button)
            self.actions.addStretch(1)
            self.action_host = QWidget()
            self.action_host.setLayout(self.actions)
            self.action_host.setVisible(not streaming)
            outer.addWidget(self.action_host)
        self.set_sources(self.sources)

    # -- streaming ---------------------------------------------------------------------------
    def set_retrieval(self, text: str) -> None:
        self.retrieval_label.setText(f"◆ {text}" if text else "")
        self.retrieval_label.setVisible(bool(text))

    RENDER_INTERVAL_MS = 90  # while streaming, redraw at most ~11×/s however fast words arrive

    def append_token(self, token: str) -> None:
        # Rendering re-lays out the whole answer (Markdown, highlighted code, math), so doing it per
        # token froze the window on long answers. Words are kept immediately; drawing is batched.
        self.text += token
        if not getattr(self, "_render_pending", False):
            self._render_pending = True
            QTimer.singleShot(self.RENDER_INTERVAL_MS, self._render_soon)

    def _render_soon(self) -> None:
        self._render_pending = False
        if self.streaming:
            self._render_body()

    def append_thinking(self, token: str) -> None:
        self.thinking += token
        self.thinking_label.setText(self.thinking.strip())
        words = len(self.thinking.split())
        self.thinking_button.setText(f"▸ REASONING · {words} WORDS")
        self.thinking_button.show()

    def finish(self, text: str, sources: list[dict]) -> None:
        self.streaming = False
        self.text = text
        self._render_body()
        self.set_sources(sources)
        if hasattr(self, "action_host"):
            self.action_host.show()

    def _render_body(self) -> None:
        shown = self.text
        if self.streaming:
            shown = (shown or "") + " ▍"
        self._show(shown)

    def _show(self, markdown: str) -> None:
        if self.role == "user":
            self.body.setText(markdown)
        else:
            colors = {**COLORS, "cyan": COLORS["muted"]} if getattr(self, "neutral", False) else COLORS
            self.body.setText(render_markdown(markdown, colors, mono_family(), copy_links=not self.streaming,
                                              copied=self._copied_block))

    def _link_clicked(self, link: str) -> None:
        """A code block's ⧉ COPY: put exactly that block's code on the clipboard."""
        if not link.startswith(COPY_LINK):
            return
        try:
            index = int(link.removeprefix(COPY_LINK))
            code = code_blocks(self.text)[index]
        except (ValueError, IndexError):
            return
        QApplication.clipboard().setText(code)
        self._copied_block = index
        self._render_body()

        def reset() -> None:
            if self._copied_block == index:
                self._copied_block = None
                self._render_body()

        QTimer.singleShot(1500, reset)

    def set_waiting(self, text: str) -> None:
        """While nothing has streamed yet, say what we're waiting on (and for how long)."""
        if self.streaming and not self.text:
            self._show(f"*{text}* ▍")

    # -- sources -------------------------------------------------------------------------------
    def set_sources(self, sources: list[dict]) -> None:
        self.sources = sources or []
        for button in self.source_buttons:
            button.setParent(None)
            button.deleteLater()
        self.source_buttons = []
        for index, source in enumerate(self.sources, start=1):
            button = QPushButton(f"[{index}]  {self._source_label(source)}")
            button.setObjectName("sourceLink")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setToolTip("Open this page in the side panel")
            button.clicked.connect(lambda _checked=False, page=source: self.source_opened.emit(page))
            self.source_buttons.append(button)
            host = self.first_source_host if index == 1 else self.sources_host
            host.addWidget(button, 0, Qt.AlignmentFlag.AlignLeft)
        self.sources_toggle.setVisible(len(self.sources) > 1)
        self._show_sources(self.sources_toggle.isChecked())

    def _show_sources(self, expanded: bool) -> None:
        more = len(self.sources) - 1
        arrow = "▾" if expanded else "▸"
        self.sources_toggle.setText(f"{arrow}  {more} MORE SOURCE{'S' if more != 1 else ''}")
        self.sources_toggle.setToolTip("Hide the other pages this answer cites" if expanded
                                       else "Show the other pages this answer cites")
        self.sources_panel.setVisible(expanded and more > 0)

    def set_available_width(self, width: int) -> None:
        maximum = max(280, width)
        self.setMaximumWidth(maximum)
        for button in self.source_buttons:
            button.setMaximumWidth(maximum)

    @staticmethod
    def _source_label(source: dict) -> str:
        return source_label(source)

    # -- actions -----------------------------------------------------------------------------
    def copy_text(self) -> None:
        selected = self.body.selectedText().strip()
        QApplication.clipboard().setText(selected or self.text)

    def explain_passage(self) -> None:
        selected = self.body.selectedText().strip()
        if not selected:
            QMessageBox.information(self, "Select some text", "Highlight part of the answer first, then choose Explain selected.")
            return
        self.explain_selected.emit(selected)


# ---------------------------------------------------------------------------------------------
# Composer and small bits
# ---------------------------------------------------------------------------------------------

class ComposerInput(QPlainTextEdit):
    """Multi-line input: Enter sends, Shift+Enter adds a newline; grows with its content."""

    submitted = Signal()
    focus_changed = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("composerInput")
        self.setTabChangesFocus(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.document().contentsChanged.connect(self._fit)
        self._fit()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (
            event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self.submitted.emit()
            return
        super().keyPressEvent(event)

    def focusInEvent(self, event) -> None:  # noqa: N802
        super().focusInEvent(event)
        self.focus_changed.emit(True)

    def focusOutEvent(self, event) -> None:  # noqa: N802
        super().focusOutEvent(event)
        self.focus_changed.emit(False)

    def _fit(self) -> None:
        lines = max(1, min(7, int(self.document().size().height())))
        height = int(self.fontMetrics().lineSpacing() * lines + 16)
        self.setFixedHeight(max(40, height))

    # QLineEdit-compatible helpers used by the window and tests
    def text(self) -> str:
        return self.toPlainText()

    def setText(self, text: str) -> None:  # noqa: N802
        self.setPlainText(text)


class MeterBar(QWidget):
    """A thin horizontal meter used for speed and context gauges in the header."""

    def __init__(self, width: int = 44, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._value = 0.0
        self.setFixedSize(QSize(width, 4))

    def set_value(self, value: float) -> None:
        self._value = max(0.0, min(1.0, value))
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(COLORS["border_hi"]))
        filled = self.rect().adjusted(0, 0, -int(self.width() * (1 - self._value)), 0)
        painter.fillRect(filled, QColor(COLORS["cyan"]))
        painter.end()


# ---------------------------------------------------------------------------------------------
# Course navigator rows
# ---------------------------------------------------------------------------------------------

NAV_KIND = Qt.ItemDataRole.UserRole + 1  # "all" | "course" | "lecture"
NAV_DETAIL = Qt.ItemDataRole.UserRole + 2  # small right-aligned note: page or lecture count


class NavigatorDelegate(QStyledItemDelegate):
    """Draws the course list: soft rounded selection with a slim accent marker, bold course rows with
    a chevron (click it to fold the course), indented lectures with their page count on the right."""

    CHEVRON_WIDTH = 26

    def __init__(self, sans: str, mono: str, parent=None) -> None:
        super().__init__(parent)
        self.sans, self.mono = sans, mono

    def sizeHint(self, option, index) -> QSize:  # noqa: N802
        kind = index.data(NAV_KIND)
        return QSize(option.rect.width(), 30 if kind == "lecture" else 36)

    def editorEvent(self, event, model, option, index) -> bool:  # noqa: N802
        # A click on a course's chevron folds or unfolds it (the row itself still selects the course).
        if (event.type() == QEvent.Type.MouseButtonRelease and index.data(NAV_KIND) == "course"
                and event.position().x() < option.rect.left() + self.CHEVRON_WIDTH):
            tree = option.widget
            tree.setExpanded(index, not tree.isExpanded(index))
            return True
        return super().editorEvent(event, model, option, index)

    def paint(self, painter: QPainter, option, index) -> None:
        kind = index.data(NAV_KIND) or "lecture"
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(option.rect).adjusted(2, 2, -2, -2)
        if selected or hovered:
            path = QPainterPath()
            path.addRoundedRect(rect, 6, 6)
            painter.fillPath(path, QColor(COLORS["card"] if selected else COLORS["raised"]))
        if selected:  # slim accent marker, inset from the corners
            marker = QRectF(rect.left() + 1, rect.top() + rect.height() * 0.25, 3, rect.height() * 0.5)
            path = QPainterPath()
            path.addRoundedRect(marker, 1.5, 1.5)
            painter.fillPath(path, QColor(COLORS["cyan"]))

        x = rect.left() + 12
        right = rect.right() - 10
        middle = rect.center().y()
        if kind == "course":
            tree = option.widget
            expanded = tree.isExpanded(index) if tree is not None else True
            pen = QPen(QColor(COLORS["muted"]), 1.6)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            cx = x + 4
            if expanded:  # ▾
                painter.drawPolyline([QPointF(cx - 4, middle - 2), QPointF(cx, middle + 2), QPointF(cx + 4, middle - 2)])
            else:  # ▸
                painter.drawPolyline([QPointF(cx - 2, middle - 4), QPointF(cx + 2, middle), QPointF(cx - 2, middle + 4)])
            x += self.CHEVRON_WIDTH - 10
        elif kind == "all":
            painter.setPen(QPen(QColor(COLORS["cyan"] if selected else COLORS["muted"]), 1.4))
            painter.setBrush(QColor(COLORS["cyan"]) if selected else Qt.BrushStyle.NoBrush)
            size = 4.5
            painter.drawPolygon([QPointF(x + 4, middle - size), QPointF(x + 4 + size, middle),
                                 QPointF(x + 4, middle + size), QPointF(x + 4 - size, middle)])
            x += 16
        else:
            x += 18  # lectures sit under their course's name

        detail = index.data(NAV_DETAIL) or ""
        if detail:
            small = QFont(self.mono)
            small.setPixelSize(11)
            painter.setFont(small)
            painter.setPen(QColor(COLORS["muted"] if selected else COLORS["dim"]))
            width = QFontMetrics(small).horizontalAdvance(detail)
            painter.drawText(QRectF(right - width, rect.top(), width, rect.height()),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, detail)
            right -= width + 10

        font = QFont(self.sans)
        font.setPixelSize(14 if kind != "lecture" else 13)
        font.setWeight(QFont.Weight.DemiBold if kind != "lecture" else QFont.Weight.Normal)
        painter.setFont(font)
        colour = COLORS["white"] if selected else (COLORS["text"] if kind != "lecture" else COLORS["muted"])
        if hovered and not selected:
            colour = COLORS["white"]
        painter.setPen(QColor(colour))
        name = QFontMetrics(font).elidedText(index.data(Qt.ItemDataRole.DisplayRole) or "",
                                             Qt.TextElideMode.ElideRight, int(right - x))
        painter.drawText(QRectF(x, rect.top(), right - x, rect.height()),
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, name)
        painter.restore()
