"""Dark, terminal-inspired visual theme."""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QFontDatabase

APP_NAME = "Cortex"
APP_TAGLINE = "LOCAL NOTES INTERFACE // v2.0"

COLORS = {
    "bg": "#07070a",
    "panel": "#0c0c11",
    "raised": "#111118",
    "card": "#14141c",
    "border": "#23232d",
    "border_hi": "#3a3a48",
    "text": "#e7e7ee",
    "muted": "#8b8b99",
    "dim": "#5b5b68",
    "ok": "#52e3a4",
    "warn": "#f5c451",
    "bad": "#ff5d73",
    "white": "#f4f4f8",
}

ACCENTS = ["#9b87f5", "#2ee6d6", "#ff6ac1", "#f5c451", "#52e3a4", "#e8e8ee"]

_MONO_CANDIDATES = ["JetBrains Mono", "SF Mono", "Menlo", "IBM Plex Mono", "Consolas", "DejaVu Sans Mono"]
_SANS_CANDIDATES = ["Inter", "SF Pro Text", ".AppleSystemUIFont", "Helvetica Neue", "Segoe UI", "DejaVu Sans"]


def _first_available(candidates: list[str], fallback: str) -> str:
    families = set(QFontDatabase.families())
    for name in candidates:
        if name in families:
            return name
    return fallback


def mono_family() -> str:
    return _first_available(_MONO_CANDIDATES, "monospace")


def sans_family() -> str:
    return _first_available(_SANS_CANDIDATES, "sans-serif")


def mono_font(point_size: float = 9, bold: bool = False, spacing: float = 1.5) -> QFont:
    font = QFont(mono_family())
    font.setPointSizeF(point_size)
    font.setBold(bold)
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    return font


def with_alpha(color: str, alpha: float) -> str:
    c = QColor(color)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {alpha:.2f})"


def creativity_label(temperature: float) -> str:
    if temperature < 0.35:
        return "PRECISE"
    if temperature < 0.75:
        return "BALANCED"
    return "CREATIVE"


def stylesheet(accent: str) -> str:
    c = COLORS
    mono = mono_family()
    sans = sans_family()
    accent_soft = with_alpha(accent, 0.14)
    accent_mid = with_alpha(accent, 0.45)
    return f"""
    QWidget {{ color: {c['text']}; font-family: "{sans}"; font-size: 13px; }}
    QMainWindow, QDialog {{ background: {c['bg']}; }}
    QToolTip {{ background: {c['raised']}; color: {c['text']}; border: 1px solid {c['border_hi']}; padding: 6px; }}

    #sidebar {{ background: {c['panel']}; border-right: 1px solid {c['border']}; }}
    #rightPanel {{ background: {c['panel']}; border-left: 1px solid {c['border']}; }}
    #centerPanel, #chatBody, #chatScroll {{ background: {c['bg']}; border: none; }}
    #header {{ background: {c['bg']}; border-bottom: 1px solid {c['border']}; }}

    #brandTitle {{ font-family: "{mono}"; font-size: 20px; letter-spacing: 9px; color: {c['white']}; }}
    #brandRule {{ background: {c['white']}; }}
    #brandTag, #monoMuted, #sectionHeading, #footerNote, #statLabel, #hintText {{
        font-family: "{mono}"; color: {c['muted']}; font-size: 10px; letter-spacing: 1.5px; }}
    #sectionHeading {{ color: {c['dim']}; font-size: 10px; letter-spacing: 2px; }}
    #onlineStatus {{ font-family: "{mono}"; font-size: 10px; letter-spacing: 1.5px; }}
    #statValue {{ font-family: "{mono}"; color: {c['white']}; font-size: 10px; font-weight: 700; }}
    #agentTitle {{ font-family: "{mono}"; font-size: 16px; letter-spacing: 3px; color: {c['white']}; }}
    #panelName {{ font-family: "{mono}"; font-size: 18px; letter-spacing: 5px; color: {c['white']}; }}
    #panelSubject {{ font-family: "{mono}"; font-size: 9px; letter-spacing: 2px; color: {c['muted']}; }}
    #panelPrompt {{ color: {c['muted']}; font-size: 12px; line-height: 150%; }}
    #scopeChip {{ font-family: "{mono}"; font-size: 10px; letter-spacing: 1.5px; color: {accent};
        border: 1px solid {accent_mid}; background: {accent_soft}; padding: 3px 8px; }}

    QPushButton {{ font-family: "{mono}"; font-size: 10px; letter-spacing: 1.5px; font-weight: 700;
        background: transparent; color: {c['text']}; border: 1px solid {c['border_hi']}; padding: 7px 12px; }}
    QPushButton:hover {{ border-color: {accent}; color: {c['white']}; background: {accent_soft}; }}
    QPushButton:pressed {{ background: {accent_mid}; }}
    QPushButton:disabled {{ color: {c['dim']}; border-color: {c['border']}; }}
    QPushButton:checked {{ border-color: {accent}; color: {accent}; background: {accent_soft}; }}
    #sendButton {{ background: {c['white']}; color: #000; border: none; padding: 10px 18px; }}
    #sendButton:hover {{ background: {accent}; color: #000; }}
    #sendButton:disabled {{ background: {c['border_hi']}; color: {c['dim']}; }}
    #ghostButton {{ border: none; color: {c['muted']}; padding: 4px 6px; }}
    #ghostButton:hover {{ color: {c['white']}; background: transparent; }}
    #modeButton {{ border: 1px solid {c['border']}; color: {c['muted']}; padding: 5px 9px; }}
    #modeButton:checked {{ border-color: {accent}; color: {accent}; }}
    #sourceLink {{ font-family: "{mono}"; font-size: 10px; font-weight: 400; letter-spacing: 0.5px; text-align: left;
        color: {accent}; border: 1px solid {c['border']}; background: {c['raised']}; padding: 5px 9px; }}
    #sourceLink:hover {{ border-color: {accent}; }}
    #chipButton {{ font-weight: 400; color: {c['muted']}; border: 1px solid {c['border']};
        background: {c['raised']}; padding: 4px 9px; }}

    #agentCard {{ background: transparent; border: 1px solid transparent; }}
    #agentCard:hover {{ border: 1px solid {c['border_hi']}; }}
    #agentCard[active="true"] {{ background: {c['card']}; border: 1px solid {c['border_hi']}; }}
    #agentName {{ font-size: 13px; font-weight: 600; color: {c['white']}; }}
    #agentModel {{ font-family: "{mono}"; font-size: 10px; color: {c['muted']}; }}
    #tag {{ font-family: "{mono}"; font-size: 8px; letter-spacing: 1px; font-weight: 700; color: {c['muted']};
        border: 1px solid {c['border_hi']}; padding: 1px 4px; }}

    #assistantMessage, #userMessage {{ background: transparent; border: none; }}
    #userBubble {{ background: {c['card']}; border: 1px solid {c['border']}; }}
    #messageHeading {{ font-family: "{mono}"; font-size: 10px; letter-spacing: 2px; font-weight: 700; color: {c['white']}; }}
    #messageRule {{ background: {c['border']}; }}
    #retrievalLine {{ font-family: "{mono}"; font-size: 10px; color: {c['muted']}; }}
    #thinkingText {{ color: {c['muted']}; font-size: 12px; border-left: 2px solid {c['border_hi']}; padding-left: 10px; }}
    #messageBody {{ font-size: 14px; color: {c['text']}; }}

    #composer {{ background: {c['panel']}; border: 1px solid {c['border_hi']}; }}
    #composer[focused="true"] {{ border: 1px solid {accent_mid}; }}
    #composerInput {{ background: transparent; border: none; font-size: 14px; color: {c['text']}; }}

    QLineEdit, QPlainTextEdit, QTextBrowser, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget, QTableWidget, QTreeWidget {{
        background: {c['raised']}; color: {c['text']}; border: 1px solid {c['border']}; padding: 5px;
        selection-background-color: {accent_mid}; }}
    QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus {{ border: 1px solid {accent}; }}
    QComboBox QAbstractItemView {{ background: {c['raised']}; border: 1px solid {c['border_hi']}; selection-background-color: {accent_soft}; }}
    QComboBox::drop-down {{ border: none; width: 18px; }}
    #courseNavigator {{ background: transparent; border: none; padding: 0; }}
    QTreeWidget::item, QListWidget::item {{ padding: 5px 4px; }}
    QTreeWidget::item:selected, QListWidget::item:selected {{ background: {accent_soft}; color: {c['white']}; }}
    QTreeWidget::item:hover, QListWidget::item:hover {{ background: {c['card']}; }}
    QHeaderView::section {{ background: {c['panel']}; color: {c['muted']}; border: none; padding: 5px;
        font-family: "{mono}"; font-size: 10px; letter-spacing: 1px; }}

    QTabBar::tab {{ font-family: "{mono}"; font-size: 10px; letter-spacing: 2px; font-weight: 700; color: {c['muted']};
        background: transparent; padding: 9px 14px; border: none; border-bottom: 2px solid transparent; }}
    QTabBar::tab:selected {{ color: {c['white']}; border-bottom: 2px solid {c['white']}; }}
    QTabWidget::pane {{ border: none; border-top: 1px solid {c['border']}; }}

    #pdfPreview {{ background: {c['raised']}; border: 1px solid {c['border']}; color: {c['muted']}; }}
    QCheckBox, QRadioButton {{ spacing: 8px; }}
    QCheckBox::indicator, QRadioButton::indicator {{ width: 13px; height: 13px; border: 1px solid {c['border_hi']}; background: {c['raised']}; }}
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {{ background: {accent}; border-color: {accent}; }}
    QSlider::groove:horizontal {{ height: 2px; background: {c['border_hi']}; }}
    QSlider::handle:horizontal {{ background: {c['white']}; width: 10px; margin: -6px 0; }}
    QSlider::sub-page:horizontal {{ background: {accent}; }}
    QSplitter::handle {{ background: {c['border']}; }}
    QScrollArea {{ background: transparent; border: none; }}
    QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {c['border_hi']}; min-height: 30px; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical, QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ height: 0; background: none; }}
    QStatusBar {{ background: {c['panel']}; color: {c['muted']}; font-family: "{mono}"; font-size: 10px; border-top: 1px solid {c['border']}; }}
    QMessageBox QLabel {{ color: {c['text']}; }}
    #dialogHelp {{ color: {c['muted']}; }}
    """
