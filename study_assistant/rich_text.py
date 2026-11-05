"""Turn a model's Markdown answer into Qt rich text, with rendered math and highlighted code.

Qt's own Markdown support covers headings, lists, emphasis, and tables, but it shows LaTeX as raw
"$\\frac{a}{b}$" and code blocks as plain grey text. This module:

* pulls code blocks and math out before Markdown sees them (so ``*`` and ``_`` inside them survive),
* renders math with matplotlib's mathtext to crisp PNGs in the theme's text colour (cached on disk),
  falling back to readable Unicode ("α ≤ β", "√(x)") for LaTeX that mathtext can't draw,
* highlights code with Pygments inside a dark panel labelled with its language,
* converts the rest with Qt's Markdown reader and puts the pieces back.
"""

from __future__ import annotations

import hashlib
import html
import math
import re
import tempfile
from functools import lru_cache
from pathlib import Path

from PySide6.QtGui import QTextDocument

MATH_FONTSET = "cm"  # Computer Modern, like LaTeX
STROKE_WEIGHT = 0.6  # points of extra outline on each glyph; 0 is the font's own (thin) weight
MATH_POINT_SIZE = 17  # inline math; CM glyphs run small, so this sits level with the 15px answer text
DISPLAY_POINT_SIZE = 22
_SCALE = 2  # render at 2x so formulas stay sharp on Retina screens

_TOKEN = "CTXTOKEN{}X"
_TOKEN_PATTERN = re.compile(r"CTXTOKEN(\d+)X")

_FENCE = re.compile(r"^([ \t]*)(```|~~~)[ \t]*([\w+#.-]*)[^\n]*\n(.*?)(?:^\1\2[ \t]*$|\Z)", re.MULTILINE | re.DOTALL)
_INLINE_CODE = re.compile(r"(`+)(.+?)\1", re.DOTALL)
_DISPLAY_MATH = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]", re.DOTALL)
# Inline $…$ follows Pandoc's rule so prices ("$5 and $10") aren't mistaken for math: no space just
# inside the dollars, and the closing $ isn't followed by a digit.
_INLINE_MATH = re.compile(r"\\\((.+?)\\\)|(?<![\\$\w])\$(?!\s)([^$\n]+?)(?<![\s\\])\$(?!\d)")


def _cache_dir() -> Path:
    folder = Path(tempfile.gettempdir()) / "cortex-math"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# ---------------------------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------------------------

def _normalise_latex(latex: str) -> str:
    """Rewrite common LaTeX that mathtext doesn't know into forms it does."""
    latex = latex.strip()
    latex = re.sub(r"\\(?:displaystyle|textstyle|limits|nolimits|,|;|!|quad|qquad)\b", " ", latex)
    latex = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg)(?![a-zA-Z])", "", latex)
    latex = latex.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    latex = latex.replace("\\le ", "\\leq ").replace("\\ge ", "\\geq ").replace("\\ne ", "\\neq ")
    latex = re.sub(r"\\operatorname\{([^}]*)\}", r"\\mathrm{\1}", latex)
    latex = re.sub(r"\\text(?:bf|it|rm)?\{([^}]*)\}", lambda m: r"\mathrm{" + m.group(1).replace(" ", r"\ ") + "}",
                   latex)
    return latex


# Qt centres an inline image on the text's middle, about this far (px) above the baseline at 15px.
TEXT_MIDDLE = 4.0


def _draw_formula(mathtext: str, path: Path, size: float, color: str, inline: bool = False) -> None:
    """Like matplotlib's math_to_image, but on a transparent background so it sits on the dark chat."""
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import MathTextParser

    from matplotlib import rc_context

    # Computer Modern: the font LaTeX (and KaTeX in ChatGPT / Claude / Gemini) sets math in.
    with rc_context({"mathtext.fontset": MATH_FONTSET}):
        prop = FontProperties(size=size)
        width, height, depth, *_ = MathTextParser("path").parse(mathtext, dpi=72, prop=prop)
        pad = 2.0  # points of air so descenders and hats aren't clipped
        above, below = height - depth + pad, depth + pad
        if inline:  # pad so the formula's baseline lands on the text's baseline once Qt centres it
            if above - below > 2 * TEXT_MIDDLE:
                below = above - 2 * TEXT_MIDDLE
            else:
                above = below + 2 * TEXT_MIDDLE
        # Whole pixels, so the 1x and 2x copies line up exactly.
        total = math.ceil(above + below)
        above = total - below
        full_width = math.ceil(width + 2 * pad)
        figure = Figure(figsize=(full_width / 72, total / 72))
        from matplotlib import patheffects

        # Computer Modern's hairlines vanish as light-on-dark; a thin outline in the same colour thickens
        # every stroke (fraction bars and root signs included) without changing the letter shapes.
        figure.text(pad / full_width, below / total, mathtext, fontproperties=prop, color=color,
                    path_effects=[patheffects.withStroke(linewidth=STROKE_WEIGHT, foreground=color)])
        # One copy per screen density. Qt picks name@2x.png on Retina by itself, so neither screen
        # gets a shrunk or stretched image (shrinking is what made thin strokes break up).
        figure.savefig(str(path), dpi=72, format="png", transparent=True)
        figure.savefig(str(_retina(path)), dpi=72 * _SCALE, format="png", transparent=True)


def _retina(path: Path) -> Path:
    return path.with_name(f"{path.stem}@{_SCALE}x{path.suffix}")


@lru_cache(maxsize=512)
def math_image(latex: str, color: str, display: bool = False) -> tuple[str, int, int] | None:
    """Render one formula to a PNG; return (path, logical width, logical height), or None if mathtext
    can't draw it. Results are cached in memory and on disk."""
    size = DISPLAY_POINT_SIZE if display else MATH_POINT_SIZE
    key = hashlib.sha1(f"{latex}|{color}|{size}|{_SCALE}|{MATH_FONTSET}|{STROKE_WEIGHT}|6".encode()).hexdigest()[:20]
    path = _cache_dir() / f"{key}.png"
    if not (path.exists() and _retina(path).exists()):
        try:
            formula = _normalise_latex(latex)
            # mathtext sets fractions at script size, unreadably small; draw them full size like KaTeX
            formula = formula.replace("\\frac", "\\dfrac")
            _draw_formula(f"${formula}$", path, size, color, inline=not display)
        except Exception:  # unsupported LaTeX, or matplotlib missing: the caller falls back to text
            path.unlink(missing_ok=True)
            _retina(path).unlink(missing_ok=True)
            return None
    try:
        from PySide6.QtGui import QImageReader

        dimensions = QImageReader(str(path)).size()
    except Exception:
        return None
    if not dimensions.isValid():
        return None
    return str(path), max(1, dimensions.width()), max(1, dimensions.height())


_SYMBOLS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ",
    "eta": "η", "theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν",
    "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "φ", "varphi": "φ",
    "chi": "χ", "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
    "Pi": "Π", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
    "cdot": "·", "times": "×", "div": "÷", "pm": "±", "mp": "∓", "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥",
    "neq": "≠", "ne": "≠", "approx": "≈", "equiv": "≡", "sim": "∼", "propto": "∝", "infty": "∞",
    "sum": "∑", "prod": "∏", "int": "∫", "oint": "∮", "partial": "∂", "nabla": "∇",
    "to": "→", "rightarrow": "→", "leftarrow": "←", "Rightarrow": "⇒", "Leftarrow": "⇐", "iff": "⇔",
    "Leftrightarrow": "⇔", "leftrightarrow": "↔", "mapsto": "↦", "implies": "⇒",
    "in": "∈", "notin": "∉", "subset": "⊂", "subseteq": "⊆", "supset": "⊃", "supseteq": "⊇", "cup": "∪",
    "cap": "∩", "emptyset": "∅", "varnothing": "∅", "forall": "∀", "exists": "∃", "neg": "¬", "lnot": "¬",
    "land": "∧", "wedge": "∧", "lor": "∨", "vee": "∨", "oplus": "⊕", "otimes": "⊗", "circ": "∘",
    "ldots": "…", "cdots": "⋯", "dots": "…", "vdots": "⋮", "ddots": "⋱", "angle": "∠", "perp": "⊥",
    "mid": "∣", "parallel": "∥", "lceil": "⌈", "rceil": "⌉", "lfloor": "⌊", "rfloor": "⌋", "langle": "⟨",
    "rangle": "⟩", "log": "log", "ln": "ln", "sin": "sin", "cos": "cos", "tan": "tan", "exp": "exp",
    "max": "max", "min": "min", "lim": "lim", "det": "det", "mod": "mod", "bmod": "mod", "gcd": "gcd",
}
_BLACKBOARD = {"R": "ℝ", "N": "ℕ", "Z": "ℤ", "Q": "ℚ", "C": "ℂ", "P": "ℙ", "E": "𝔼"}
_SUPER = str.maketrans("0123456789+-=()niT", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱᵀ")
_SUB = str.maketrans("0123456789+-=()aeijkmnoprstx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑᵢⱼₖₘₙₒₚᵣₛₜₓ")


def latex_to_unicode(latex: str) -> str:
    """A readable plain-text version of LaTeX, for formulas mathtext can't render."""
    text = _normalise_latex(latex)
    text = re.sub(r"\\begin\{[a-z*]+\}|\\end\{[a-z*]+\}", "", text)
    text = text.replace("\\\\", "; ").replace("&", " ")
    for _ in range(3):  # nested fractions and roots, innermost first
        text = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", text)
        text = re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", text)
    text = re.sub(r"\\mathbb\{(\w)\}", lambda m: _BLACKBOARD.get(m.group(1), m.group(1)), text)
    text = re.sub(r"\\(?:mathrm|mathbf|mathit|mathcal|boldsymbol|bar|hat|vec|overline)\{([^{}]*)\}", r"\1", text)

    def script(match: re.Match, table) -> str:
        body = match.group(1) or match.group(2)
        converted = body.translate(table)
        if all(ch != src for ch, src in zip(converted, body)) or body.translate(table) != body and len(body) == 1:
            return converted
        return ("^" if table is _SUPER else "_") + (f"({body})" if len(body) > 1 else body)

    text = re.sub(r"\^\{([^{}]*)\}|\^(\w)", lambda m: script(m, _SUPER), text)
    text = re.sub(r"_\{([^{}]*)\}|_(\w)", lambda m: script(m, _SUB), text)
    text = re.sub(r"\\([A-Za-z]+)", lambda m: _SYMBOLS.get(m.group(1), m.group(1)), text)
    text = text.replace("{", "").replace("}", "").replace("\\", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def _math_html(latex: str, display: bool, color: str, mono: str) -> str:
    lines = [latex]
    if display and re.search(r"\\begin\{(?:aligned|align\*?|gathered|gather\*?|split|eqnarray\*?)\}", latex):
        body = re.sub(r"\\(?:begin|end)\{[a-z*]+\}", "", latex)
        lines = [line.replace("&", "") for line in body.split("\\\\") if line.strip()]
    parts = []
    for line in lines:
        image = math_image(line.strip(), color, display)
        if image:
            path, width, height = image
            parts.append(f'<img src="{html.escape(Path(path).as_uri())}" width="{width}" height="{height}" '
                         f'style="vertical-align: middle;">')
        else:
            parts.append(f'<span style="font-family: \'{mono}\'; color: {color};">'
                         f"{html.escape(latex_to_unicode(line))}</span>")
    if display:
        return "".join(f'<p align="center" style="margin-top: 2px; margin-bottom: 16px;">{part}</p>' for part in parts)
    return " ".join(parts)


# ---------------------------------------------------------------------------------------------
# Code
# ---------------------------------------------------------------------------------------------

def _highlight(code: str, language: str) -> str:
    try:
        from pygments import highlight
        from pygments.formatters import HtmlFormatter
        from pygments.lexers import TextLexer, get_lexer_by_name, guess_lexer

        try:
            lexer = get_lexer_by_name(language) if language else guess_lexer(code)
        except Exception:
            lexer = TextLexer()
        formatted = highlight(code, lexer, HtmlFormatter(nowrap=True, noclasses=True, style="monokai"))
        # Qt draws span backgrounds as boxes; the panel already has one.
        return re.sub(r"background(?:-color)?:\s*#[0-9a-fA-F]+;?\s*", "", formatted)
    except Exception:
        return html.escape(code)


COPY_LINK = "copy:"  # href of a code block's COPY link: copy:<index of the block in the answer>


def code_blocks(text: str) -> list[str]:
    """The fenced code blocks in an answer, in order: what each block's COPY link copies."""
    return [m.group(4).rstrip("\n") for m in _FENCE.finditer(text)]


def _code_block_html(code: str, language: str, colors: dict, mono: str, index: int | None = None,
                     copied: bool = False) -> str:
    code = code.rstrip("\n")
    small = f"font-family: '{mono}'; font-size: 11px;"
    copy = ""
    if index is not None:  # a finished answer: offer to copy the block
        copy = (f'<span style="{small} color: {colors["cyan"]};">✓ COPIED</span>' if copied else
                f'<a href="{COPY_LINK}{index}" style="{small} color: {colors["cyan"]}; text-decoration: none;">'
                "⧉ COPY</a>")
    label = ""
    if language or copy:
        label = (f'<table width="100%" cellspacing="0" cellpadding="0" style="margin-bottom: 6px;"><tr>'
                 f'<td style="{small} color: {colors["muted"]};">{html.escape(language.upper())}</td>'
                 f'<td align="right">{copy}</td></tr></table>')
    body = _highlight(code, language).rstrip("\n")
    return (f'<table width="100%" cellspacing="0" cellpadding="10" style="margin-top: 4px; margin-bottom: 18px;" '
            f'bgcolor="{colors["raised"]}"><tr><td style="border: 1px solid {colors["border"]};">{label}'
            f'<pre style="font-family: \'{mono}\'; font-size: 13px; color: {colors["text"]}; margin: 0; '
            f'white-space: pre-wrap;">'
            f"{body}</pre></td></tr></table>")


# ---------------------------------------------------------------------------------------------
# Whole answers
# ---------------------------------------------------------------------------------------------

LINE_HEIGHT = 165  # percent of the font height; Qt's default (~120%) reads as a wall of text
PARAGRAPH_GAP = 14
LIST_ITEM_GAP = 3
HEADING_GAP = 20


def _space_out(document: QTextDocument) -> None:
    """Give every block readable line height and paragraph spacing before converting to HTML."""
    from PySide6.QtGui import QTextBlockFormat, QTextCursor

    block = document.begin()
    while block.isValid():
        current = block.blockFormat()
        spaced = QTextBlockFormat()
        spaced.setLineHeight(LINE_HEIGHT, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value)
        if current.headingLevel():
            spaced.setTopMargin(HEADING_GAP)
            spaced.setBottomMargin(8)
        elif block.textList() is not None:
            next_block = block.next()
            last_item = not (next_block.isValid() and next_block.textList() is block.textList())
            spaced.setBottomMargin(PARAGRAPH_GAP if last_item else LIST_ITEM_GAP)
        else:
            spaced.setBottomMargin(PARAGRAPH_GAP)
        cursor = QTextCursor(block)
        cursor.mergeBlockFormat(spaced)
        block = block.next()


def warm_up() -> None:
    """Import matplotlib's math renderer and Pygments ahead of time (off the UI thread), so the first
    formula or code block in an answer doesn't stall the window for a second or two."""
    try:
        from matplotlib.figure import Figure  # the slow part is importing these
        from matplotlib.mathtext import MathTextParser

        from pygments.lexers import get_lexer_by_name

        assert Figure is not None
        MathTextParser("path")
        get_lexer_by_name("python")
    except Exception:
        pass


def render_markdown(text: str, colors: dict, mono: str, *, copy_links: bool = False,
                    copied: int | None = None) -> str:
    """Markdown with LaTeX math and fenced code → Qt rich-text HTML in the theme's colours."""
    pieces: list[str] = []

    def stash(fragment: str, block: bool = False) -> str:
        pieces.append(fragment)
        token = _TOKEN.format(len(pieces) - 1)
        return f"\n\n{token}\n\n" if block else token

    # Order matters: code first (its contents are literal), then display math, then inline pieces.
    blocks = iter(range(10_000))

    def code(match: re.Match) -> str:
        index = next(blocks)
        return stash(_code_block_html(match.group(4), match.group(3), colors, mono,
                                      index if copy_links else None, copied == index), True)

    text = _FENCE.sub(code, text)
    text = _INLINE_CODE.sub(lambda m: stash(
        f'<code style="font-family: \'{mono}\'; background-color: {colors["raised"]}; color: {colors["text"]};">'
        f"&nbsp;{html.escape(m.group(2).strip())}&nbsp;</code>"), text)
    text = _DISPLAY_MATH.sub(lambda m: stash(_math_html(m.group(1) or m.group(2), True, colors["white"], mono), True),
                             text)
    text = _INLINE_MATH.sub(lambda m: stash(_math_html(m.group(1) or m.group(2), False, colors["white"], mono)), text)

    document = QTextDocument()
    document.setMarkdown(text, QTextDocument.MarkdownFeature.MarkdownDialectGitHub)
    _space_out(document)
    body = document.toHtml()
    match = re.search(r"<body[^>]*>(.*)</body>", body, re.DOTALL)
    body = match.group(1) if match else body
    # A block token sits alone in its own paragraph: replace the whole paragraph, not just the token.
    body = re.sub(r"<p[^>]*>\s*(CTXTOKEN\d+X)\s*</p>",
                  lambda m: pieces[int(_TOKEN_PATTERN.match(m.group(1)).group(1))], body)
    return _TOKEN_PATTERN.sub(lambda m: pieces[int(m.group(1))], body)
