import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from study_assistant.rich_text import latex_to_unicode, math_image, render_markdown
from study_assistant.theme import COLORS

APP = QApplication.instance() or QApplication([])


class RichTextTests(unittest.TestCase):
    def render(self, text: str) -> str:
        return render_markdown(text, COLORS, "Menlo")

    def test_inline_and_display_math_become_images(self):
        html = self.render("Euler: $e^{i\\pi} + 1 = 0$ and\n\n$$\\sum_{k=1}^{n} k = \\frac{n(n+1)}{2}$$")
        self.assertNotIn("$", html)
        self.assertGreaterEqual(html.count("<img"), 2)

    def test_bracket_delimiters_are_math_too(self):
        html = self.render("Inline \\(x^2\\) and display \\[y = mx + b\\]")
        self.assertNotIn("\\(", html)
        self.assertGreaterEqual(html.count("<img"), 2)

    def test_prices_are_left_alone(self):
        html = self.render("It costs $5 or $10 at the store.")
        self.assertIn("$5 or $10", html)
        self.assertNotIn("<img", html)

    def test_code_blocks_are_highlighted_and_math_inside_code_is_not(self):
        html = self.render("```python\ndef f(x):\n    return '$a$' + x\n```")
        self.assertIn("python", html.lower())
        self.assertIn("$a$", html)
        self.assertNotIn("<img", html)
        self.assertIn("color", html)  # Pygments colours

    def test_inline_code_and_unclosed_fence_while_streaming(self):
        html = self.render("Use `x = $y$` then\n```js\nconsole.log(1)")
        self.assertIn("x = $y$", html)
        self.assertIn("console", html)

    def test_markdown_still_renders(self):
        html = self.render("**bold** and a list:\n\n- one\n- two")
        self.assertIn("font-weight", html)
        self.assertIn("<li", html)

    def test_math_image_is_cached_png(self):
        first = math_image("x^2", "#ffffff")
        self.assertIsNotNone(first)
        self.assertTrue(first[0].endswith(".png") and os.path.exists(first[0]))
        self.assertEqual(first, math_image("x^2", "#ffffff"))

    def test_finished_code_blocks_get_a_copy_link_that_copies_just_that_block(self):
        from study_assistant.widgets import ChatMessageWidget

        text = "First:\n\n```python\nprint(1)\n```\n\nSecond:\n\n```js\nconsole.log(2)\n```"
        self.assertNotIn("COPY", render_markdown(text, COLORS, "Menlo"))  # streaming: no links yet
        self.assertEqual(render_markdown(text, COLORS, "Menlo", copy_links=True).count("⧉ COPY"), 2)
        message = ChatMessageWidget("assistant", text, [])
        message.body.linkActivated.emit("copy:1")
        self.assertEqual(APP.clipboard().text(), "console.log(2)")
        self.assertIn("✓ COPIED", message.body.text())
        message.body.linkActivated.emit("https://example.com")  # anything else is ignored
        self.assertEqual(APP.clipboard().text(), "console.log(2)")

    def test_unicode_fallback(self):
        self.assertEqual(latex_to_unicode("\\alpha \\leq \\beta^2"), "α ≤ β²")


if __name__ == "__main__":
    unittest.main()
