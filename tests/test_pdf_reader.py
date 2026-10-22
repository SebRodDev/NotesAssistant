import tempfile
import unittest
from pathlib import Path

import pymupdf

from study_assistant.pdf_reader import extract_page_texts, render_page


class PdfReaderTests(unittest.TestCase):
    def test_extracts_pages_and_renders_preview(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "lecture.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "A page about deadlock prevention")
            document.new_page()
            document.save(path)
            document.close()

            pages = extract_page_texts(path)
            image = render_page(path, 1)

        self.assertEqual(len(pages), 2)
        self.assertIn("deadlock prevention", pages[0])
        self.assertEqual(pages[1], "")
        self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_rejects_out_of_range_page(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "lecture.pdf"
            document = pymupdf.open()
            document.new_page()
            document.save(path)
            document.close()
            with self.assertRaises(ValueError):
                render_page(path, 0)


if __name__ == "__main__":
    unittest.main()
