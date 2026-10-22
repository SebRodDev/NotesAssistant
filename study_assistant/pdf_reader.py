"""PDF text extraction and page rendering helpers."""

from pathlib import Path

import pymupdf


def extract_page_texts(file_path: Path) -> list[str]:
    """Extract embedded text page by page, preserving page order."""
    with pymupdf.open(file_path) as document:
        if document.is_encrypted:
            raise ValueError("This PDF is password-protected. Unlock it before importing.")
        if document.page_count == 0:
            raise ValueError("This PDF has no pages.")
        return [page.get_text("text").strip() for page in document]


def render_page(file_path: Path, page_number: int, max_dimension: int = 1400) -> bytes:
    """Render a one-based PDF page as PNG bytes for the page preview."""
    with pymupdf.open(file_path) as document:
        if page_number < 1 or page_number > document.page_count:
            raise ValueError("The requested page is outside this PDF.")
        page = document[page_number - 1]
        scale = min(max_dimension / max(page.rect.width, page.rect.height), 2.0)
        return page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).tobytes("png")
