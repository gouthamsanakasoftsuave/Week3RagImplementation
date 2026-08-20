"""Optional OCR helpers for image-only PDF pages and image files."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


def ocr_enabled() -> bool:
    return os.getenv("ENABLE_OCR", "true").lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def configure_tesseract() -> str | None:
    """
    Locate the Tesseract binary.
    Returns the path if usable, otherwise None.
    """
    try:
        import pytesseract
    except ImportError:
        return None

    candidates = [
        os.getenv("TESSERACT_CMD") or "",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "tesseract",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if candidate != "tesseract" and not path.exists():
            continue
        pytesseract.pytesseract.tesseract_cmd = candidate
        try:
            pytesseract.get_tesseract_version()
            return candidate
        except Exception:
            continue
    return None


def tesseract_available() -> bool:
    return configure_tesseract() is not None


def ocr_pil_image(image) -> str:
    """Run Tesseract OCR on a PIL image. Returns empty string if OCR unavailable."""
    if not ocr_enabled():
        return ""
    if not configure_tesseract():
        return ""
    import pytesseract

    text = pytesseract.image_to_string(image) or ""
    return text.strip()


def ocr_pdf_page(pdf_path: str | Path, page_number: int, dpi: int = 200) -> str:
    """
    Render one PDF page (1-based) to an image with PyMuPDF, then OCR it.
    Avoids Poppler (easier on Windows than pdf2image).
    """
    if not ocr_enabled() or not configure_tesseract():
        return ""

    import fitz  # PyMuPDF
    from PIL import Image

    doc = fitz.open(str(pdf_path))
    try:
        if page_number < 1 or page_number > doc.page_count:
            return ""
        page = doc.load_page(page_number - 1)
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return ocr_pil_image(image)
    finally:
        doc.close()


def ocr_image_file(path: str | Path) -> str:
    if not ocr_enabled() or not configure_tesseract():
        return ""
    from PIL import Image

    with Image.open(path) as img:
        return ocr_pil_image(img.convert("RGB"))
