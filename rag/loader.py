"""Load text from PDF, TXT, MD, and (optionally) image files via OCR."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from rag.ocr import (
    ocr_enabled,
    ocr_image_file,
    ocr_pdf_page,
    tesseract_available,
)


@dataclass
class Document:
    content: str
    source: str
    metadata: dict


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}

# If digital text is shorter than this, try OCR (scanned/image-heavy pages)
OCR_MIN_CHARS = int(os.getenv("OCR_MIN_CHARS", "40"))


def load_file(path: Path) -> list[Document]:
    """Load one file into Document objects (one per PDF page or whole file)."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in IMAGE_EXTENSIONS:
        return _load_image(path)
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return []
        return [
            Document(
                content=text,
                source=path.name,
                metadata={"path": str(path), "type": suffix.lstrip("."), "extraction": "text"},
            )
        ]
    raise ValueError(f"Unsupported file type: {suffix}")


def load_directory(directory: str | Path) -> list[Document]:
    """Load all supported documents from a folder (non-recursive)."""
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Documents folder not found: {root}")

    documents: list[Document] = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.extend(load_file(path))
    return documents


def _load_pdf(path: Path) -> list[Document]:
    reader = PdfReader(str(path))
    docs: list[Document] = []
    use_ocr = ocr_enabled() and tesseract_available()

    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        extraction = "text"

        if use_ocr and len(text) < OCR_MIN_CHARS:
            ocr_text = ocr_pdf_page(path, i)
            if len(ocr_text) > len(text):
                text = ocr_text
                extraction = "ocr"

        if not text:
            continue

        docs.append(
            Document(
                content=text,
                source=path.name,
                metadata={
                    "path": str(path),
                    "type": "pdf",
                    "page": i,
                    "extraction": extraction,
                },
            )
        )
    return docs


def _load_image(path: Path) -> list[Document]:
    if not (ocr_enabled() and tesseract_available()):
        return []
    text = ocr_image_file(path)
    if not text:
        return []
    return [
        Document(
            content=text,
            source=path.name,
            metadata={
                "path": str(path),
                "type": "image",
                "page": 1,
                "extraction": "ocr",
            },
        )
    ]
