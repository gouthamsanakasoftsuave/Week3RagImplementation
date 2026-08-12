"""Load text from PDF, TXT, and MD files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Document:
    content: str
    source: str
    metadata: dict


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_file(path: Path) -> list[Document]:
    """Load one file into Document objects (one per PDF page or whole file)."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return []
        return [
            Document(
                content=text,
                source=path.name,
                metadata={"path": str(path), "type": suffix.lstrip(".")},
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
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
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
                },
            )
        )
    return docs
