"""Split documents into overlapping chunks."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.loader import Document

# Opening “who signed” blocks are often split mid-sentence (Week 5 t01).


def extract_party_preamble(text: str) -> tuple[str | None, str]:
    """Keep FIRST PART / SECOND PART names in one chunk; return (preamble, remainder)."""
    if not text:
        return None, text
    head = text[:4000]
    if "FIRST PART" not in head.upper() and "EIH LIMITED" not in head.upper():
        return None, text
    end = -1
    lower = head.lower()
    idx = lower.find("collectively as")
    if idx != -1:
        # skip past “Parties” even if a newline sits between the words
        end = min(len(text), idx + 80)
    if end == -1:
        idx = head.find("WHEREAS")
        end = idx if idx != -1 else min(1800, len(text))
    preamble = text[:end].strip()
    rest = text[end:].strip()
    if len(preamble) < 120:
        return None, text
    return preamble, rest


@dataclass
class Chunk:
    content: str
    source: str
    chunk_id: str
    metadata: dict


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    keep_preamble: bool = True,
) -> list[Chunk]:
    """
    Split documents into smaller pieces.

    Chunk size matters: too large → noisy retrieval; too small → missing context.
    Overlap keeps sentences that span a boundary from being cut in half.
    keep_preamble stores the party-name opening as its own chunk (Week 6 fix).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: list[Chunk] = []
    for doc_idx, doc in enumerate(documents):
        pieces: list[str] = []
        body = doc.content
        if keep_preamble:
            preamble, rest = extract_party_preamble(doc.content)
            if preamble:
                pieces.append(preamble)
                body = rest
        pieces.extend(splitter.split_text(body) if body else [])
        for piece_idx, piece in enumerate(pieces):
            page = doc.metadata.get("page")
            chunk_id = f"{doc.source}::p{page or 1}::c{piece_idx}"
            meta = {
                **doc.metadata,
                "source": doc.source,
                "doc_index": doc_idx,
                "chunk_index": piece_idx,
            }
            chunks.append(
                Chunk(
                    content=piece,
                    source=doc.source,
                    chunk_id=chunk_id,
                    metadata=meta,
                )
            )
    return chunks
