"""Split documents into overlapping chunks."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.loader import Document


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
) -> list[Chunk]:
    """
    Split documents into smaller pieces.

    Chunk size matters: too large → noisy retrieval; too small → missing context.
    Overlap keeps sentences that span a boundary from being cut in half.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: list[Chunk] = []
    for doc_idx, doc in enumerate(documents):
        pieces = splitter.split_text(doc.content)
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
