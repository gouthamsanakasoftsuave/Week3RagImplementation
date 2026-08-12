"""End-to-end RAG pipeline: load → chunk → embed → retrieve → generate."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from rag.chunking import Chunk, chunk_documents
from rag.generate import RagAnswer, generate_answer
from rag.loader import Document, load_directory
from rag.store import RetrievedChunk, VectorStore

load_dotenv()


@dataclass
class IngestResult:
    documents: int
    chunks: int
    sources: list[str]


class RagPipeline:
    def __init__(
        self,
        documents_dir: str | Path = "documents",
        persist_dir: str | Path = "chroma_db",
    ) -> None:
        self.documents_dir = Path(documents_dir)
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))
        self.top_k = int(os.getenv("TOP_K", "4"))
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
        self.store = VectorStore(
            embedding_model_name=os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            persist_dir=persist_dir,
        )

    def ingest(self, rebuild: bool = True) -> IngestResult:
        docs = load_directory(self.documents_dir)
        chunks = chunk_documents(
            docs,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        if rebuild:
            self.store.clear()
        count = self.store.add_chunks(chunks)
        sources = sorted({d.source for d in docs})
        return IngestResult(documents=len(docs), chunks=count, sources=sources)

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        source_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        k = top_k or self.top_k
        results = self.store.search(question, top_k=k, source_filter=source_filter)
        # Drop weak matches so the model is not tempted to invent from noise
        return [r for r in results if r.score >= self.similarity_threshold]

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        source_filter: str | None = None,
    ) -> RagAnswer:
        chunks = self.retrieve(question, top_k=top_k, source_filter=source_filter)
        return generate_answer(question, chunks)

    def status(self) -> dict:
        return {
            "indexed_chunks": self.store.count,
            "documents_dir": str(self.documents_dir.resolve()),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
        }
