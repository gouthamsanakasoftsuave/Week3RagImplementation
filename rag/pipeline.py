"""End-to-end RAG pipeline: load → chunk → embed → retrieve → generate."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from rag.chunking import chunk_documents
from rag.generate import RagAnswer, generate_answer
from rag.loader import load_directory
from rag.observability import (
    chunks_payload,
    get_langfuse,
    session_id,
    trace_tags,
    tracing_enabled,
)
from rag.query import rewrite_query
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
        self.default_mode = os.getenv("RETRIEVAL_MODE", "hybrid")
        self.store = VectorStore(
            embedding_model_name=os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            persist_dir=persist_dir,
        )

    def ingest(self, rebuild: bool = True, keep_preamble: bool = True) -> IngestResult:
        docs = load_directory(self.documents_dir)
        chunks = chunk_documents(
            docs,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            keep_preamble=keep_preamble,
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
        mode: str | None = None,
        rewrite: bool = True,
    ) -> list[RetrievedChunk]:
        k = top_k or self.top_k
        retrieval_mode = (mode or self.default_mode).lower()
        query = rewrite_query(question) if rewrite else question
        payload_in = {
            "question": question,
            "rewritten_query": query,
            "mode": retrieval_mode,
            "top_k": k,
            "source_filter": source_filter,
        }

        def _search() -> list[RetrievedChunk]:
            results = self.store.search(
                query,
                top_k=k,
                source_filter=source_filter,
                mode=retrieval_mode,
            )
            # Semantic scores are cosine similarity; hybrid RRF scores are tiny — don't reuse 0.35
            if retrieval_mode == "semantic":
                return [r for r in results if r.score >= self.similarity_threshold]
            return results

        if not tracing_enabled():
            return _search()

        from langfuse import propagate_attributes

        lf = get_langfuse()
        with propagate_attributes(
            session_id=session_id(),
            tags=trace_tags("retrieve"),
            metadata={"mode": retrieval_mode},
            version=os.getenv("LANGFUSE_RELEASE", "week5-error-analysis"),
        ):
            with lf.start_as_current_observation(
                as_type="retriever",
                name="retrieve",
                input=payload_in,
            ) as span:
                results = _search()
                span.update(output=chunks_payload(results))
                return results

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        source_filter: str | None = None,
        mode: str | None = None,
        rewrite: bool = True,
    ) -> RagAnswer:
        retrieval_mode = (mode or self.default_mode).lower()
        k = top_k or self.top_k
        payload_in = {
            "question": question,
            "mode": retrieval_mode,
            "top_k": k,
            "source_filter": source_filter,
        }

        def _run() -> RagAnswer:
            chunks = self.retrieve(
                question,
                top_k=top_k,
                source_filter=source_filter,
                mode=mode,
                rewrite=rewrite,
            )
            return generate_answer(question, chunks)

        if not tracing_enabled():
            return _run()

        from langfuse import propagate_attributes

        lf = get_langfuse()
        with lf.start_as_current_observation(
            as_type="chain",
            name="rag-ask",
            input=payload_in,
        ) as root:
            with propagate_attributes(
                session_id=session_id(),
                tags=trace_tags("ask"),
                metadata={"mode": retrieval_mode},
                version=os.getenv("LANGFUSE_RELEASE", "week5-error-analysis"),
            ):
                answer = _run()
                root.update(
                    output={
                        "answer": answer.answer,
                        "grounded": answer.grounded,
                        "sources": answer.sources,
                    }
                )
                return answer

    def status(self) -> dict:
        return {
            "indexed_chunks": self.store.count,
            "documents_dir": str(self.documents_dir.resolve()),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "retrieval_mode": self.default_mode,
            "bm25_docs": self.store.bm25.size,
        }
