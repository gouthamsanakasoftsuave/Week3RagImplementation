"""End-to-end RAG pipeline: load → chunk → embed → retrieve → generate."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from rag.agent_types import estimate_cost_usd
from rag.answer_cache import (
    AnswerCache,
    cache_enabled_default,
    chunks_from_payload,
    chunks_to_payload,
)
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
from rag.request_log import append_request, elapsed_ms, make_span, new_trace_id, totals_from_spans
from rag.store import RetrievedChunk, VectorStore
from rag.termination_guard import is_wrong_termination_citation

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
        use_cache: bool | None = None,
        cache_path: str | Path | None = None,
        channel: str = "app",
    ) -> RagAnswer:
        retrieval_mode = (mode or self.default_mode).lower()
        k = top_k or self.top_k
        cache_on = cache_enabled_default() if use_cache is None else use_cache
        model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        trace_id = new_trace_id()
        spans: list[dict] = []
        cache = AnswerCache(Path(cache_path) if cache_path else None)
        cache_key = AnswerCache.make_key(
            question=question,
            mode=retrieval_mode,
            source_filter=source_filter,
            top_k=k,
            index_id=str(self.store.persist_dir),
        )

        t_cache = time.perf_counter()
        cached = cache.get(cache_key) if cache_on else None
        spans.append(
            make_span(
                "cache_lookup",
                elapsed_ms(t_cache),
                cache_hit=cached is not None,
                note="hit" if cached is not None else "miss",
            )
        )

        avoided = 0.0
        if cached is not None:
            avoided = float(cached.get("cost_usd") or 0.0)
            spans.append(
                make_span(
                    "retrieve",
                    0,
                    cache_hit=True,
                    note="skipped; exact cache hit",
                )
            )
            spans.append(
                make_span(
                    "generate",
                    0,
                    cache_hit=True,
                    model=model_name,
                    note="skipped; exact cache hit",
                )
            )
            answer = RagAnswer(
                answer=cached.get("answer") or "",
                sources=list(cached.get("sources") or []),
                chunks_used=chunks_from_payload(cached.get("chunks") or []),
                grounded=bool(cached.get("grounded")),
                prompt_tokens=0,
                completion_tokens=0,
                llm_calls=0,
            )
        else:
            payload_in = {
                "question": question,
                "mode": retrieval_mode,
                "top_k": k,
                "source_filter": source_filter,
            }

            def _run() -> RagAnswer:
                t_retrieve = time.perf_counter()
                chunks = self.retrieve(
                    question,
                    top_k=top_k,
                    source_filter=source_filter,
                    mode=mode,
                    rewrite=rewrite,
                )
                spans.append(
                    make_span(
                        "retrieve",
                        elapsed_ms(t_retrieve),
                        note="local hybrid search; embedding cost is 0",
                    )
                )
                t_generate = time.perf_counter()
                generated = generate_answer(question, chunks)
                spans.append(
                    make_span(
                        "generate",
                        elapsed_ms(t_generate),
                        prompt_tokens=generated.prompt_tokens,
                        completion_tokens=generated.completion_tokens,
                        cost_usd=estimate_cost_usd(
                            generated.prompt_tokens,
                            generated.completion_tokens,
                        ),
                        model=model_name,
                    )
                )
                return generated

            if not tracing_enabled():
                answer = _run()
            else:
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
                        metadata={"mode": retrieval_mode, "trace_id": trace_id},
                        version=os.getenv("LANGFUSE_RELEASE", "week5-error-analysis"),
                    ):
                        answer = _run()
                        root.update(
                            output={
                                "answer": answer.answer,
                                "grounded": answer.grounded,
                                "sources": answer.sources,
                                "trace_id": trace_id,
                            }
                        )

            if cache_on:
                paid = estimate_cost_usd(answer.prompt_tokens, answer.completion_tokens)
                cache.put(
                    cache_key,
                    {
                        "answer": answer.answer,
                        "sources": answer.sources,
                        "grounded": answer.grounded,
                        "chunks": chunks_to_payload(answer.chunks_used),
                        "cost_usd": paid,
                    },
                )

        totals = totals_from_spans(spans)
        quality_flag = (
            "wrong_termination_clause" if is_wrong_termination_citation(answer.answer) else ""
        )
        answer.trace_id = trace_id
        answer.cache_hit = cached is not None
        answer.cost_usd = float(totals["cost_usd"])
        answer.avoided_cost_usd = avoided
        answer.duration_ms = int(totals["duration_ms"])
        answer.quality_flag = quality_flag
        answer.spans = spans
        append_request(
            {
                "trace_id": trace_id,
                "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
                "channel": channel,
                "question": question,
                "answer": answer.answer,
                "sources": answer.sources,
                "model": model_name,
                "cache_hit": answer.cache_hit,
                "quality_flag": quality_flag,
                "spans": spans,
                "avoided_cost_usd": round(avoided, 8),
                "chunks": chunks_payload(answer.chunks_used),
                **totals,
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
