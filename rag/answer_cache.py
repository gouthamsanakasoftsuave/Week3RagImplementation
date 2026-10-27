"""Exact answer cache. Same question, mode, and index → no second Groq call.

A near-match cache is the wrong tool for contracts: two similar questions can
be about different clauses. This cache only hits when the key is identical.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from rag.store import RetrievedChunk


def cache_enabled_default() -> bool:
    return os.getenv("ANSWER_CACHE", "true").lower() not in {"0", "false", "no"}


def default_cache_path() -> Path:
    return Path(os.getenv("ANSWER_CACHE_PATH", "logs/answer_cache.json"))


def normalize_question(question: str) -> str:
    return " ".join(question.lower().split())


class AnswerCache:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_cache_path()
        self._data: dict[str, dict] = {}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._data = loaded
            except json.JSONDecodeError:
                self._data = {}

    @staticmethod
    def make_key(
        *,
        question: str,
        mode: str,
        source_filter: str | None,
        top_k: int,
        index_id: str,
    ) -> str:
        return "|".join(
            [
                index_id,
                mode,
                source_filter or "",
                str(top_k),
                normalize_question(question),
            ]
        )

    def get(self, key: str) -> dict | None:
        row = self._data.get(key)
        return row if isinstance(row, dict) else None

    def put(self, key: str, payload: dict) -> None:
        self._data[key] = payload
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")


def chunks_from_payload(rows: list[dict]) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    for row in rows:
        chunks.append(
            RetrievedChunk(
                content=row.get("content") or "",
                source=row.get("source") or "",
                chunk_id=row.get("chunk_id") or "",
                score=float(row.get("score") or 0.0),
                metadata=row.get("metadata") or {},
            )
        )
    return chunks


def chunks_to_payload(chunks: list[RetrievedChunk]) -> list[dict]:
    rows: list[dict] = []
    for chunk in chunks:
        rows.append(
            {
                "content": chunk.content,
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
                "score": chunk.score,
                "metadata": chunk.metadata,
            }
        )
    return rows
