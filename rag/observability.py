"""Optional Langfuse tracing. No-ops unless public + secret keys are set."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_CLIENT = None


def tracing_enabled() -> bool:
    if os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() in {"0", "false", "no"}:
        return False
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def get_langfuse():
    """Return a shared Langfuse client, or None if keys are missing."""
    global _CLIENT
    if not tracing_enabled():
        return None
    if _CLIENT is None:
        from langfuse import Langfuse

        _CLIENT = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL"),
            release=os.getenv("LANGFUSE_RELEASE", "week5-error-analysis"),
            environment=os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "default"),
        )
    return _CLIENT


def session_id(default: str = "ask-legal-contracts") -> str:                                                                                                                             
    return os.getenv("LANGFUSE_SESSION_ID", default)                               
                        
def trace_tags(*extra: str) -> list[str]:
    tags = ["rag", "legal-contracts", "track-f"]
    tags.extend(t for t in extra if t)
    return tags


def chunks_payload(chunks: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks, start=1):
        meta = getattr(chunk, "metadata", None) or {}
        content = getattr(chunk, "content", "") or ""
        rows.append(
            {
                "rank": i,
                "source": getattr(chunk, "source", ""),
                "page": meta.get("page") if isinstance(meta, dict) else None,
                "score": round(float(getattr(chunk, "score", 0.0) or 0.0), 4),
                "chunk_id": getattr(chunk, "chunk_id", ""),
                "snippet": content[:400],
            }
        )
    return rows


def flush_langfuse() -> None:
    client = get_langfuse()
    if client is not None:
        client.flush()
