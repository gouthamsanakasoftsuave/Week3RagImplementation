"""Local per-request traces. One JSONL row per Ask, with time and cost on every step.

Langfuse is optional. This file is the log you can still search when a customer
says the answer was wrong and does not have a ticket number.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Iterator


def logging_enabled() -> bool:
    return os.getenv("REQUEST_LOG", "true").lower() not in {"0", "false", "no"}


def log_path() -> Path:
    return Path(os.getenv("REQUEST_LOG_PATH", "logs/requests.jsonl"))


def new_trace_id() -> str:
    return "trc_" + uuid.uuid4().hex[:12]


def elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def make_span(
    name: str,
    duration_ms: int,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost_usd: float = 0.0,
    cache_hit: bool = False,
    model: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "duration_ms": int(duration_ms),
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "cost_usd": round(float(cost_usd), 8),
        "cache_hit": bool(cache_hit),
    }
    if model:
        row["model"] = model
    if note:
        row["note"] = note
    return row


def totals_from_spans(spans: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "duration_ms": int(sum(int(s.get("duration_ms") or 0) for s in spans)),
        "prompt_tokens": int(sum(int(s.get("prompt_tokens") or 0) for s in spans)),
        "completion_tokens": int(sum(int(s.get("completion_tokens") or 0) for s in spans)),
        "cost_usd": round(sum(float(s.get("cost_usd") or 0) for s in spans), 8),
    }


def append_request(record: dict[str, Any], path: Path | None = None) -> None:
    if not logging_enabled() and path is None:
        return
    dest = path or log_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))
