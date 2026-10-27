"""Measure cost per Ask, then the same questions again with the exact cache.

Usage (from repo root):
    python eval/week11_cost.py

Four live Groq calls (cache misses), then the same four served from cache.
Writes eval/week11_cost.json and WEEK_11_REPORT.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from rag.pipeline import RagPipeline  # noqa: E402
from rag.week11 import (  # noqa: E402
    COST_PATH,
    DRILL_PATH,
    ensure_drill,
    drill_report,
    write_report_file,
)

QUESTIONS = [
    "How much written notice does a party need to give to terminate this agreement?",
    "Which country's law governs this agreement?",
    "Who are the two parties to the Service Provider Agreement?",
    "Who is the Lead Manager appointed for the Issue?",
]
CACHE_PATH = ROOT / "eval" / "week11_cost_cache.json"


def main() -> None:
    if CACHE_PATH.exists():
        CACHE_PATH.unlink()

    pipeline = RagPipeline(
        documents_dir=ROOT / "documents",
        persist_dir=ROOT / "eval" / "chroma_week11",
    )
    if pipeline.store.count == 0:
        print("Building a local index for the cost run...", flush=True)
        pipeline.ingest(rebuild=True)

    misses = []
    for question in QUESTIONS:
        answer = pipeline.ask(
            question,
            top_k=4,
            mode="hybrid",
            rewrite=False,
            use_cache=True,
            cache_path=CACHE_PATH,
            channel="cost-benchmark",
        )
        misses.append((question, answer))
        print(
            f"miss {answer.trace_id} ${answer.cost_usd:.6f} {answer.duration_ms} ms",
            flush=True,
        )

    hits = []
    for question in QUESTIONS:
        answer = pipeline.ask(
            question,
            top_k=4,
            mode="hybrid",
            rewrite=False,
            use_cache=True,
            cache_path=CACHE_PATH,
            channel="cost-benchmark",
        )
        if not answer.cache_hit:
            raise SystemExit(f"Expected a cache hit for: {question}")
        if answer.cost_usd != 0:
            raise SystemExit(f"Cache hit still charged ${answer.cost_usd}")
        hits.append((question, answer))
        print(
            f"hit  {answer.trace_id} ${answer.cost_usd:.6f} {answer.duration_ms} ms",
            flush=True,
        )

    baseline_costs = [answer.cost_usd for _, answer in misses]
    # Without a cache, the repeat would pay the miss price again.
    baseline_mean = sum(baseline_costs) / len(baseline_costs)
    improved_costs = baseline_costs + [answer.cost_usd for _, answer in hits]
    improved_mean = sum(improved_costs) / len(improved_costs)
    reduction = 0.0 if baseline_mean == 0 else (1 - improved_mean / baseline_mean) * 100

    def pack(pairs):
        rows = []
        for question, answer in pairs:
            rows.append(
                {
                    "question": question,
                    "trace_id": answer.trace_id,
                    "cache_hit": answer.cache_hit,
                    "cost_usd": answer.cost_usd,
                    "duration_ms": answer.duration_ms,
                    "spans": answer.spans,
                    "quality_flag": answer.quality_flag,
                }
            )
        return rows

    import os

    result = {
        "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        "workload": "Each of 4 contract questions once, then the same 4 again.",
        "baseline_requests": len(QUESTIONS) * 2,
        "baseline_cost_per_request_usd": round(baseline_mean, 8),
        "improved_requests": len(improved_costs),
        "improved_cost_per_request_usd": round(improved_mean, 8),
        "cache_hit_cost_per_request_usd": 0.0,
        "reduction_pct": round(reduction, 1),
        "misses": pack(misses),
        "hits": pack(hits),
        "note": (
            "Baseline cost per request is the mean of the four measured Groq calls. "
            "A repeat of those calls with no cache would cost the same again. "
            "The improved figure is the mean across the four misses and four cache hits."
        ),
    }
    COST_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    drill = drill_report(ensure_drill(DRILL_PATH))
    write_report_file(drill, result)
    print(
        f"Cost per request ${baseline_mean:.8f} -> ${improved_mean:.8f} ({reduction:.1f}% lower)",
        flush=True,
    )
    print(f"Wrote {COST_PATH} and WEEK_11_REPORT.md", flush=True)


if __name__ == "__main__":
    main()
