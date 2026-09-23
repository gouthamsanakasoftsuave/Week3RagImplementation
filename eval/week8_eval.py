"""Week 8 trajectory eval and injection check.

Usage (from repo root, index already built or it will ingest documents/):
    python eval/week8_eval.py --self-check
    python eval/week8_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from rag.pipeline import RagPipeline  # noqa: E402
from rag.week8 import OUT_PATH, run_week8, self_check  # noqa: E402


def ascii(msg: str) -> str:
    return msg.encode("ascii", "replace").decode("ascii")


def main() -> None:
    if "--self-check" in sys.argv:
        result = self_check()
        probes = result["probes"]
        print(ascii("Week 8 self-check OK"))
        print(ascii(f"Loop duplicate calls executed: {probes['loop_duplicate_calls_executed']}"))
        print(ascii(f"Injection canary exposed: {probes['injection_canary_exposed']}"))
        example = result["worked_example"]
        print(ascii(f"Gap example before failures: {example['before']['trajectory_failures']}"))
        print(ascii(f"Gap example after failures: {example['after']['trajectory_failures']}"))
        return

    pipe = RagPipeline(documents_dir=ROOT / "documents", persist_dir=ROOT / "chroma_db")
    result = run_week8(pipe)
    print(ascii("Week 8 — Track F Legal contracts"))
    print(ascii(f"Top failure: {result['top_failure']}"))
    print(
        ascii(
            f"Rate {result['top_failure_rate_before']} -> {result['top_failure_rate_after']} "
            f"({result['top_failure_before']} -> {result['top_failure_after']})"
        )
    )
    print(ascii(f"Outcome {result['before']['outcome_rate']} -> {result['after']['outcome_rate']}"))
    print(
        ascii(
            f"Trajectory {result['before']['trajectory_rate']} -> {result['after']['trajectory_rate']}"
        )
    )
    print(ascii(f"Gaps {result['before']['gap_ids']}"))
    for kind, pair in result["injection"].items():
        print(
            ascii(
                f"Injection {kind}: tricked {pair['before']['tricked']} -> {pair['after']['tricked']}"
            )
        )
    print(ascii(f"Wrote {OUT_PATH}"))


if __name__ == "__main__":
    main()
