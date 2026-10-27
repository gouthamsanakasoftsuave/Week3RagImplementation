"""Permanent test for the wrong termination clause (trace trc_7f3a91c2e0).

Usage (from repo root):
    python eval/week11_failure_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.termination_guard import is_wrong_termination_citation  # noqa: E402
from rag.week11 import (  # noqa: E402
    DRILL_PATH,
    FAILURE_CASE_PATH,
    GOLDEN_TERMINATION_ANSWER,
    PLANTED_ANSWER,
    PLANTED_TRACE_ID,
    load_records,
)

# A real use of Clause 12.9: where to send a notice, not a 60-day termination right.
NOTICE_ADDRESS_ANSWER = (
    "Notices, including a termination notice, are delivered under Clause 12.9 to the address "
    "listed there. The notice period itself is Clause 7.1 (30 days' prior written notice)."
)


def main() -> None:
    if not FAILURE_CASE_PATH.exists() or not DRILL_PATH.exists():
        raise SystemExit("Run python eval/week11_drill.py first so the production log exists.")

    case = json.loads(FAILURE_CASE_PATH.read_text(encoding="utf-8"))
    if case.get("trace_id") != PLANTED_TRACE_ID:
        raise SystemExit("Failure case is not the logged production trace.")
    if case.get("answer") != PLANTED_ANSWER:
        raise SystemExit("Failure case answer was edited. Keep the logged bad answer as the test.")
    if not is_wrong_termination_citation(case["answer"]):
        raise SystemExit("Guard no longer catches the production failure.")
    if is_wrong_termination_citation(GOLDEN_TERMINATION_ANSWER):
        raise SystemExit("Guard flagged a correct Clause 7.1 answer.")
    if is_wrong_termination_citation(NOTICE_ADDRESS_ANSWER):
        raise SystemExit("Guard flagged Clause 12.9 used only as a delivery address.")

    logs = load_records(DRILL_PATH)
    flagged = [row["trace_id"] for row in logs if is_wrong_termination_citation(row.get("answer") or "")]
    if flagged != [PLANTED_TRACE_ID]:
        raise SystemExit(f"Drill log flags {flagged}, expected only {PLANTED_TRACE_ID}.")

    spans = case.get("spans") or []
    names = [span.get("name") for span in spans]
    if names != ["cache_lookup", "retrieve", "generate"]:
        raise SystemExit(f"Failure case is missing per-step spans: {names}")
    if not any(span.get("name") == "generate" and float(span.get("cost_usd") or 0) > 0 for span in spans):
        raise SystemExit("Generate span has no cost of its own.")

    print(
        f"PASS {PLANTED_TRACE_ID} stays flagged. "
        "Clause 7.1 (30 days) and a Clause 12.9 delivery address do not.",
        flush=True,
    )


if __name__ == "__main__":
    main()
