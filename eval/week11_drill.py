"""Find the planted wrong termination clause in the Week 11 support log.

Usage (from repo root):
    python eval/week11_drill.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.week11 import (  # noqa: E402
    DRILL_PATH,
    DRILL_RESULT_PATH,
    FAILURE_CASE_PATH,
    PLANTED_TRACE_ID,
    build_drill_records,
    drill_report,
    write_jsonl,
)


def main() -> None:
    records = build_drill_records()
    write_jsonl(DRILL_PATH, records)
    report = drill_report(records)
    if report["found_trace_id"] != PLANTED_TRACE_ID:
        raise SystemExit(f"Found {report['found_trace_id']}, expected {PLANTED_TRACE_ID}")
    found = next(row for row in records if row["trace_id"] == PLANTED_TRACE_ID)
    DRILL_RESULT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    FAILURE_CASE_PATH.write_text(json.dumps(found, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Searched {report['log_count']} logs. "
        f"Day {report['day']} shortlist {report['shortlist_count']}. "
        f"Wrong termination clause: {report['found_trace_id']}",
        flush=True,
    )
    print(f"Wrote {DRILL_PATH}, {DRILL_RESULT_PATH}, {FAILURE_CASE_PATH}", flush=True)


if __name__ == "__main__":
    main()
