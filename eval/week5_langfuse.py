"""Upload Week 5 frozen traces + human scores to Langfuse (one command).

Creates dataset week5-spa-error-analysis, a dataset run, 20 traces
(question → retrieved chunks → answer), and OK/FAIL scores.

Does not call Groq — it replays eval/week5_traces.json.

  python eval/week5_langfuse.py
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

os.environ.setdefault("LANGFUSE_RELEASE", "week5-error-analysis")
os.environ.setdefault("LANGFUSE_SESSION_ID", "week5-error-analysis")
os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "default")

from rag.observability import flush_langfuse, get_langfuse, tracing_enabled  # noqa: E402

DATASET = "week5-spa-error-analysis"
RUN_NAME = "week5-frozen-traces"
TRACES = ROOT / "eval" / "week5_traces.json"
LABELS = ROOT / "eval" / "week5_error_analysis.csv"


def load_labels() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with LABELS.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = (row.get("id") or "").strip()
            if not re.fullmatch(r"t\d+", tid):
                continue
            rows[tid] = row
    return rows


def ensure_dataset(lf) -> None:
    try:
        lf.get_dataset(DATASET)
        return
    except Exception:
        pass
    lf.create_dataset(
        name=DATASET,
        description=(
            "Week 5 error analysis — 20 questions on Service-Provider-Agreement.pdf only. "
            "Human open-coding and OK/FAIL labels."
        ),
        metadata={"module": "week5", "corpus": "Service-Provider-Agreement.pdf"},
    )


def upsert_items(lf, traces: list[dict], labels: dict[str, dict]) -> None:
    for t in traces:
        tid = t["id"]
        lab = labels.get(tid, {})
        lf.create_dataset_item(
            dataset_name=DATASET,
            id=f"week5-{tid}",
            input={"question": t["question"], "id": tid, "type": t.get("type")},
            expected_output={
                "ok_or_fail": (lab.get("OK_or_FAIL") or "").strip(),
                "expected_from_pdf": lab.get("expected_from_pdf") or "",
            },
            metadata={
                "type": t.get("type") or "",
                "open_code_note": (lab.get("open_code_note") or "")[:500],
            },
        )


def replay(lf, traces: list[dict], labels: dict[str, dict]) -> int:
    from langfuse import propagate_attributes

    dataset = lf.get_dataset(DATASET)
    items = {item.id: item for item in dataset.items}
    uploaded = 0

    for t in traces:
        tid = t["id"]
        lab = labels.get(tid, {})
        verdict = (lab.get("OK_or_FAIL") or "").strip().upper() or "UNLABELED"
        human_ok = 1.0 if verdict == "OK" else 0.0
        item = items.get(f"week5-{tid}")
        ctx = item.run(
            run_name=RUN_NAME,
            run_description="Frozen Week 5 traces (no live LLM calls)",
            run_metadata={"release": os.getenv("LANGFUSE_RELEASE", "week5-error-analysis")},
        ) if item is not None else lf.start_as_current_observation(
            as_type="chain",
            name=f"week5-{tid}",
            input={"question": t["question"], "id": tid},
        )

        with ctx as root:
            with propagate_attributes(
                session_id="week5-error-analysis",
                tags=["week5", "spa-only", str(t.get("type") or ""), verdict],
                metadata={"local_id": tid, "mode": str(t.get("mode") or "hybrid")},
                version=os.getenv("LANGFUSE_RELEASE", "week5-error-analysis"),
            ):
                root.update(input={"question": t["question"], "id": tid, "type": t.get("type")})
                with lf.start_as_current_observation(
                    as_type="retriever",
                    name="retrieve",
                    input={"question": t["question"], "top_k": t.get("top_k"), "mode": t.get("mode")},
                ) as retr:
                    retr.update(output=t.get("chunks") or [])
                with lf.start_as_current_observation(
                    as_type="generation",
                    name="groq-generate",
                    input={"question": t["question"]},
                ) as gen:
                    gen.update(output=t.get("answer"))
                root.update(
                    output={
                        "answer": t.get("answer"),
                        "grounded": t.get("grounded"),
                        "sources": t.get("sources"),
                    }
                )
                comment = (lab.get("open_code_note") or "")[:400]
                root.score_trace(
                    name="human_ok",
                    value=human_ok,
                    data_type="BOOLEAN",
                    comment=comment,
                )
                root.score_trace(
                    name="human_label",
                    value=verdict,
                    data_type="CATEGORICAL",
                    comment=comment,
                )
        uploaded += 1
        print(f"  uploaded {tid}  {verdict}", flush=True)
    return uploaded


def main() -> None:
    if not tracing_enabled():
        raise SystemExit(
            "Langfuse keys missing. Add LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY "
            "to .env (from https://cloud.langfuse.com project settings), then rerun:\n"
            "  python eval/week5_langfuse.py"
        )
    if not TRACES.exists() or not LABELS.exists():
        raise SystemExit("Need eval/week5_traces.json and eval/week5_error_analysis.csv")

    traces = json.loads(TRACES.read_text(encoding="utf-8"))
    labels = load_labels()
    lf = get_langfuse()
    if not lf.auth_check():
        raise SystemExit("Langfuse auth failed. Check public/secret keys and LANGFUSE_HOST.")

    print(f"Dataset {DATASET}  run {RUN_NAME}  release={os.getenv('LANGFUSE_RELEASE')}", flush=True)
    ensure_dataset(lf)
    upsert_items(lf, traces, labels)
    n = replay(lf, traces, labels)
    flush_langfuse()
    host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com"
    print(f"Uploaded {n} traces. Open {host} → Traces (session week5-error-analysis) and Datasets → {DATASET}.", flush=True)


if __name__ == "__main__":
    main()
