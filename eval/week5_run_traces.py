"""Run 20 Week-5 questions against Service-Provider-Agreement.pdf only."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.observability import flush_langfuse, tracing_enabled  # noqa: E402
from rag.pipeline import RagPipeline  # noqa: E402

PDF_NAME = "Service-Provider-Agreement.pdf"
SRC_PDF = ROOT / "documents" / PDF_NAME
DOCS_DIR = ROOT / "eval" / "spa_only_docs"
DB_DIR = ROOT / "eval" / "chroma_spa_only"

QUESTIONS = [
    ("t01", "easy", "Who are the two parties to the Service Provider Agreement, and what is each party called in the contract?"),
    ("t02", "easy", "On what date and in which city was the Service Provider Agreement entered into?"),
    ("t03", "easy", "What is the face value of the Rights Equity Shares and up to what amount is the Issue aggregating?"),
    ("t04", "easy", "Who is the Lead Manager appointed for the Issue?"),
    ("t05", "easy", "What professional fee will the Agency be paid for media monitoring under the commercial terms?"),
    ("t06", "paraphrase", "How much written notice does a party need to give to terminate this agreement?"),
    ("t07", "paraphrase", "Which country's law governs this agreement?"),
    ("t08", "paraphrase", "Which courts have exclusive jurisdiction over disputes under this agreement?"),
    ("t09", "detail", "Within how many days must advertising bills be settled after the month in which ads were released?"),
    ("t10", "detail", "If the Company asks the Agency to return Confidential Information, how soon must the Agency return it?"),
    ("t11", "detail", "Can the Company terminate the agreement without notice if it thinks the Agency's services are deficient?"),
    ("t12", "detail", "What is the maximum aggregate liability of the Agency under the indemnity clause?"),
    ("t13", "detail", "How long does the confidentiality clause survive after expiry or early termination of the agreement?"),
    ("t14", "multi", "If the parties have a dispute, what must they try first and when can they go to arbitration?"),
    ("t15", "detail", "Who owns the creatives, advertisements, reports and other materials produced from the Agency's services?"),
    ("t16", "detail", "Who is the contact person and email for notices to the Company?"),
    ("t17", "out of scope", "What is the GDPR fine in the Data Processing Agreement attached to this contract?"),
    ("t18", "out of scope", "What is Jordan Hale's base salary under this Service Provider Agreement?"),
    ("t19", "out of scope", "Does this agreement require the Agency to maintain cyber insurance of $2 million?"),
    ("t20", "vague", "Who must indemnify whom if there is a breach of a third party's intellectual property?"),
]


def snippet(text: str, n: int = 180) -> str:
    t = (text or "").replace("\n", " ").strip()
    if not t:
        return "(empty)"
    return t if len(t) <= n else t[: n - 1] + "…"


def ascii(msg: str) -> str:
    return msg.encode("ascii", "replace").decode("ascii")


def main() -> None:
    os.environ.setdefault("LANGFUSE_SESSION_ID", "week5-live")
    os.environ.setdefault("LANGFUSE_RELEASE", "week5-error-analysis")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    dest = DOCS_DIR / PDF_NAME
    shutil.copy2(SRC_PDF, dest)
    for extra in DOCS_DIR.iterdir():
        if extra.is_file() and extra.name != PDF_NAME:
            extra.unlink()

    pipeline = RagPipeline(documents_dir=DOCS_DIR, persist_dir=DB_DIR)
    print("Rebuilding index from PDF only...", flush=True)
    result = pipeline.ingest(rebuild=True)
    print(ascii(f"Indexed {result.chunks} chunks from {result.sources}"), flush=True)

    traces = []
    for i, (tid, qtype, question) in enumerate(QUESTIONS, start=1):
        print(ascii(f"\n[{i}/20] {tid}: {question}"), flush=True)
        answer = pipeline.ask(question, top_k=4, source_filter=None, mode="hybrid")
        chunks = answer.chunks_used
        chunk_rows = []
        for rank, c in enumerate(chunks, start=1):
            page = c.metadata.get("page") or "?"
            chunk_rows.append(
                {
                    "rank": rank,
                    "source": c.source,
                    "page": page,
                    "score": round(float(c.score), 4),
                    "empty": not (c.content or "").strip(),
                    "snippet": snippet(c.content or ""),
                }
            )
            print(
                ascii(
                    f"  #{rank} {c.source} p{page} empty={not (c.content or '').strip()} "
                    f"{snippet(c.content or '', 90)}"
                ),
                flush=True,
            )
        print(ascii(f"  grounded={answer.grounded}"), flush=True)
        print(ascii(f"  answer={(answer.answer or '')[:500]}"), flush=True)
        traces.append(
            {
                "id": tid,
                "type": qtype,
                "question": question,
                "mode": "hybrid",
                "top_k": 4,
                "chunks": chunk_rows,
                "answer": answer.answer,
                "grounded": answer.grounded,
                "sources": answer.sources,
            }
        )

    out = ROOT / "eval" / "week5_traces.json"
    out.write_text(json.dumps(traces, indent=2, ensure_ascii=False), encoding="utf-8")
    print(ascii(f"\nWrote {out}"), flush=True)
    if tracing_enabled():
        flush_langfuse()
        print("Flushed live traces to Langfuse (session week5-live).", flush=True)
    else:
        print("Langfuse keys not set — local JSON only. After adding keys: python eval/week5_langfuse.py", flush=True)


if __name__ == "__main__":
    main()
