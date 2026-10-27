"""Week 10 race: contract squad vs the single agent on the same tests.

Usage (from repo root):
    python eval/week10_race.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from rag.pipeline import RagPipeline  # noqa: E402
from rag.squad_race import run_squad_race  # noqa: E402


def ascii(msg: str) -> str:
    return msg.encode("ascii", "replace").decode("ascii")


def main() -> None:
    pipe = RagPipeline(documents_dir=ROOT / "documents", persist_dir=ROOT / "chroma_db")
    result = run_squad_race(
        pipe,
        tasks_path=ROOT / "eval" / "week7_tasks.json",
        out_path=ROOT / "eval" / "week10_race.json",
        report_path=ROOT / "WEEK_10_REPORT.md",
    )
    summary = result["summary"]
    verdict = result["verdict"]
    print(ascii("Week 10 race — Track F Legal contracts"))
    print(ascii(f"Single  {summary['single']}"))
    print(ascii(f"Squad   {summary['team']}"))
    print(ascii(f"Re-sent prompt tokens {summary['context_resend_prompt_tokens']}"))
    print(ascii(f"Keep the {verdict['keep']}. {verdict['reason']}"))
    print(ascii(f"Wrote {ROOT / 'eval' / 'week10_race.json'}"))
    print(ascii(f"Wrote {ROOT / 'WEEK_10_REPORT.md'}"))


if __name__ == "__main__":
    main()
