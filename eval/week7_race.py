"""Week 7 race: contract agent vs fixed workflow on speed, cost, reliability.

Usage (from repo root, with index already built or it will ingest documents/):
    python eval/week7_race.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from rag.agent_race import run_race  # noqa: E402
from rag.pipeline import RagPipeline  # noqa: E402

OUT_PATH = ROOT / "eval" / "week7_race.json"


def ascii(msg: str) -> str:
    return msg.encode("ascii", "replace").decode("ascii")


def main() -> None:
    pipe = RagPipeline(documents_dir=ROOT / "documents", persist_dir=ROOT / "chroma_db")
    result = run_race(
        pipe,
        tasks_path=ROOT / "eval" / "week7_tasks.json",
        out_path=OUT_PATH,
        memory_path=ROOT / "eval" / "agent_memory.json",
    )
    s = result["summary"]
    print(ascii("Week 7 race — Track F Legal contracts"))
    print(ascii(f"Workflow  {s['workflow']}"))
    print(ascii(f"Agent     {s['agent']}"))
    print(ascii(f"Ship: {s['ship']}"))
    print(ascii(f"Wrote {OUT_PATH}"))


if __name__ == "__main__":
    main()
