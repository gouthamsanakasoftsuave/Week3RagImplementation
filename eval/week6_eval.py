"""Week 6 evals: one-command test set, validated clause judge, before/after scores.

Usage (from repo root):
    python eval/week6_eval.py

Uses Service-Provider-Agreement.pdf only (same corpus as Week 5).
Before scores come from eval/week5_traces.json (the real failures).
After scores re-run the app with preamble-preserving chunks (the Week 6 fix).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from groq import Groq  # noqa: E402
from rag.pipeline import RagPipeline  # noqa: E402

PDF_NAME = "Service-Provider-Agreement.pdf"
SRC_PDF = ROOT / "documents" / PDF_NAME
DOCS_DIR = ROOT / "eval" / "spa_only_docs"
DB_DIR = ROOT / "eval" / "chroma_week6"
EVALSET = ROOT / "eval" / "week6_evalset.json"
TRACES_BEFORE = ROOT / "eval" / "week5_traces.json"
OUT_PATH = ROOT / "eval" / "week6_results.json"

IDK = "i don't know based on the provided documents"
JUDGE_PASS_AT = 7


def ascii(msg: str) -> str:
    return msg.encode("ascii", "replace").decode("ascii")


def norm(text: str) -> str:
    t = (text or "").lower()
    t = t.replace("\u20b9", "rs").replace(",", "")
    t = re.sub(r"\s+", " ", t)
    return t


def has_phrase(haystack: str, needle: str) -> bool:
    return norm(needle) in norm(haystack)


def is_refuse(answer: str) -> bool:
    return IDK in (answer or "").lower()


def assertion_score(item: dict, answer: str, context: str) -> dict:
    """Rule checks first (free): refuse when required; required phrases present."""
    failures: list[str] = []
    if item["should_refuse"]:
        if not is_refuse(answer):
            failures.append("expected refuse but answer did not say I don't know")
    else:
        if is_refuse(answer):
            failures.append("refused but the clause is in the contract")
        for phrase in item.get("must_contain_all") or []:
            if not has_phrase(answer, phrase):
                failures.append(f"missing required phrase: {phrase}")
        any_need = item.get("must_contain_any") or []
        if any_need and not any(has_phrase(answer, p) for p in any_need):
            failures.append(f"missing any of: {any_need}")

    required = list(item.get("must_contain_all") or []) + list(
        item.get("must_contain_any") or []
    )
    if item["should_refuse"]:
        context_recall = 1.0 if is_refuse(answer) else 0.0
        context_precision = 1.0
        faithfulness = 1.0 if is_refuse(answer) else 0.0
        relevancy = 1.0 if is_refuse(answer) else 0.0
    else:
        hits = [p for p in required if has_phrase(context, p)] if required else []
        context_recall = (len(hits) / len(required)) if required else 1.0
        chunks = [c.strip() for c in context.split("\n\n---\n\n") if c.strip()] or [context]
        if chunks and required:
            useful = sum(1 for c in chunks if any(has_phrase(c, p) for p in required))
            context_precision = useful / len(chunks)
        else:
            context_precision = 1.0
        answer_facts = [p for p in required if has_phrase(answer, p)]
        invented = [p for p in answer_facts if not has_phrase(context, p)]
        faithfulness = 0.0 if invented else 1.0
        relevancy = 0.0 if failures else 1.0

    return {
        "assertion_pass": not failures,
        "assertion_failures": failures,
        "context_recall": round(context_recall, 3),
        "context_precision": round(context_precision, 3),
        "faithfulness": round(faithfulness, 3),
        "answer_relevancy": round(relevancy, 3),
    }


def judge_clause_answer(item: dict, answer: str, context: str) -> dict:
    """LLM-as-judge (G-Eval style 1-10 + binary pass). Track F: clause-answer judge."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    expected = (
        "The answer MUST refuse with I don't know (fact is not in the contract)."
        if item["should_refuse"]
        else (
            "Must include: "
            + ", ".join(item.get("must_contain_all") or [])
            + ("; any of: " + ", ".join(item.get("must_contain_any") or []) if item.get("must_contain_any") else "")
        )
    )
    prompt = f"""You grade a legal-contract RAG answer. Use only the question, expected facts, and the model answer.
Do not reward extra contracts or invented parties/dates/amounts.

Question: {item['question']}
Expected: {expected}
Should refuse: {item['should_refuse']}
Model answer:
{answer[:2500]}

Score 1-10:
10 = fully correct and grounded
7 = usable, small wording issues
4 = partial / missing a key name or number
1 = wrong, mixed contracts, or failed to refuse

Return JSON only: {{"score": <int>, "pass": <true if score>={JUDGE_PASS_AT}>, "reason": "<one sentence>"}}"""
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=200,
        messages=[
            {
                "role": "system",
                "content": "You are a strict legal-clause answer judge. Return JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()
    return parse_judge_payload(raw)


def parse_judge_payload(raw: str, assertion_pass: bool | None = None) -> dict:
    """Parse G-Eval JSON; tolerate truncated output from the model."""
    text = (raw or "").strip()
    score_m = re.search(r'"score"\s*:\s*(\d+)', text)
    pass_m = re.search(r'"pass"\s*:\s*(true|false)', text, re.I)
    reason_m = re.search(r'"reason"\s*:\s*"([^"]*)', text)
    if score_m:
        score = int(score_m.group(1))
        passed = (
            pass_m.group(1).lower() == "true" if pass_m else score >= JUDGE_PASS_AT
        )
        return {
            "judge_score": score,
            "judge_pass": passed,
            "judge_reason": (reason_m.group(1) if reason_m else text)[:240],
        }
    if assertion_pass is not None:
        return {
            "judge_score": 7 if assertion_pass else 3,
            "judge_pass": assertion_pass,
            "judge_reason": "judge returned no JSON; fell back to assertion check",
            "raw": text[:200],
        }
    return {
        "judge_score": 0,
        "judge_pass": False,
        "judge_reason": text[:200],
        "raw": text[:200],
    }


def load_evalset() -> list[dict]:
    return json.loads(EVALSET.read_text(encoding="utf-8"))


def traces_by_id() -> dict[str, dict]:
    rows = json.loads(TRACES_BEFORE.read_text(encoding="utf-8"))
    return {row["id"]: row for row in rows}


def context_from_trace(trace: dict) -> str:
    parts = []
    for c in trace.get("chunks") or []:
        parts.append(f"p{c.get('page')} {c.get('snippet') or ''}")
    return "\n\n---\n\n".join(parts)


def score_answers(
    items: list[dict],
    answers: dict[str, tuple[str, str]],
    *,
    use_judge: bool,
) -> list[dict]:
    rows = []
    for item in items:
        answer, context = answers[item["id"]]
        row = {
            "id": item["id"],
            "problem_type": item["problem_type"],
            "human_ok": item["human_ok"],
            "answer": answer,
        }
        row.update(assertion_score(item, answer, context))
        if use_judge:
            print(ascii(f"  judge {item['id']}..."), flush=True)
            judged = judge_clause_answer(item, answer, context)
            if judged.get("judge_score") == 0:
                judged = parse_judge_payload(
                    judged.get("raw") or judged.get("judge_reason") or "",
                    assertion_pass=row["assertion_pass"],
                )
            row.update(judged)
        row["pass"] = row["assertion_pass"]
        rows.append(row)
    return rows


def by_problem(rows: list[dict]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["problem_type"]].append(row)
    out = {}
    for name, grp in groups.items():
        n = len(grp)
        out[name] = {
            "n": n,
            "assertion_pass_rate": round(sum(r["assertion_pass"] for r in grp) / n, 3),
            "faithfulness": round(sum(r["faithfulness"] for r in grp) / n, 3),
            "context_recall": round(sum(r["context_recall"] for r in grp) / n, 3),
            "judge_pass_rate": round(
                sum(r.get("judge_pass", False) for r in grp) / n, 3
            )
            if any("judge_pass" in r for r in grp)
            else None,
        }
    return out


def overall_pass(rows: list[dict]) -> float:
    return round(sum(r["assertion_pass"] for r in rows) / (len(rows) or 1), 3)


def validate_judge(items: list[dict], before_rows: list[dict]) -> dict:
    """Check the AI judge against Week 5 human OK/FAIL labels (frozen answers)."""
    judged = [r for r in before_rows if "judge_pass" in r]
    if not judged:
        return {"agreement": None, "n": 0}
    agree = 0
    disagreements = []
    for row in judged:
        human = bool(row["human_ok"])
        judge = bool(row["judge_pass"])
        if human == judge:
            agree += 1
        else:
            disagreements.append(
                {
                    "id": row["id"],
                    "human_ok": human,
                    "judge_pass": judge,
                    "reason": row.get("judge_reason"),
                }
            )
    return {
        "n": len(judged),
        "agreement": round(agree / len(judged), 3),
        "disagreements": disagreements,
        "note": "Agreement vs Week 5 human labels on frozen traces. Trust the judge only if this is high.",
    }


def run_after(items: list[dict]) -> dict[str, tuple[str, str]]:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    dest = DOCS_DIR / PDF_NAME
    shutil.copy2(SRC_PDF, dest)
    for extra in DOCS_DIR.iterdir():
        if extra.is_file() and extra.name != PDF_NAME:
            extra.unlink()

    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)

    pipeline = RagPipeline(documents_dir=DOCS_DIR, persist_dir=DB_DIR)
    print("Indexing PDF with preamble chunks kept together...", flush=True)
    result = pipeline.ingest(rebuild=True, keep_preamble=True)
    print(ascii(f"Indexed {result.chunks} chunks from {result.sources}"), flush=True)

    answers: dict[str, tuple[str, str]] = {}
    for i, item in enumerate(items, start=1):
        print(ascii(f"[{i}/{len(items)}] {item['id']}: {item['question'][:70]}"), flush=True)
        rag = pipeline.ask(item["question"], top_k=4, mode="hybrid", rewrite=False)
        context = "\n\n---\n\n".join(c.content or "" for c in rag.chunks_used)
        answers[item["id"]] = (rag.answer, context)
        print(ascii(f"  assertion preview: {(rag.answer or '')[:160]}"), flush=True)
    return answers


def print_table(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    print(f"{'id':<6} {'type':<18} {'assert':<8} {'human':<8} {'judge':<8}")
    for row in rows:
        jp = row.get("judge_pass")
        judge = "-" if jp is None else ("PASS" if jp else "FAIL")
        print(
            f"{row['id']:<6} {row['problem_type']:<18} "
            f"{'PASS' if row['assertion_pass'] else 'FAIL':<8} "
            f"{'OK' if row['human_ok'] else 'FAIL':<8} {judge:<8}"
        )


def main() -> None:
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("Missing GROQ_API_KEY in .env")
    items = load_evalset()
    traces = traces_by_id()
    missing = [it["id"] for it in items if it["id"] not in traces]
    if missing:
        raise SystemExit(f"week5_traces.json missing ids: {missing}")

    before_answers = {
        it["id"]: (traces[it["id"]]["answer"], context_from_trace(traces[it["id"]]))
        for it in items
    }

    print("=== 1. Score Week 5 traces (BEFORE) + validate clause judge ===", flush=True)
    before_rows = score_answers(items, before_answers, use_judge=True)
    print_table("BEFORE (Week 5 frozen answers)", before_rows)
    judge_val = validate_judge(items, before_rows)
    print(
        ascii(
            f"\nJudge vs human agreement: {judge_val['agreement']} "
            f"({judge_val['n']} traces)"
        ),
        flush=True,
    )

    print("\n=== 2. Re-run after preamble-chunk fix (AFTER) ===", flush=True)
    after_answers = run_after(items)
    after_rows = score_answers(items, after_answers, use_judge=True)
    print_table("AFTER (keep party preamble as one chunk)", after_rows)

    before_by = by_problem(before_rows)
    after_by = by_problem(after_rows)
    types = sorted(set(before_by) | set(after_by))
    deltas = {}
    print("\n=== 3. Before / after by problem type (assertion pass rate) ===")
    print(f"{'problem_type':<22} {'before':<10} {'after':<10} {'delta':<10}")
    for name in types:
        b = before_by[name]["assertion_pass_rate"]
        a = after_by[name]["assertion_pass_rate"]
        d = round(a - b, 3)
        deltas[name] = {"before": b, "after": a, "delta": d, "n": after_by[name]["n"]}
        print(f"{name:<22} {b:<10} {a:<10} {d:<+10}")

    overall = {
        "before": overall_pass(before_rows),
        "after": overall_pass(after_rows),
    }
    overall["delta"] = round(overall["after"] - overall["before"], 3)
    print(
        f"\nOVERALL assertion pass  before={overall['before']}  "
        f"after={overall['after']}  delta={overall['delta']:+}"
    )
    t01_b = next(r["assertion_pass"] for r in before_rows if r["id"] == "t01")
    t01_a = next(r["assertion_pass"] for r in after_rows if r["id"] == "t01")
    print(f"t01 preamble_split (the Week 5 fail): before={t01_b} after={t01_a}")

    payload = {
        "command": "python eval/week6_eval.py",
        "corpus": PDF_NAME,
        "change": "Keep FIRST PART / SECOND PART preamble as one chunk so party names stay together.",
        "prediction": "t01 should name EIH Limited and Pressman Advertising Limited; other types should not drop.",
        "judge_validation": judge_val,
        "overall": overall,
        "by_problem_type": deltas,
        "before": before_rows,
        "after": after_rows,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(ascii(f"\nWrote {OUT_PATH}"))


if __name__ == "__main__":
    main()
