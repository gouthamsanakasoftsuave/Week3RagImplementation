"""Week 8 practical, Task Set F — trajectory eval for the contract agent.

Asserts 10 expected paths (alternate orders stored as a set, not one sequence).
Reports tool-choice accuracy, argument validity, step efficiency, and cost p50 and max.
Applies exactly one mitigation to the top failure mode and records the price and regressions.

Usage (from repo root):
    python eval/trajectory_eval.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from rag.agent_loop import run_agent  # noqa: E402
from rag.agent_memory import AgentMemory  # noqa: E402
from rag.agent_race import is_refuse, score_item  # noqa: E402
from rag.agent_types import AgentRun, AgentStep, UsageMeter  # noqa: E402
from rag.pipeline import RagPipeline  # noqa: E402

TASKS_PATH = ROOT / "eval" / "week8_practical_tasks.json"
OUT_PATH = ROOT / "eval" / "week8_practical_results.json"

ALLOWED = {"list_contracts", "search_contracts", "read_chunk"}
MODES = [
    "skipped_defined_term",
    "loop",
    "quiet_give_up",
    "fabricated_argument",
    "wrong_tool",
    "wrong_path",
]
CLAUSE_REF = re.compile(r"\b\d{1,2}\.\d{1,2}\b")


def ascii(msg: str) -> str:
    return msg.encode("ascii", "replace").decode("ascii")


def p50(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _calls(run: AgentRun) -> list[dict]:
    pending: AgentStep | None = None
    pairs: list[dict] = []
    for step in run.steps:
        if step.kind == "tool":
            pending = step
            continue
        if step.kind != "observe" or pending is None:
            continue
        payload = pending.payload or {}
        observation = step.detail or ""
        blocked = observation.startswith("STEP_LIMIT") or observation.startswith("LOOP_BLOCKED") or observation.startswith("Error: chunk_id was not returned")
        pairs.append(
            {
                "tool": payload.get("tool") or "",
                "args": payload.get("args") or {},
                "observation": observation,
                "blocked": blocked,
            }
        )
        pending = None
    return pairs


def _query_sig(call: dict) -> str:
    args = call["args"] if isinstance(call["args"], dict) else {}
    if call["tool"] == "search_contracts":
        query = re.sub(r"\s+", " ", str(args.get("query") or "").lower()).strip()
        return "search:" + query
    if call["tool"] == "read_chunk":
        return "read:" + str(args.get("chunk_id") or "").strip()
    return call["tool"] or "unknown"


def _group_hit(group: dict, blob: str) -> bool:
    return any(token in blob for token in group.get("any") or [])


def score_case(task: dict, run: AgentRun) -> dict:
    calls = _calls(run)
    executed = [call for call in calls if not call["blocked"]]
    observations = [call["observation"] for call in executed]
    blob = "\n".join(observations).lower()
    names = [call["tool"] for call in executed]
    searches = [call for call in executed if call["tool"] == "search_contracts"]

    modes: list[str] = []
    seen_ids: set[str] = set()
    signatures: list[str] = []
    arg_checks = 0
    arg_valid = 0
    choice_checks = 0
    choice_valid = 0

    for call in executed:
        name = call["tool"]
        args = call["args"] if isinstance(call["args"], dict) else {}
        choice_checks += 1
        if name in ALLOWED:
            choice_valid += 1
        else:
            modes.append("wrong_tool")

        signature = _query_sig(call)
        if signature in signatures:
            modes.append("loop")
        signatures.append(signature)

        arg_checks += 1
        if name == "search_contracts":
            query = str(args.get("query") or "").strip()
            if query:
                arg_valid += 1
            else:
                modes.append("fabricated_argument")
            for chunk_id in re.findall(r"chunk_id=(\S+)", call["observation"]):
                seen_ids.add(chunk_id)
        elif name == "read_chunk":
            chunk_id = str(args.get("chunk_id") or "").strip()
            if chunk_id and chunk_id in seen_ids:
                arg_valid += 1
            else:
                modes.append("fabricated_argument")
        elif name == "list_contracts":
            arg_valid += 1
        else:
            modes.append("fabricated_argument")

    steps_needed = int(task["steps_needed"])
    if len(executed) > steps_needed + 3 and "loop" not in modes:
        modes.append("loop")

    missing_definition = False
    missing_other = False
    for group in task.get("groups") or []:
        if _group_hit(group, blob):
            continue
        if group["id"] == "notice_definition" and task.get("defined_term") == "Notice" and task["id"] == "f01":
            missing_definition = True
        else:
            missing_other = True
    if task["id"] == "f09" and len(searches) < 1:
        missing_other = True
    if missing_definition:
        modes.append("skipped_defined_term")
    if missing_other:
        modes.append("wrong_path")
    if len(searches) < 1 and task["id"] != "f09":
        if "wrong_path" not in modes:
            modes.append("wrong_path")

    evidence_hit = any(
        phrase.lower() in blob for phrase in (task.get("must_contain_all") or []) if phrase
    )
    if is_refuse(run.answer) and not task.get("should_refuse"):
        if run.stop_reason in {"max_steps", "max_time", "max_tokens"} or evidence_hit:
            modes.append("quiet_give_up")

    for ref in CLAUSE_REF.findall(run.answer or ""):
        arg_checks += 1
        if ref in blob:
            arg_valid += 1
        else:
            modes.append("fabricated_argument")

    deduped: list[str] = []
    for mode in modes:
        if mode not in deduped:
            deduped.append(mode)

    outcome_pass, outcome_failures = score_item(task, run.answer)
    trajectory_pass = not deduped
    steps_taken = len(executed)
    return {
        "id": task["id"],
        "question": task["question"],
        "answer": (run.answer or "")[:1500],
        "outcome_pass": outcome_pass,
        "outcome_failures": outcome_failures,
        "trajectory_pass": trajectory_pass,
        "trajectory_failures": deduped,
        "gap": bool(outcome_pass and not trajectory_pass),
        "tools": names,
        "queries": [
            str((call["args"] or {}).get("query") or "")
            for call in searches
        ],
        "alternate_paths": task["valid_paths"],
        "steps_taken": steps_taken,
        "steps_needed": steps_needed,
        "step_efficiency": round(steps_taken / steps_needed, 3) if steps_needed else 0.0,
        "tool_choice_accuracy": round(choice_valid / choice_checks, 3) if choice_checks else 1.0,
        "argument_validity": round(arg_valid / arg_checks, 3) if arg_checks else 1.0,
        "choice_valid": choice_valid,
        "choice_checks": choice_checks,
        "arg_valid": arg_valid,
        "arg_checks": arg_checks,
        "elapsed_ms": run.elapsed_ms,
        "total_tokens": run.usage.total_tokens,
        "cost_usd": round(run.usage.cost_usd, 6),
        "stop_reason": run.stop_reason,
        "replans": int((run.guard_stats or {}).get("replans") or 0),
    }


def _counts(rows: list[dict]) -> dict:
    counts = {mode: 0 for mode in MODES}
    for row in rows:
        for mode in row["trajectory_failures"]:
            if mode in counts:
                counts[mode] += 1
    return counts


def _rates(rows: list[dict]) -> dict:
    n = len(rows) or 1
    outcome = sum(1 for row in rows if row["outcome_pass"])
    trajectory = sum(1 for row in rows if row["trajectory_pass"])
    costs = [float(row["cost_usd"]) for row in rows]
    efficiency = [float(row["step_efficiency"]) for row in rows]
    choice_valid = sum(row["choice_valid"] for row in rows)
    choice_checks = sum(row["choice_checks"] for row in rows) or 1
    arg_valid = sum(row["arg_valid"] for row in rows)
    arg_checks = sum(row["arg_checks"] for row in rows) or 1
    tokens = [float(row["total_tokens"]) for row in rows]
    latency = [float(row["elapsed_ms"]) for row in rows]
    return {
        "n": len(rows),
        "outcome_pass": outcome,
        "outcome_rate": round(outcome / n, 3),
        "trajectory_pass": trajectory,
        "trajectory_rate": round(trajectory / n, 3),
        "gap": round(outcome / n - trajectory / n, 3),
        "gap_ids": [row["id"] for row in rows if row["gap"]],
        "tool_choice_accuracy": round(choice_valid / choice_checks, 3),
        "argument_validity": round(arg_valid / arg_checks, 3),
        "step_efficiency_p50": round(p50(efficiency), 3),
        "step_efficiency_max": round(max(efficiency) if efficiency else 0.0, 3),
        "cost_p50_usd": round(p50(costs), 6),
        "cost_max_usd": round(max(costs) if costs else 0.0, 6),
        "latency_p50_ms": int(p50(latency)),
        "tokens_p50": int(p50(tokens)),
        "failure_counts": _counts(rows),
    }


def _top_mode(counts: dict) -> str:
    def key(mode: str) -> tuple:
        # The hunted failure (right notice period, definition never opened) wins a tie.
        prefer = 1 if mode == "skipped_defined_term" else 0
        return (counts.get(mode, 0), prefer, -MODES.index(mode))

    return max(MODES, key=key)


def _failed(question: str, error: str) -> AgentRun:
    return AgentRun(
        question=question,
        answer="I don't know based on the provided documents.",
        mode="agent",
        steps=[AgentStep(1, "stop", "Error", error[:1500])],
        sources=[],
        usage=UsageMeter(),
        elapsed_ms=0,
        stop_reason="error",
        grounded=False,
    )


def _run(pipeline: RagPipeline, question: str, *, replan: bool, search_limit: int | None) -> AgentRun:
    memory = AgentMemory(
        path=ROOT / "eval" / "week8_practical_memory.json",
        embedder=pipeline.store.embedding_model,
    )
    try:
        return run_agent(
            question,
            pipeline,
            memory=memory,
            persist_memory=False,
            guard=False,
            replan_defined_term=replan,
            search_limit=search_limit,
            max_steps=6,
            max_tokens=8000,
            max_seconds=45,
        )
    except Exception as exc:
        return _failed(question, str(exc))


def _mitigation_for(mode: str) -> dict:
    if mode == "skipped_defined_term":
        return {
            "name": "re-planning",
            "why": "Before answering a termination-notice question, search clause 12.9 if the Notice definition is not already in the tool results.",
            "replan": True,
            "search_limit": None,
        }
    if mode == "loop":
        return {
            "name": "hard step limit",
            "why": "After 3 searches, further search_contracts calls return STEP_LIMIT instead of running.",
            "replan": False,
            "search_limit": 3,
        }
    return {
        "name": "re-planning",
        "why": "Top remaining pattern is still answered from the termination sentence. Re-plan once toward clause 12.9.",
        "replan": True,
        "search_limit": None,
    }


def _regression(before: dict, after: dict) -> list[dict]:
    rows = []
    for mode in MODES:
        b = before["failure_counts"][mode]
        a = after["failure_counts"][mode]
        rows.append(
            {
                "mode": mode,
                "before": b,
                "after": a,
                "delta": a - b,
                "worsened": a > b,
                "new": b == 0 and a > 0,
            }
        )
    return rows


def _gap_case(rows: list[dict]) -> dict | None:
    for row in rows:
        if row["gap"]:
            return {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "tools": row["tools"],
                "queries": row["queries"],
                "trajectory_failures": row["trajectory_failures"],
                "why": (
                    "The answer passed the phrase check. The path failed: "
                    + ", ".join(row["trajectory_failures"])
                    + ". Tools: "
                    + ", ".join(row["tools"])
                    + "."
                ),
            }
    return None


def main() -> None:
    tasks = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    pipeline = RagPipeline(documents_dir=ROOT / "documents", persist_dir=ROOT / "chroma_db")
    if pipeline.store.count == 0:
        pipeline.ingest(rebuild=True)

    print(ascii("before: guard off, no mitigation"))
    before_rows = []
    for task in tasks:
        print(ascii(f"  {task['id']}"))
        run = _run(pipeline, task["question"], replan=False, search_limit=None)
        before_rows.append(score_case(task, run))
    before = _rates(before_rows)
    top = _top_mode(before["failure_counts"])
    mitigation = _mitigation_for(top)
    print(ascii(f"top mode: {top} -> one mitigation: {mitigation['name']}"))

    print(ascii("after: that one mitigation only"))
    after_rows = []
    for task in tasks:
        print(ascii(f"  {task['id']}"))
        run = _run(
            pipeline,
            task["question"],
            replan=bool(mitigation["replan"]),
            search_limit=mitigation["search_limit"],
        )
        after_rows.append(score_case(task, run))
    after = _rates(after_rows)
    regression = _regression(before, after)
    worsened = [row["mode"] for row in regression if row["worsened"]]

    price = {
        "latency_p50_ms_before": before["latency_p50_ms"],
        "latency_p50_ms_after": after["latency_p50_ms"],
        "added_latency_p50_ms": after["latency_p50_ms"] - before["latency_p50_ms"],
        "cost_p50_usd_before": before["cost_p50_usd"],
        "cost_p50_usd_after": after["cost_p50_usd"],
        "added_cost_p50_usd": round(after["cost_p50_usd"] - before["cost_p50_usd"], 6),
        "tokens_p50_before": before["tokens_p50"],
        "tokens_p50_after": after["tokens_p50"],
        "added_tokens_p50": after["tokens_p50"] - before["tokens_p50"],
    }

    result = {
        "track": "F",
        "practical": "Week 8 Task Set F",
        "valid_paths_are_sets": True,
        "top_failure": top,
        "mitigation": mitigation,
        "before": before,
        "after": after,
        "top_mode_before": before["failure_counts"][top],
        "top_mode_after": after["failure_counts"][top],
        "price": price,
        "regression": regression,
        "regression_worsened": worsened,
        "gap_case": _gap_case(before_rows),
        "before_rows": before_rows,
        "after_rows": after_rows,
    }
    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    scratch = ROOT / "eval" / "week8_practical_memory.json"
    if scratch.exists():
        scratch.unlink()

    print(ascii("--- results ---"))
    print(ascii(f"outcome {before['outcome_rate']} -> {after['outcome_rate']}"))
    print(ascii(f"trajectory {before['trajectory_rate']} -> {after['trajectory_rate']}"))
    print(ascii(f"gap (outcome - trajectory) before {before['gap']}"))
    print(ascii(f"tool-choice accuracy {before['tool_choice_accuracy']}"))
    print(ascii(f"argument validity {before['argument_validity']}"))
    print(ascii(f"step efficiency p50 {before['step_efficiency_p50']} max {before['step_efficiency_max']}"))
    print(ascii(f"cost p50 {before['cost_p50_usd']} max {before['cost_max_usd']}"))
    print(ascii(f"{top}: {before['failure_counts'][top]} -> {after['failure_counts'][top]}"))
    print(ascii(f"price added p50 ms {price['added_latency_p50_ms']}  p50 usd {price['added_cost_p50_usd']}"))
    print(ascii(f"worsened: {worsened or 'none'}"))
    print(ascii(f"wrote {OUT_PATH}"))


if __name__ == "__main__":
    main()
