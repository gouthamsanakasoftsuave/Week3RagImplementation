"""Score tasks and race agent vs fixed workflow."""

from __future__ import annotations

import json
import re
from pathlib import Path

from rag.agent_loop import run_agent
from rag.agent_memory import AgentMemory
from rag.agent_types import AgentRun, AgentStep, UsageMeter
from rag.agent_workflow import run_workflow
from rag.pipeline import RagPipeline

IDK = "i don't know based on the provided documents"


def norm(text: str) -> str:
    t = (text or "").lower()
    t = t.replace("\u20b9", "rs").replace(",", "")
    t = re.sub(r"\s+", " ", t)
    return t


def has_phrase(haystack: str, needle: str) -> bool:
    return norm(needle) in norm(haystack)


def is_refuse(answer: str) -> bool:
    return IDK in (answer or "").lower()


def score_item(item: dict, answer: str) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if item.get("should_refuse"):
        if not is_refuse(answer):
            failures.append("expected refuse but did not say I don't know")
        return not failures, failures
    if is_refuse(answer):
        failures.append("refused but the clauses are in the contract")
    for phrase in item.get("must_contain_all") or []:
        if not has_phrase(answer, phrase):
            failures.append(f"missing required phrase: {phrase}")
    any_need = item.get("must_contain_any") or []
    if any_need and not any(has_phrase(answer, p) for p in any_need):
        failures.append(f"missing any of: {any_need}")
    return not failures, failures


def metrics(run: AgentRun) -> dict:
    return {
        "answer": run.answer,
        "elapsed_ms": run.elapsed_ms,
        "llm_calls": run.usage.llm_calls,
        "prompt_tokens": run.usage.prompt_tokens,
        "completion_tokens": run.usage.completion_tokens,
        "total_tokens": run.usage.total_tokens,
        "cost_usd": round(run.usage.cost_usd, 6),
        "stop_reason": run.stop_reason,
        "step_count": len(run.steps),
        "sources": run.sources,
    }


def ship_recommendation(summary: dict) -> str:
    a = summary["agent"]
    w = summary["workflow"]
    if w["reliability_pct"] >= a["reliability_pct"] and w["avg_ms"] <= a["avg_ms"]:
        return (
            "Ship the fixed workflow. It matched or beat the agent on reliability and was "
            "faster. The steps are known in advance (list → search each part → answer once)."
        )
    if a["reliability_pct"] >= w["reliability_pct"] + 15:
        return (
            "Consider the agent only if this extra reliability is worth the extra calls. "
            "Otherwise keep the workflow as the default path."
        )
    return (
        "Ship the fixed workflow as the default. The agent is useful when the next document "
        "or clause depends on what you just found; this race's tasks have a stable path."
    )


def _failed_run(question: str, mode: str, error: str) -> AgentRun:
    return AgentRun(
        question=question,
        answer="I don't know based on the provided documents.",
        mode=mode,
        steps=[AgentStep(1, "stop", "Error", error[:2000])],
        sources=[],
        usage=UsageMeter(),
        elapsed_ms=0,
        stop_reason="error",
        grounded=False,
    )


def run_race(
    pipeline: RagPipeline,
    *,
    tasks_path: str | Path = "eval/week7_tasks.json",
    out_path: str | Path = "eval/week7_race.json",
    memory_path: str | Path = "eval/agent_memory.json",
) -> dict:
    tasks_path = Path(tasks_path)
    out_path = Path(out_path)
    items = json.loads(tasks_path.read_text(encoding="utf-8"))
    if pipeline.store.count == 0:
        pipeline.ingest(rebuild=True)

    memory = AgentMemory(path=memory_path, embedder=pipeline.store.embedding_model)
    rows = []
    for item in items:
        q = item["question"]
        try:
            wf = run_workflow(q, pipeline, memory=memory, persist_memory=False)
        except Exception as exc:
            wf = _failed_run(q, "workflow", str(exc))
        try:
            ag = run_agent(q, pipeline, memory=memory, persist_memory=False)
        except Exception as exc:
            ag = _failed_run(q, "agent", str(exc))
        ok_w, fail_w = score_item(item, wf.answer)
        ok_a, fail_a = score_item(item, ag.answer)
        rows.append(
            {
                "id": item["id"],
                "question": q,
                "workflow": {**metrics(wf), "pass": ok_w, "failures": fail_w},
                "agent": {**metrics(ag), "pass": ok_a, "failures": fail_a},
            }
        )

    def agg(key: str) -> dict:
        chunk = [r[key] for r in rows]
        n = len(chunk) or 1
        passed = sum(1 for x in chunk if x["pass"])
        return {
            "pass": passed,
            "n": len(chunk),
            "reliability_pct": round(100.0 * passed / n, 1),
            "avg_ms": int(sum(x["elapsed_ms"] for x in chunk) / n),
            "avg_llm_calls": round(sum(x["llm_calls"] for x in chunk) / n, 2),
            "total_tokens": sum(x["total_tokens"] for x in chunk),
            "total_cost_usd": round(sum(x["cost_usd"] for x in chunk), 6),
        }

    summary = {"workflow": agg("workflow"), "agent": agg("agent")}
    summary["ship"] = ship_recommendation(summary)
    result = {"summary": summary, "rows": rows}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
