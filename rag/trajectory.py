"""Score an agent run on the answer and on the path that produced it."""

from __future__ import annotations

import re
from collections import Counter

from rag.agent_guard import (
    FABRICATED_CHUNK_PREFIX,
    LOOP_BLOCKED_PREFIX,
    MEMORY_LOCK_PREFIX,
    UNKNOWN_TOOL_PREFIX,
)
from rag.agent_race import is_refuse, score_item
from rag.agent_types import AgentRun, AgentStep


def _norm_query(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _blocked(observation: str) -> bool:
    text = observation or ""
    return (
        text.startswith(LOOP_BLOCKED_PREFIX)
        or text.startswith(FABRICATED_CHUNK_PREFIX)
        or text.startswith(UNKNOWN_TOOL_PREFIX)
        or text.startswith(MEMORY_LOCK_PREFIX)
    )


def tool_calls(steps: list[AgentStep]) -> list[dict]:
    """Pair each tool step with the observation that followed it."""
    pending: AgentStep | None = None
    pairs: list[dict] = []
    for step in steps:
        if step.kind == "tool":
            pending = step
            continue
        if step.kind == "observe" and pending is not None:
            payload = pending.payload or {}
            pairs.append(
                {
                    "tool": payload.get("tool") or "",
                    "args": payload.get("args") or {},
                    "observation": step.detail or "",
                    "blocked": _blocked(step.detail or ""),
                }
            )
            pending = None
    return pairs


def _signature(name: str, args: dict) -> str:
    if name == "search_contracts":
        source = _norm_query(str(args.get("source") or ""))
        return "search:" + _norm_query(str(args.get("query") or "")) + "|" + source
    if name in {"read_chunk", "read_contracts", "get_chunk"}:
        return "read:" + str(args.get("chunk_id") or "").strip()
    if name == "recall_memory":
        return "recall:" + _norm_query(str(args.get("query") or ""))
    return name or "unknown"


def score_trajectory(run: AgentRun, task: dict) -> dict:
    """Outcome (right answer) vs trajectory (right steps).

    A gap is an answer that passes the phrase checks while the executed path
    still has a failure: loop, wrong tool, made-up input, quiet give-up, or
    too few searches. Guard blocks are attempts, not executed failures.
    """
    allowed = set(task.get("allowed_tools") or ["search_contracts", "read_chunk"])
    min_searches = int(task.get("min_searches") or (0 if task.get("should_refuse") else 1))
    max_tool_calls = int(task.get("max_tool_calls") or 5)
    calls = tool_calls(run.steps)

    failures: list[str] = []
    executed_sigs: list[str] = []
    seen_ids: set[str] = set()
    searches = 0
    executed = 0
    good = 0
    loop_attempts = 0
    observations: list[str] = []

    for call in calls:
        name = str(call["tool"])
        args = call["args"] if isinstance(call["args"], dict) else {}
        observation = str(call["observation"])
        signature = _signature(name, args)
        if call["blocked"]:
            if observation.startswith(LOOP_BLOCKED_PREFIX):
                loop_attempts += 1
            continue

        executed += 1
        observations.append(observation)

        bad = False
        if name not in allowed:
            failures.append(f"wrong_tool:{name}")
            bad = True
        if signature in executed_sigs:
            failures.append("loop")
            loop_attempts += 1
            bad = True
        if name in {"read_chunk", "read_contracts", "get_chunk"}:
            chunk_id = str(args.get("chunk_id") or "").strip()
            if not chunk_id or chunk_id not in seen_ids:
                failures.append("fabricated_input")
                bad = True
        if name == "search_contracts":
            query = str(args.get("query") or "").strip()
            if not query:
                failures.append("fabricated_input")
                bad = True
            else:
                searches += 1
                for chunk_id in re.findall(r"chunk_id=(\S+)", observation):
                    seen_ids.add(chunk_id)
        if not bad:
            good += 1
        executed_sigs.append(signature)

    if executed > max_tool_calls and "loop" not in failures:
        failures.append("loop")

    if searches < min_searches:
        failures.append("too_few_searches")

    evidence = "\n".join(observations).lower()
    gave_up = is_refuse(run.answer) and not task.get("should_refuse")
    budget_stop = run.stop_reason in {"max_steps", "max_time", "max_tokens"}
    evidence_was_there = any(
        (phrase or "").lower() in evidence for phrase in (task.get("must_contain_all") or [])
    )
    if gave_up and (evidence_was_there or budget_stop):
        failures.append("quiet_give_up")

    # Unique, stable order for the report.
    deduped: list[str] = []
    for item in failures:
        if item not in deduped:
            deduped.append(item)

    outcome_pass, outcome_failures = score_item(task, run.answer)
    trajectory_pass = not deduped
    names = [c["tool"] for c in calls if not c["blocked"]]
    name_counts = Counter(names)

    return {
        "outcome_pass": outcome_pass,
        "outcome_failures": outcome_failures,
        "trajectory_pass": trajectory_pass,
        "trajectory_failures": deduped,
        "gap": bool(outcome_pass and not trajectory_pass),
        "tools": [c["tool"] for c in calls],
        "executed_tools": names,
        "searches": searches,
        "loop_attempts": loop_attempts,
        "tool_choice_accuracy": round(good / executed, 3) if executed else 0.0,
        "tool_counts": dict(name_counts),
        "expected_tools": list(task.get("expected_tools") or []),
    }
