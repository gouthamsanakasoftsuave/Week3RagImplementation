"""Race the contract squad against the single agent on the same tasks."""

from __future__ import annotations

import json
import time
from pathlib import Path

from rag.agent_loop import run_agent
from rag.agent_memory import AgentMemory
from rag.agent_race import _failed_run, metrics, score_item
from rag.pipeline import RagPipeline
from rag.squad import AGENT_CARDS, run_squad

WHEN_IT_HELPS = (
    "A squad is worth the re-send when the parts of the job are independent and can "
    "run at the same time, or when the specialists need instructions that would fight "
    "each other inside one prompt (quote a clause versus decide the fact is absent). "
    "It is not worth it when one agent can search and answer on its own. Every hand-off "
    "starts a new prompt, so the question and the retrieved clauses are paid for again."
)


def _blank_memory(pipeline: RagPipeline) -> AgentMemory:
    return AgentMemory(
        path="eval/week10_blank_memory.json",
        embedder=pipeline.store.embedding_model,
    )


def choose_winner(summary: dict) -> dict:
    single = summary["single"]
    team = summary["team"]

    def winner(single_value: float, team_value: float, *, lower_is_better: bool) -> str:
        if single_value == team_value:
            return "tie"
        if lower_is_better:
            return "team" if team_value < single_value else "single"
        return "team" if team_value > single_value else "single"

    quality = winner(single["reliability_pct"], team["reliability_pct"], lower_is_better=False)
    speed = winner(single["avg_ms"], team["avg_ms"], lower_is_better=True)
    tokens = winner(single["total_tokens"], team["total_tokens"], lower_is_better=True)
    cost = winner(single["total_cost_usd"], team["total_cost_usd"], lower_is_better=True)

    cost_ratio = team["total_cost_usd"] / single["total_cost_usd"] if single["total_cost_usd"] else None
    if quality == "team" and (cost_ratio is None or cost_ratio <= 1.5):
        keep = "team"
        reason = (
            "The squad passed more of the same tests, and its cost stayed within 1.5× "
            "of the single agent. Keep the squad for this task."
        )
    elif quality == "tie" and cost == "team" and speed in {"team", "tie"}:
        keep = "team"
        reason = (
            "Quality matched, and the squad was cheaper and at least as fast. Keep the squad."
        )
    elif quality == "team":
        keep = "single"
        reason = (
            "The squad was more accurate, but it cost more than 1.5× the single agent "
            "because each hand-off re-sends context. Keep the single agent unless that "
            "accuracy gap is worth the extra bill."
        )
    else:
        keep = "single"
        reason = (
            "The single agent matched or beat the squad on the same tests. "
            "The extra hand-offs did not buy a better answer. Keep the single agent."
        )

    return {
        "quality": quality,
        "speed": speed,
        "tokens": tokens,
        "cost": cost,
        "keep": keep,
        "reason": reason,
        "cost_ratio_team_over_single": None if cost_ratio is None else round(cost_ratio, 2),
        "when_multi_agent_is_worth_it": WHEN_IT_HELPS,
    }


def _agg(rows: list[dict], key: str) -> dict:
    chunk = [row[key] for row in rows]
    n = len(chunk) or 1
    passed = sum(1 for item in chunk if item["pass"])
    return {
        "pass": passed,
        "n": len(chunk),
        "reliability_pct": round(100.0 * passed / n, 1),
        "avg_ms": int(sum(item["elapsed_ms"] for item in chunk) / n),
        "avg_llm_calls": round(sum(item["llm_calls"] for item in chunk) / n, 2),
        "total_tokens": sum(item["total_tokens"] for item in chunk),
        "total_prompt_tokens": sum(item["prompt_tokens"] for item in chunk),
        "total_cost_usd": round(sum(item["cost_usd"] for item in chunk), 6),
    }


def write_report(result: dict, path: Path) -> None:
    single = result["summary"]["single"]
    team = result["summary"]["team"]
    verdict = result["verdict"]
    lines = [
        "# Week 10 · Track F · Contract squad vs single agent",
        "",
        "Same six multi-hop contract questions as Week 7 (`eval/week7_tasks.json`).",
        "Same phrase scorer. Same model. The single agent is the existing ReAct loop.",
        "The squad is a manager plus two specialists (`clause-reader`, `coverage-checker`).",
        "Specialists run at the same time. The manager does not search.",
        "",
        "## The four numbers",
        "",
        "| | Quality (tests passed) | Speed (avg ms) | Tokens | Cost (USD) |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Single agent | {single['pass']}/{single['n']} ({single['reliability_pct']}%) | "
            f"{single['avg_ms']} | {single['total_tokens']} | {single['total_cost_usd']:.6f} |"
        ),
        (
            f"| Contract squad | {team['pass']}/{team['n']} ({team['reliability_pct']}%) | "
            f"{team['avg_ms']} | {team['total_tokens']} | {team['total_cost_usd']:.6f} |"
        ),
        "",
        f"Context re-sent on hand-offs (specialist prompts + manager merge prompts): "
        f"**{result['summary']['context_resend_prompt_tokens']}** prompt tokens. "
        f"The single agent used **{single['total_prompt_tokens']}** prompt tokens in total.",
        "",
        "## Verdict",
        "",
        f"**Keep the {verdict['keep']}.** {verdict['reason']}",
        "",
        (
            f"Winners by metric — quality: {verdict['quality']}, speed: {verdict['speed']}, "
            f"tokens: {verdict['tokens']}, cost: {verdict['cost']}."
        ),
        "",
        "## When a team is worth it",
        "",
        verdict["when_multi_agent_is_worth_it"],
        "",
        "## A2A and MCP",
        "",
        "MCP connects an agent to tools. The specialists call `search_contracts` through the same tool runner as the single agent.",
        "A2A connects an agent to another agent. The manager submits a task, the specialist moves submitted → working → completed (or failed), and returns an artifact.",
        "Agent cards for `contract-manager`, `clause-reader`, and `coverage-checker` are in the race JSON.",
        "This hand-off is in-process so the token meter stays visible. CrewAI and AutoGen implement the same orchestrator–worker pattern behind a framework; they are not used here because they would hide the re-send.",
        "",
        "## Per question",
        "",
        "| id | single | squad | single ms | squad ms | single tokens | squad tokens |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| {row['id']} | {'pass' if row['single']['pass'] else 'fail'} | "
            f"{'pass' if row['team']['pass'] else 'fail'} | {row['single']['elapsed_ms']} | "
            f"{row['team']['elapsed_ms']} | {row['single']['total_tokens']} | {row['team']['total_tokens']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_squad_race(
    pipeline: RagPipeline,
    *,
    tasks_path: str | Path = "eval/week7_tasks.json",
    out_path: str | Path = "eval/week10_race.json",
    report_path: str | Path = "WEEK_10_REPORT.md",
) -> dict:
    tasks_path = Path(tasks_path)
    out_path = Path(out_path)
    report_path = Path(report_path)
    items = json.loads(tasks_path.read_text(encoding="utf-8"))
    if pipeline.store.count == 0:
        pipeline.ingest(rebuild=True)

    rows = []
    for item in items:
        question = item["question"]
        memory = _blank_memory(pipeline)
        single = None
        single_error = ""
        for attempt in range(2):
            try:
                single = run_agent(question, pipeline, memory=memory, persist_memory=False)
                break
            except Exception as exc:
                single_error = str(exc)
                time.sleep(2)
        if single is None:
            single = _failed_run(question, "agent", single_error)
        try:
            team = run_squad(question, pipeline)
        except Exception as exc:
            team = _failed_run(question, "squad", str(exc))
        ok_s, fail_s = score_item(item, single.answer)
        ok_t, fail_t = score_item(item, team.answer)
        if single.stop_reason == "error":
            ok_s = False
            fail_s = ["run failed before an answer", *fail_s]
        if team.stop_reason == "error":
            ok_t = False
            fail_t = ["run failed before an answer", *fail_t]
        team_stats = team.guard_stats or {}
        print(
            f"[{item['id']}] single {'pass' if ok_s else 'fail'} {single.elapsed_ms}ms | "
            f"squad {'pass' if ok_t else 'fail'} {team.elapsed_ms}ms",
            flush=True,
        )
        rows.append(
            {
                "id": item["id"],
                "question": question,
                "single": {**metrics(single), "pass": ok_s, "failures": fail_s},
                "team": {
                    **metrics(team),
                    "pass": ok_t,
                    "failures": fail_t,
                    "context_resend_prompt_tokens": team_stats.get("context_resend_prompt_tokens", 0),
                    "tasks": team_stats.get("tasks") or [],
                },
            }
        )

    summary = {"single": _agg(rows, "single"), "team": _agg(rows, "team")}
    summary["context_resend_prompt_tokens"] = sum(
        row["team"].get("context_resend_prompt_tokens") or 0 for row in rows
    )
    verdict = choose_winner(summary)
    result = {
        "track": "F",
        "topic": "Legal contracts",
        "tasks": str(tasks_path).replace("\\", "/"),
        "same_tests": True,
        "agent_cards": AGENT_CARDS,
        "summary": summary,
        "verdict": verdict,
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result, report_path)
    return result
