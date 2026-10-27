"""Contract squad: one manager and two specialists, handed off in A2A task states.

MCP is how an agent reaches tools. A2A is how an agent reaches another agent.
This module is the second of those. The specialists still call the same contract
tools as the single agent. The manager never searches. It only splits work and
merges artifacts, so every hand-off starts a new prompt.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from groq import Groq

from rag.agent_memory import AgentMemory
from rag.agent_tools import TOOL_SPECS, make_tool_runner
from rag.agent_types import AgentRun, AgentStep, UsageMeter
from rag.pipeline import RagPipeline

IDK = "I don't know based on the provided documents."

# A2A task states we actually transition through (subset of the protocol).
SUBMITTED = "submitted"
WORKING = "working"
COMPLETED = "completed"
FAILED = "failed"

_STORE_LOCK = threading.Lock()

SEARCH_TOOLS = [
    spec
    for spec in TOOL_SPECS
    if spec["function"]["name"] in {"list_contracts", "search_contracts", "read_chunk"}
]

AGENT_CARDS: list[dict[str, Any]] = [
    {
        "name": "contract-manager",
        "description": (
            "Splits one contract question into two independent sub-questions and "
            "writes the final answer from specialist artifacts only. Does not search."
        ),
        "url": "a2a://local/contract-manager",
        "version": "0.1.0",
        "protocol": "A2A-style in-process handoff",
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": True,
        },
        "skills": [
            {
                "id": "split-question",
                "name": "Split a contract question",
                "description": "Turn one question into two sub-questions that can be researched separately.",
            },
            {
                "id": "merge-artifacts",
                "name": "Merge specialist artifacts",
                "description": "Write the user-facing answer from clause-reader and coverage-checker reports.",
            },
        ],
    },
    {
        "name": "clause-reader",
        "description": (
            "Quotes the clause that answers one assigned sub-question. "
            "Copies names and numbers. Does not decide absence for the whole matter."
        ),
        "url": "a2a://local/clause-reader",
        "version": "0.1.0",
        "protocol": "A2A-style in-process handoff",
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "capabilities": {"streaming": False, "pushNotifications": False, "stateTransitionHistory": True},
        "skills": [
            {
                "id": "quote-clause",
                "name": "Quote a clause",
                "description": "Search contracts and return STATUS, FACT, QUOTE, and SOURCE for one fact.",
            }
        ],
    },
    {
        "name": "coverage-checker",
        "description": (
            "Decides whether the indexed contracts contain one assigned fact. "
            "Absence is a valid result. Different instructions from the clause reader."
        ),
        "url": "a2a://local/coverage-checker",
        "version": "0.1.0",
        "protocol": "A2A-style in-process handoff",
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "capabilities": {"streaming": False, "pushNotifications": False, "stateTransitionHistory": True},
        "skills": [
            {
                "id": "check-coverage",
                "name": "Check whether a fact is in the contracts",
                "description": "Search, then return SUPPORTED or ABSENT with the evidence line.",
            }
        ],
    },
]

SPLIT_SYSTEM = """You are the contract-manager. You do not search and you do not answer the user.
Split the user question into exactly two self-contained sub-questions that can be researched independently.
Return JSON only, no markdown:
{"tasks":[{"specialist":"clause-reader","subquestion":"..."},{"specialist":"coverage-checker","subquestion":"..."}]}
Rules:
- Use each specialist once.
- When the question asks two things, give a different fact to each specialist.
- Copy names, amounts, and the agreement title from the user question. Never invent party names or placeholders.
- Do not answer the question yourself.
"""

CLAUSE_SYSTEM = """You are the clause-reader specialist. You are not a lawyer and you do not give legal advice.
You receive ONE sub-question. Search the contracts and quote the clause that answers it.
Use search_contracts with a short query for this fact only. Use read_chunk if a snippet is cut off.
Do not answer facts you were not asked.
If the retrieved text does not contain the fact, reply exactly:
STATUS: NOT_FOUND
FACT: none
SOURCE: none
Otherwise reply exactly:
STATUS: FOUND
FACT: <one sentence; copy names and numbers from the clause>
QUOTE: <the sentence you relied on>
SOURCE: <filename>
"""

COVERAGE_SYSTEM = """You are the coverage-checker specialist. You are not a lawyer and you do not give legal advice.
You receive ONE sub-question. Your job is different from a clause summary: decide whether the indexed contracts contain this fact.
Search before you decide. If the fact is not in a retrieved clause, absence is the correct result.
Do not use outside knowledge. Do not invent parties, dates, amounts, or clauses.
If you cannot find the fact, reply exactly:
STATUS: ABSENT
FACT: none
SOURCE: none
If you find it, reply exactly:
STATUS: SUPPORTED
FACT: <one sentence; copy names and numbers from the clause>
QUOTE: <the sentence you relied on>
SOURCE: <filename>
"""

MERGE_SYSTEM = f"""You are the contract-manager. You do not search. Write the final answer from the specialist artifacts only.
If any artifact contains STATUS: NOT_FOUND or STATUS: ABSENT, reply with exactly this sentence and nothing else:
{IDK}
If both artifacts are FOUND or SUPPORTED, answer the original question in plain sentences using only their FACT and QUOTE lines.
Copy names and numbers. Do not add facts that are not in the artifacts.
End with Sources: and the filenames from the artifacts.
You are not a lawyer.
"""

_SPECIALIST_SYSTEMS = {
    "clause-reader": CLAUSE_SYSTEM,
    "coverage-checker": COVERAGE_SYSTEM,
}


@dataclass
class SpecialistResult:
    name: str
    task_id: str
    subquestion: str
    state: str
    artifact: str
    steps: list[AgentStep] = field(default_factory=list)
    usage: UsageMeter = field(default_factory=UsageMeter)
    elapsed_ms: int = 0
    sources: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)


def _usage_from(response) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    return int(getattr(usage, "prompt_tokens", 0) or 0), int(
        getattr(usage, "completion_tokens", 0) or 0
    )


def _complete(client: Groq, system: str, user: str, *, tools: list | None = None):
    kwargs: dict[str, Any] = {
        "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if tools is not None:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            last_error = exc
            text = str(exc)
            if tools is None and "tool" in text.lower():
                kwargs["messages"] = list(kwargs["messages"]) + [
                    {
                        "role": "user",
                        "content": "Do not call a tool. Reply in plain text only.",
                    }
                ]
                continue
            if "429" in text or "rate" in text.lower():
                time.sleep(min(2 ** attempt, 8))
                continue
            raise
    assert last_error is not None
    raise last_error


def parse_manager_tasks(text: str, question: str) -> list[dict[str, str]]:
    """Accept the manager JSON, otherwise split the question on 'and'."""
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    data = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.S)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None
    tasks = []
    if isinstance(data, dict):
        for item in data.get("tasks") or []:
            if not isinstance(item, dict):
                continue
            sub = str(item.get("subquestion") or "").strip()
            if sub:
                tasks.append({"subquestion": sub})
    if len(tasks) < 2:
        pieces = re.split(r"\s+\band\b\s+", question, maxsplit=1, flags=re.I)
        if len(pieces) < 2:
            pieces = [question, question]
        tasks = [{"subquestion": pieces[0].strip()}, {"subquestion": pieces[1].strip()}]
    tasks = tasks[:2]
    names = ["clause-reader", "coverage-checker"]
    return [
        {"specialist": names[i], "subquestion": tasks[i]["subquestion"]}
        for i in range(2)
    ]


def _recover_tool_call(exc: Exception) -> tuple[str, dict] | None:
    """Groq sometimes rejects a tool call instead of returning it. Use that call."""
    text = str(exc)
    if "tool" not in text.lower():
        return None
    match = re.search(r"failed_generation': '(\{.*\})'", text)
    if not match:
        match = re.search(r'failed_generation": "(\{.*\})"', text)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    name = str(data.get("name") or "").strip()
    args = data.get("arguments") or {}
    if not name or not isinstance(args, dict):
        return None
    if "filename" in args and "source" not in args:
        args["source"] = args.pop("filename")
    return name, args


def _locked_runner(pipeline: RagPipeline):
    memory = AgentMemory(
        path="eval/week10_blank_memory.json",
        embedder=pipeline.store.embedding_model,
    )
    inner = make_tool_runner(pipeline, memory, search_limit=3)

    def run(name: str, args: dict) -> str:
        with _STORE_LOCK:
            return inner(name, args)

    return run


def _run_specialist(
    pipeline: RagPipeline,
    *,
    name: str,
    subquestion: str,
    task_id: str,
    max_steps: int = 4,
) -> SpecialistResult:
    started = time.perf_counter()
    history = [SUBMITTED, WORKING]
    system = _SPECIALIST_SYSTEMS[name]
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    run_tool = _locked_runner(pipeline)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": subquestion},
    ]
    steps = [
        AgentStep(1, "thought", f"{name} · task {task_id} · {SUBMITTED}", subquestion),
        AgentStep(2, "thought", f"{name} · task {task_id} · {WORKING}", "Specialist started a fresh prompt."),
    ]
    usage = UsageMeter()
    sources: set[str] = set()
    artifact = ""
    state = WORKING

    try:
        for i in range(1, max_steps + 1):
            kwargs: dict[str, Any] = {
                "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
                "temperature": 0.1,
                "messages": messages,
                "tools": SEARCH_TOOLS,
                "tool_choice": "auto",
            }
            last_error: Exception | None = None
            response = None
            for attempt in range(4):
                try:
                    response = client.chat.completions.create(**kwargs)
                    break
                except Exception as exc:
                    last_error = exc
                    recovered = _recover_tool_call(exc)
                    if recovered:
                        tool_name, args = recovered
                        observation = run_tool(tool_name, args)
                        sources.update(re.findall(r"source=([^\s]+)", observation))
                        steps.append(
                            AgentStep(
                                i,
                                "tool",
                                f"{name} · {tool_name}",
                                json.dumps(args, ensure_ascii=False)[:2000],
                            )
                        )
                        steps.append(
                            AgentStep(i, "observe", f"{name} · result", observation[:4000])
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": f"Observation from {tool_name}:\n{observation[:6000]}",
                            }
                        )
                        response = None
                        break
                    if "429" in str(exc) or "rate" in str(exc).lower():
                        time.sleep(min(2 ** attempt, 8))
                        continue
                    raise
            if response is None:
                if last_error is not None and _recover_tool_call(last_error):
                    continue
                if last_error is not None and response is None and "429" not in str(last_error):
                    raise last_error
                raise last_error or RuntimeError("no response")

            prompt_tokens, completion_tokens = _usage_from(response)
            usage.add(prompt_tokens, completion_tokens, calls=1)
            msg = response.choices[0].message
            text = (msg.content or "").strip()
            tool_calls = list(getattr(msg, "tool_calls", None) or [])

            if not tool_calls:
                artifact = text or f"STATUS: NOT_FOUND\nFACT: none\nSOURCE: none"
                state = COMPLETED
                steps.append(AgentStep(i, "observe", f"{name} · artifact", artifact[:4000]))
                break

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": text or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                    for tc in tool_calls
                ],
            }
            messages.append(assistant_msg)
            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                if isinstance(args, dict) and "filename" in args and "source" not in args:
                    args["source"] = args.pop("filename")
                observation = run_tool(tool_name, args)
                sources.update(re.findall(r"source=([^\s]+)", observation))
                steps.append(
                    AgentStep(
                        i,
                        "tool",
                        f"{name} · {tool_name}",
                        json.dumps(args, ensure_ascii=False)[:2000],
                    )
                )
                steps.append(
                    AgentStep(i, "observe", f"{name} · result", observation[:4000])
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tool_name,
                        "content": observation[:6000],
                    }
                )
        else:
            notes = "\n\n".join(
                f"{step.title}: {step.detail[:1500]}" for step in steps if step.kind == "observe"
            )
            wrap = _complete(
                client,
                system + "\nDo not call tools. Reply in the STATUS format now.",
                f"Sub-question: {subquestion}\n\nNotes:\n{notes or '(none)'}",
            )
            prompt_tokens, completion_tokens = _usage_from(wrap)
            usage.add(prompt_tokens, completion_tokens, calls=1)
            artifact = (wrap.choices[0].message.content or "").strip()
            state = COMPLETED
            steps.append(AgentStep(max_steps, "observe", f"{name} · artifact", artifact[:4000]))
    except Exception as exc:
        state = FAILED
        artifact = f"STATUS: NOT_FOUND\nFACT: none\nSOURCE: none\nERROR: {exc}"
        steps.append(AgentStep(0, "stop", f"{name} · {FAILED}", str(exc)[:2000]))

    history.append(state)
    steps.append(AgentStep(len(steps) + 1, "stop", f"{name} · task {task_id} · {state}", artifact[:2000]))
    return SpecialistResult(
        name=name,
        task_id=task_id,
        subquestion=subquestion,
        state=state,
        artifact=artifact,
        steps=steps,
        usage=usage,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        sources=sorted(sources),
        history=history,
    )


def run_squad(
    question: str,
    pipeline: RagPipeline,
    *,
    max_steps: int = 4,
) -> AgentRun:
    """Manager splits, both specialists run together, manager merges.

    Wall-clock time overlaps the two specialist loops. Token totals do not:
    each specialist is billed for a new system prompt plus its sub-question,
    and the merge call is billed again for both artifacts.
    """
    started = time.perf_counter()
    usage = UsageMeter()
    steps: list[AgentStep] = []
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    split = _complete(client, SPLIT_SYSTEM, question)
    split_prompt, split_completion = _usage_from(split)
    usage.add(split_prompt, split_completion, calls=1)
    split_text = (split.choices[0].message.content or "").strip()
    tasks = parse_manager_tasks(split_text, question)
    steps.append(AgentStep(1, "thought", "Manager · split", split_text[:4000] or json.dumps(tasks)))

    work = []
    for task in tasks:
        task_id = uuid.uuid4().hex[:8]
        work.append((task["specialist"], task["subquestion"], task_id))
        steps.append(
            AgentStep(
                2,
                "tool",
                f"A2A handoff · {task['specialist']} · {SUBMITTED}",
                task["subquestion"],
                {"task_id": task_id, "state": SUBMITTED, "specialist": task["specialist"]},
            )
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                _run_specialist,
                pipeline,
                name=name,
                subquestion=subquestion,
                task_id=task_id,
                max_steps=max_steps,
            )
            for name, subquestion, task_id in work
        ]
        results = [future.result() for future in futures]

    specialist_prompt = 0
    sources: set[str] = set()
    reports = []
    task_rows = []
    for result in results:
        usage.add(
            result.usage.prompt_tokens,
            result.usage.completion_tokens,
            calls=result.usage.llm_calls,
        )
        specialist_prompt += result.usage.prompt_tokens
        sources.update(result.sources)
        steps.extend(result.steps)
        reports.append(
            f"Specialist: {result.name}\n"
            f"Task: {result.task_id}\n"
            f"State: {result.state}\n"
            f"Sub-question: {result.subquestion}\n"
            f"{result.artifact}"
        )
        task_rows.append(
            {
                "id": result.task_id,
                "specialist": result.name,
                "state": result.state,
                "history": result.history,
                "subquestion": result.subquestion,
                "elapsed_ms": result.elapsed_ms,
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "artifact": result.artifact,
            }
        )

    merge_user = "Original question:\n" + question + "\n\nSpecialist artifacts:\n\n" + "\n\n---\n\n".join(reports)
    merge = _complete(client, MERGE_SYSTEM, merge_user)
    merge_prompt, merge_completion = _usage_from(merge)
    usage.add(merge_prompt, merge_completion, calls=1)
    answer = (merge.choices[0].message.content or "").strip() or IDK
    steps.append(AgentStep(len(steps) + 1, "answer", "Manager · final answer", answer))

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    grounded = IDK.lower() not in answer.lower()
    failed = [row for row in task_rows if row["state"] == FAILED]
    return AgentRun(
        question=question,
        answer=answer,
        mode="squad",
        steps=steps,
        sources=sorted(sources),
        usage=usage,
        elapsed_ms=elapsed_ms,
        stop_reason="failed" if len(failed) == len(task_rows) else "finished",
        grounded=grounded,
        guard_stats={
            "pattern": "orchestrator-worker",
            "protocol": "A2A task handoff (in-process). Tools stay on the specialist side.",
            "mcp_vs_a2a": (
                "MCP connects an agent to tools (search_contracts). "
                "A2A connects the manager to the two specialists."
            ),
            "agent_cards": [card["name"] for card in AGENT_CARDS],
            "tasks": task_rows,
            "split_prompt_tokens": split_prompt,
            "specialist_prompt_tokens": specialist_prompt,
            "synthesis_prompt_tokens": merge_prompt,
            "context_resend_prompt_tokens": specialist_prompt + merge_prompt,
            "parallel_specialists": True,
        },
    )
