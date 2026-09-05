"""Hand-built ReAct loop: plan → tool → observe → repeat, with budgets."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from groq import Groq

from rag.agent_memory import AgentMemory
from rag.agent_tools import TOOL_SPECS, make_tool_runner
from rag.agent_types import AgentRun, AgentStep, UsageMeter
from rag.pipeline import RagPipeline

SYSTEM = """You are a contract-research agent. You are not a lawyer and you do not give legal advice.

Loop:
1. Think about what fact you still need.
2. Call one tool to get it (prefer focused search_contracts queries).
3. Read the observation. Repeat until you can answer.

Rules:
- Answer ONLY from tool observations (contracts or recalled summaries of those contracts).
- Multi-part questions need a separate search per fact.
- If evidence is missing, say exactly: I don't know based on the provided documents.
- Do not invent parties, dates, amounts, or clauses.
- When done, stop calling tools and write the final answer. End with Sources: filenames.
"""

REACT_SYSTEM = SYSTEM + """

If you cannot use API tools, reply in exactly this shape:
Thought: <one sentence>
Action: <list_contracts|search_contracts|read_chunk|recall_memory>
Action Input: {"query": "..."} 

Or when finished:
Thought: <one sentence>
Final Answer: <answer>
"""


def _usage_from(response) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    return int(getattr(usage, "prompt_tokens", 0) or 0), int(
        getattr(usage, "completion_tokens", 0) or 0
    )


def _parse_react_text(text: str) -> tuple[str | None, dict | None, str | None]:
    """Fallback parser if the model writes Thought/Action instead of tool calls."""
    final = re.search(r"Final Answer:\s*(.*)", text or "", re.S | re.I)
    if final:
        return None, None, final.group(1).strip()
    action = re.search(r"Action:\s*([A-Za-z0-9_]+)", text or "")
    raw_input = re.search(r"Action Input:\s*(\{.*\}|.*)", text or "", re.S)
    if not action:
        return None, None, None
    name = action.group(1).strip()
    args: dict = {}
    if raw_input:
        blob = raw_input.group(1).strip()
        try:
            args = json.loads(blob)
        except json.JSONDecodeError:
            if name == "search_contracts":
                args = {"query": blob.strip().strip('"')}
            elif name == "read_chunk":
                args = {"chunk_id": blob.strip().strip('"')}
            elif name == "recall_memory":
                args = {"query": blob.strip().strip('"')}
    return name, args, None


def run_agent(
    question: str,
    pipeline: RagPipeline,
    *,
    memory: AgentMemory | None = None,
    max_steps: int | None = None,
    max_tokens: int | None = None,
    max_seconds: float | None = None,
    persist_memory: bool = True,
) -> AgentRun:
    max_steps = int(max_steps or os.getenv("AGENT_MAX_STEPS", "8"))
    max_tokens = int(max_tokens or os.getenv("AGENT_MAX_TOKENS", "12000"))
    max_seconds = float(max_seconds or os.getenv("AGENT_MAX_SECONDS", "90"))

    memory = memory or AgentMemory(embedder=pipeline.store.embedding_model)
    run_tool = make_tool_runner(pipeline, memory)
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    prior = memory.recall(question, k=2)
    memory_hits = [h.summary for h in prior]
    prior_block = ""
    if prior:
        prior_block = "\n\nRelated past-task summaries (verify with tools if you use them):\n" + "\n".join(
            f"- {h.question}: {h.summary}" for h in prior
        )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question + prior_block},
    ]
    steps: list[AgentStep] = []
    usage = UsageMeter()
    started = time.perf_counter()
    stop_reason = "finished"
    answer = ""
    sources: set[str] = set()
    use_native_tools = True

    for i in range(1, max_steps + 1):
        elapsed = time.perf_counter() - started
        if elapsed >= max_seconds:
            stop_reason = "max_time"
            break
        if usage.total_tokens >= max_tokens:
            stop_reason = "max_tokens"
            break

        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": 0.1,
            "messages": messages,
        }
        if use_native_tools:
            kwargs["tools"] = TOOL_SPECS
            kwargs["tool_choice"] = "auto"
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:
            if use_native_tools:
                use_native_tools = False
                messages[0] = {"role": "system", "content": REACT_SYSTEM}
                steps.append(
                    AgentStep(
                        i,
                        "thought",
                        "Native tools unavailable — switching to text ReAct",
                        str(exc)[:500],
                    )
                )
                continue
            raise

        p, c = _usage_from(response)
        usage.add(p, c, calls=1)
        msg = response.choices[0].message
        text = (msg.content or "").strip()
        tool_calls = list(getattr(msg, "tool_calls", None) or [])

        if text:
            steps.append(AgentStep(i, "thought", f"Step {i} · think", text[:2000]))
            memory.note(text[:500])
        elif tool_calls:
            names = ", ".join(tc.function.name for tc in tool_calls)
            steps.append(
                AgentStep(i, "thought", f"Step {i} · plan", f"Next: call {names}")
            )

        if not tool_calls and text:
            name, args, final = _parse_react_text(text)
            if final:
                answer = final
                stop_reason = "finished"
                break
            if name:
                class _Fn:
                    def __init__(self, n, a):
                        self.name = n
                        self.arguments = json.dumps(a or {})

                class _Call:
                    def __init__(self, n, a):
                        self.id = f"react-{i}"
                        self.function = _Fn(n, a)

                tool_calls = [_Call(name, args or {})]

        if not tool_calls:
            answer = text or "I don't know based on the provided documents."
            stop_reason = "finished"
            break

        native_calls = bool(getattr(msg, "tool_calls", None))
        if native_calls and use_native_tools:
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                }
                for tc in tool_calls
            ]
            messages.append(assistant_msg)
        else:
            messages.append({"role": "assistant", "content": text or json.dumps(
                {"action": tool_calls[0].function.name, "arguments": tool_calls[0].function.arguments}
            )})

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            observation = run_tool(name, args)
            memory.note(f"{name}: {observation[:400]}")
            src_matches = re.findall(r"source=([^\s]+)", observation)
            sources.update(src_matches)
            steps.append(
                AgentStep(
                    i,
                    "tool",
                    f"Step {i} · {name}",
                    json.dumps(args, ensure_ascii=False),
                    {"tool": name, "args": args},
                )
            )
            steps.append(
                AgentStep(
                    i,
                    "observe",
                    f"Step {i} · result",
                    observation[:4000],
                    {"tool": name},
                )
            )
            if native_calls and use_native_tools:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": observation[:6000],
                    }
                )
            else:
                messages.append({"role": "user", "content": f"Observation from {name}:\n{observation[:6000]}"})
    else:
        stop_reason = "max_steps"

    if not answer:
        # Fresh prompt — do not reuse the tool-call transcript. Groq errors if
        # tool_choice is none but the model still emits a function call.
        notes = []
        for step in steps:
            if step.kind in {"observe", "thought"}:
                notes.append(f"{step.title}: {step.detail[:1500]}")
        blob = "\n\n".join(notes)[-12000:]
        wrap = client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are finishing a contract-research task. Do not call tools. "
                        "Answer only from the notes. If evidence is incomplete, reply exactly: "
                        "I don't know based on the provided documents."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nNotes:\n{blob or '(none)'}",
                },
            ],
        )
        p, c = _usage_from(wrap)
        usage.add(p, c, calls=1)
        answer = (wrap.choices[0].message.content or "").strip() or (
            "Stopped before a complete answer. I don't know based on the provided documents."
        )
        steps.append(AgentStep(max_steps, "stop", f"Stopped ({stop_reason})", answer[:2000]))

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    grounded = "i don't know based on the provided documents" not in answer.lower()
    source_list = sorted(sources) or pipeline.store.list_sources()
    if persist_memory and answer:
        memory.remember_task(question, answer, source_list)

    steps.append(AgentStep(len(steps) + 1, "answer", "Final answer", answer))
    return AgentRun(
        question=question,
        answer=answer,
        mode="agent",
        steps=steps,
        sources=source_list,
        usage=usage,
        elapsed_ms=elapsed_ms,
        stop_reason=stop_reason,
        grounded=grounded,
        memory_hits=memory_hits,
    )
