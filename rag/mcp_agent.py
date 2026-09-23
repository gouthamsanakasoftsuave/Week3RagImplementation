"""Host-side MCP client. Tool names are discovered. They are not written here."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any

from fastmcp import Client
from groq import Groq

from mcp_server.contract_repository import bind_pipeline, mcp
from rag.agent_types import AgentRun, AgentStep, UsageMeter
from rag.pipeline import RagPipeline

_NAME = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
_BLOCK = ("delete", "drop", "shell", "exec", "subprocess", "write_file", "send_email")

SYSTEM = """You are a contract-research assistant. You are not a lawyer.

Tools are provided by a repository server. Use only those tools.
Answer only from tool results. If the fact is missing, say exactly:
I don't know based on the provided documents.
When you can answer, stop calling tools.
"""


def tool_is_trusted(name: str, description: str) -> tuple[bool, str]:
    if not _NAME.match(name or ""):
        return False, "tool name is not a plain identifier"
    blob = f"{name} {description or ''}".lower()
    for word in _BLOCK:
        if word in blob:
            return False, f"tool looks unsafe ({word})"
    return True, ""


def _specs(tools: list[Any]) -> tuple[list[dict], list[dict], list[str]]:
    specs: list[dict] = []
    refused: list[dict] = []
    names: list[str] = []
    for tool in tools:
        name = tool.name
        description = tool.description or ""
        ok, reason = tool_is_trusted(name, description)
        if not ok:
            refused.append({"name": name, "reason": reason})
            continue
        schema = getattr(tool, "input_schema", None) or {"type": "object", "properties": {}}
        names.append(name)
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": schema,
                },
            }
        )
    return specs, refused, names


def _usage(response) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    return int(getattr(usage, "prompt_tokens", 0) or 0), int(getattr(usage, "completion_tokens", 0) or 0)


def _result_text(result: Any) -> str:
    data = getattr(result, "data", None)
    if isinstance(data, str) and data.strip():
        return data
    content = getattr(result, "content", None) or []
    parts = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts) if parts else str(data or "")


async def _discover(pipeline: RagPipeline) -> tuple[list[Any], list[dict], list[str]]:
    bind_pipeline(pipeline)
    async with Client(mcp) as client:
        tools = await client.list_tools()
    specs, refused, names = _specs(list(tools))
    return specs, refused, names


def discover_tools(pipeline: RagPipeline) -> dict:
    specs, refused, names = asyncio.run(_discover(pipeline))
    return {
        "transport": "MCP in-process client to the contract-repository server",
        "where_the_ai_runs": "in this app, via Groq. The server does not run a model.",
        "discovered": names,
        "refused": refused,
        "raw_tool_schemas": specs,
    }


async def _run(question: str, pipeline: RagPipeline, max_steps: int, max_seconds: float) -> AgentRun:
    bind_pipeline(pipeline)
    started = time.perf_counter()
    steps: list[AgentStep] = []
    usage = UsageMeter()
    answer = ""
    stop_reason = "finished"

    async with Client(mcp) as client:
        specs, refused, names = _specs(list(await client.list_tools()))
        allowed = set(names)
        steps.append(
            AgentStep(
                0,
                "observe",
                "MCP discover",
                "Discovered: " + ", ".join(names) + ("" if not refused else f" | refused: {refused}"),
            )
        )
        if not specs:
            answer = "I don't know based on the provided documents."
            stop_reason = "no_tools"
        else:
            client_llm = Groq(api_key=os.getenv("GROQ_API_KEY"))
            model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
            ]
            for i in range(1, max_steps + 1):
                if time.perf_counter() - started >= max_seconds:
                    stop_reason = "max_time"
                    break
                response = client_llm.chat.completions.create(
                    model=model,
                    temperature=0.1,
                    messages=messages,
                    tools=specs,
                    tool_choice="auto",
                )
                prompt, completion = _usage(response)
                usage.add(prompt, completion, calls=1)
                msg = response.choices[0].message
                text = (msg.content or "").strip()
                tool_calls = list(getattr(msg, "tool_calls", None) or [])
                if text:
                    steps.append(AgentStep(i, "thought", f"Step {i} · think", text[:2000]))
                if not tool_calls:
                    answer = text or "I don't know based on the provided documents."
                    stop_reason = "finished"
                    break
                messages.append(
                    {
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
                )
                for tc in tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    steps.append(
                        AgentStep(i, "tool", f"Step {i} · {name}", json.dumps(args), {"tool": name, "args": args})
                    )
                    if name not in allowed:
                        observation = f"Recoverable error: {name} was not discovered from the server."
                    else:
                        try:
                            result = await client.call_tool(name, args)
                            observation = _result_text(result)[:6000]
                        except Exception as exc:
                            observation = f"Recoverable error from the tool server: {exc}"
                    steps.append(AgentStep(i, "observe", f"Step {i} · result", observation[:4000], {"tool": name}))
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": name,
                            "content": observation[:6000],
                        }
                    )
            else:
                stop_reason = "max_steps"

            if not answer:
                notes = [step.detail[:1500] for step in steps if step.kind == "observe"]
                wrap = client_llm.chat.completions.create(
                    model=model,
                    temperature=0.1,
                    messages=[
                        {
                            "role": "system",
                            "content": "Answer only from the notes. Do not call tools. If the fact is missing, say: I don't know based on the provided documents.",
                        },
                        {"role": "user", "content": f"Question: {question}\n\nNotes:\n" + "\n\n".join(notes)[-8000:]},
                    ],
                )
                prompt, completion = _usage(wrap)
                usage.add(prompt, completion, calls=1)
                answer = (wrap.choices[0].message.content or "").strip()

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    steps.append(AgentStep(len(steps) + 1, "answer", "Final answer", answer))
    grounded = "i don't know based on the provided documents" not in answer.lower()
    return AgentRun(
        question=question,
        answer=answer,
        mode="mcp",
        steps=steps,
        sources=pipeline.store.list_sources(),
        usage=usage,
        elapsed_ms=elapsed_ms,
        stop_reason=stop_reason,
        grounded=grounded,
        guard_stats={
            "discovered": names if specs else [],
            "refused": refused,
            "where_the_ai_runs": "host (this app + Groq). Not on the MCP server.",
        },
    )


def run_mcp_agent(
    question: str,
    pipeline: RagPipeline,
    *,
    max_steps: int = 4,
    max_seconds: float = 45,
) -> AgentRun:
    return asyncio.run(_run(question, pipeline, max_steps, max_seconds))
