"""Fixed contract workflow: always the same steps, no LLM choosing tools."""

from __future__ import annotations

import re
import time

from rag.agent_memory import AgentMemory
from rag.agent_tools import format_chunks
from rag.agent_types import AgentRun, AgentStep, UsageMeter
from rag.generate import generate_answer
from rag.pipeline import RagPipeline

AND_SPLIT = re.compile(r"\b(?:and then|and also|, then|; then|\? then)\b|, and |\. then ", re.I)


def _split_parts(question: str) -> list[str]:
    parts = [p.strip(" ?") for p in AND_SPLIT.split(question) if p.strip(" ?")]
    if len(parts) >= 2:
        return parts[:3]
    if " and " in question.lower() and len(question) > 80:
        bits = [p.strip(" ?") for p in re.split(r"\band\b", question, flags=re.I) if p.strip(" ?")]
        if 2 <= len(bits) <= 3:
            return bits
    return [question.strip()]


def run_workflow(
    question: str,
    pipeline: RagPipeline,
    *,
    memory: AgentMemory | None = None,
    persist_memory: bool = True,
) -> AgentRun:
    """Always: list contracts → search each sub-question → one grounded answer."""
    started = time.perf_counter()
    steps: list[AgentStep] = []
    usage = UsageMeter()
    n = 0

    n += 1
    names = pipeline.store.list_sources()
    steps.append(
        AgentStep(
            n,
            "tool",
            f"Step {n} · list_contracts (fixed)",
            ", ".join(names) or "(none)",
            {"tool": "list_contracts"},
        )
    )

    parts = _split_parts(question)
    n += 1
    steps.append(
        AgentStep(
            n,
            "thought",
            f"Step {n} · split question (rules, not an LLM)",
            " | ".join(parts),
        )
    )

    gathered = []
    seen_ids: set[str] = set()
    searches = list(parts)
    # Always one extra pass on the full question so nothing is dropped.
    if question.strip() not in searches:
        searches.append(question.strip())

    for query in searches[:4]:
        n += 1
        chunks = pipeline.retrieve(
            query, top_k=4, source_filter=None, mode="hybrid", rewrite=True
        )
        steps.append(
            AgentStep(
                n,
                "tool",
                f"Step {n} · search_contracts (fixed)",
                query,
                {"tool": "search_contracts", "args": {"query": query}},
            )
        )
        steps.append(
            AgentStep(
                n,
                "observe",
                f"Step {n} · result",
                format_chunks(chunks)[:4000],
            )
        )
        for c in chunks:
            if c.chunk_id not in seen_ids:
                seen_ids.add(c.chunk_id)
                gathered.append(c)

    n += 1
    rag = generate_answer(question, gathered)
    usage.add(rag.prompt_tokens, rag.completion_tokens, calls=rag.llm_calls)
    steps.append(AgentStep(n, "answer", f"Step {n} · generate answer (once)", rag.answer))

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    if persist_memory:
        mem = memory or AgentMemory(embedder=pipeline.store.embedding_model)
        mem.remember_task(question, rag.answer, rag.sources)

    return AgentRun(
        question=question,
        answer=rag.answer,
        mode="workflow",
        steps=steps,
        sources=rag.sources,
        usage=usage,
        elapsed_ms=elapsed_ms,
        stop_reason="finished",
        grounded=rag.grounded,
    )
