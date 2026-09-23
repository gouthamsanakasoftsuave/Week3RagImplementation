"""Contract tools with descriptions the model can choose from."""

from __future__ import annotations

import json
from typing import Callable

from rag.agent_guard import ToolGuard, sanitize_untrusted
from rag.agent_memory import AgentMemory
from rag.pipeline import RagPipeline
from rag.store import RetrievedChunk

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "list_contracts",
            "description": (
                "List indexed contract filenames. Use this first when you need to know "
                "which agreements exist, or before restricting a search to one file."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_contracts",
            "description": (
                "Hybrid search over contract chunks (keywords + meaning). Use a focused "
                "query for ONE fact (e.g. 'termination notice period', not the whole user "
                "question). Optionally restrict to one filename from list_contracts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Short search query for a single clause or fact.",
                    },
                    "source": {
                        "type": "string",
                        "description": "Optional filename, e.g. Service-Provider-Agreement.pdf",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_chunk",
            "description": (
                "Load the full text of a chunk you already saw in search results. "
                "Use when the search snippet was cut off."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "The chunk_id returned by search_contracts.",
                    }
                },
                "required": ["chunk_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": (
                "Search summaries of earlier tasks this app already completed. "
                "Use for follow-up questions, not as a substitute for searching the contracts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look up in past tasks."}
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]


def _clip(text: str, limit: int = 900) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def format_chunks(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No matching chunks."
    parts = []
    for i, c in enumerate(chunks, start=1):
        page = c.metadata.get("page") or ""
        page_note = f" page={page}" if page else ""
        parts.append(
            f"[{i}] chunk_id={c.chunk_id} source={c.source}{page_note} score={c.score:.4f}\n"
            f"{_clip(c.content)}"
        )
    return "\n\n".join(parts)


def _with_attack_doc(observation: str, untrusted_document: str | None) -> str:
    if not untrusted_document:
        return observation
    extra = (
        "[extra] chunk_id=side-letter.txt::p1::c0 source=side-letter.txt page=1 score=0.9900\n"
        + untrusted_document.strip()
    )
    if not observation or observation == "No matching chunks.":
        return extra
    return observation + "\n\n" + extra


def make_tool_runner(
    pipeline: RagPipeline,
    memory: AgentMemory,
    *,
    guard: ToolGuard | None = None,
    untrusted_document: str | None = None,
    search_limit: int | None = None,
) -> Callable[[str, dict], str]:
    guard_state = guard or ToolGuard(enabled=False)

    def run(name: str, args: dict) -> str:
        args = args or {}
        if (
            search_limit is not None
            and name == "search_contracts"
            and guard_state.search_count >= search_limit
        ):
            return (
                "STEP_LIMIT: stop searching and answer from the observations you already have."
            )
        blocked = guard_state.before_call(name, args)
        if blocked is not None:
            return blocked

        if name == "list_contracts":
            names = pipeline.store.list_sources()
            text = json.dumps({"contracts": names, "count": len(names)})
        elif name == "search_contracts":
            query = str(args.get("query") or "").strip()
            source = str(args.get("source") or "").strip() or None
            if not query:
                text = "Error: query is required."
            else:
                chunks = pipeline.retrieve(
                    query,
                    top_k=4,
                    source_filter=source,
                    mode="hybrid",
                    rewrite=True,
                )
                text = _with_attack_doc(format_chunks(chunks), untrusted_document)
        elif name in {"read_chunk", "read_contracts", "get_chunk"}:
            chunk_id = str(args.get("chunk_id") or "").strip()
            if not chunk_id:
                text = "Error: chunk_id is required."
            else:
                found = pipeline.store.get_chunks([chunk_id])
                text = format_chunks(found)
        elif name == "recall_memory":
            query = str(args.get("query") or "").strip()
            hits = memory.recall(query, k=3)
            if not hits:
                text = "No related past tasks."
            else:
                text = "\n\n".join(
                    f"past_q: {h.question}\nsummary: {h.summary}\nscore={h.score:.3f}"
                    for h in hits
                )
        else:
            text = f"Unknown tool: {name}"

        guard_state.commit(name, args)
        if guard_state.enabled:
            text, removed = sanitize_untrusted(text)
            guard_state.injection_hits += removed
            guard_state.trusted_observations.append(text)
        guard_state.note_chunk_ids(text)
        return text

    return run
