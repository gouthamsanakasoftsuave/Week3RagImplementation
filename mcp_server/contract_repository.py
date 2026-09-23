"""Contract-repository MCP server (Week 9, Track F).

This process offers tools. It does not call a model.
Add another tool by writing a new function with @mcp.tool below.
Do not put that tool's name into rag/mcp_agent.py. The host discovers tools at runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastmcp import FastMCP

from rag.agent_tools import format_chunks
from rag.pipeline import RagPipeline

mcp = FastMCP("contract-repository")

_pipeline: RagPipeline | None = None


def bind_pipeline(pipeline: RagPipeline) -> None:
    """Let the host reuse the pipeline it already loaded. The stdio server builds its own."""
    global _pipeline
    _pipeline = pipeline


def _pipe() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline(
            documents_dir=ROOT / "documents",
            persist_dir=ROOT / "chroma_db",
        )
    return _pipeline


@mcp.tool
def list_contracts() -> str:
    """List the contract filenames in the repository. Read-only."""
    try:
        names = _pipe().store.list_sources()
        if not names:
            return "No contracts are indexed."
        return "\n".join(names)
    except Exception as exc:
        return f"Recoverable error: {exc}"


@mcp.tool
def search_contracts(query: str, source: str = "") -> str:
    """Search contract text for one fact. Read-only. query is a short phrase. source is an optional filename."""
    try:
        query = (query or "").strip()
        if not query:
            return "Recoverable error: query is required."
        chunks = _pipe().retrieve(
            query,
            top_k=4,
            source_filter=source.strip() or None,
            mode="hybrid",
            rewrite=True,
        )
        return format_chunks(chunks)
    except Exception as exc:
        return f"Recoverable error: {exc}"


@mcp.tool
def repository_status() -> str:
    """Report how many chunks are indexed and which retrieval mode is active. Read-only."""
    try:
        status = _pipe().status()
        return (
            f"indexed_chunks={status['indexed_chunks']}\n"
            f"retrieval_mode={status['retrieval_mode']}\n"
            f"documents_dir={status['documents_dir']}"
        )
    except Exception as exc:
        return f"Recoverable error: {exc}"


if __name__ == "__main__":
    mcp.run()
