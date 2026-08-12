"""Grounded answer generation with Groq + citations."""

from __future__ import annotations

import os
from dataclasses import dataclass

from groq import Groq

from rag.store import RetrievedChunk

SYSTEM_PROMPT = """You are a careful document assistant.

Rules:
1. Answer ONLY using the provided context excerpts.
2. If the context does not contain enough information, reply exactly:
   I don't know based on the provided documents.
3. Do not invent policies, numbers, names, or procedures.
4. Be concise and factual.
5. At the end of your answer, add a Sources line listing the document names you used
   (e.g. Sources: leave_policy.txt, remote_work.md).
"""


@dataclass
class RagAnswer:
    answer: str
    sources: list[str]
    chunks_used: list[RetrievedChunk]
    grounded: bool


def build_context(chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        page = chunk.metadata.get("page") or ""
        page_note = f", page {page}" if page else ""
        parts.append(
            f"[Excerpt {i} | {chunk.source}{page_note} | relevance={chunk.score:.2f}]\n"
            f"{chunk.content}"
        )
    return "\n\n".join(parts)


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> RagAnswer:
    """Generate a grounded answer from retrieved chunks via Groq."""
    if not chunks:
        return RagAnswer(
            answer="I don't know based on the provided documents.",
            sources=[],
            chunks_used=[],
            grounded=False,
        )

    client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
    model_name = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    context = build_context(chunks)

    user_prompt = (
        f"Context excerpts:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = (response.choices[0].message.content or "").strip()
    sources = sorted({c.source for c in chunks})
    grounded = "i don't know based on the provided documents" not in answer.lower()

    return RagAnswer(
        answer=answer,
        sources=sources,
        chunks_used=chunks,
        grounded=grounded,
    )
