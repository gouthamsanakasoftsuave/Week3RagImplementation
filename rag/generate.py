"""Grounded answer generation with Groq + citations."""

from __future__ import annotations

import os
from dataclasses import dataclass

from groq import Groq

from rag.observability import get_langfuse, tracing_enabled
from rag.store import RetrievedChunk

SYSTEM_PROMPT = """You are a careful legal document assistant. You summarize uploaded contracts.
You are not a lawyer and you do not give legal advice.

Rules:
1. Answer ONLY using the provided context excerpts.
2. If the context does not contain enough information, reply exactly:
   I don't know based on the provided documents.
3. Do not invent clauses, parties, dates, dollar amounts, or obligations.
4. Quottions may have typos or missing spaces. If the user asks for a field or identifier
   and e or paraphrase the relevant clause text. Be concise and factual.
5. Questhat value appears in the excerpts, quote it exactly.
6. At the end of your answer, add a Sources line listing the document names you used
   (e.. Sources: nda.pdf, msa.pdf). Include page numbers when the excerpts provide them.
"""


@dataclass
class RagAnswer:
    answer: str
    sources: list[str]
    chunks_used: list[RetrievedChunk]
    grounded: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0


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
            prompt_tokens=0,
            completion_tokens=0,
            llm_calls=0,
        )

    client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
    model_name = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    context = build_context(chunks)

    user_prompt = (
        f"Context excerpts:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    def _call() -> tuple[str, dict[str, int] | None]:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.1,
            messages=messages,
        )
        text = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        details = None
        if usage is not None:
            details = {
                "input": int(getattr(usage, "prompt_tokens", 0) or 0),
                "output": int(getattr(usage, "completion_tokens", 0) or 0),
                "total": int(getattr(usage, "total_tokens", 0) or 0),
            }
        return text, details

    if tracing_enabled():
        lf = get_langfuse()
        with lf.start_as_current_observation(
            as_type="generation",
            name="groq-generate",
            model=model_name,
            input={"question": question, "messages": messages},
            model_parameters={"temperature": 0.1},
        ) as gen:
            answer, usage_details = _call()
            update: dict = {"output": answer}
            if usage_details:
                update["usage_details"] = usage_details
            gen.update(**update)
    else:
        answer, usage_details = _call()

    sources = sorted({c.source for c in chunks})
    grounded = "i don't know based on the provided documents" not in answer.lower()
    usage_details = usage_details or {}

    return RagAnswer(
        answer=answer,
        sources=sources,
        chunks_used=chunks,
        grounded=grounded,
        prompt_tokens=int(usage_details.get("input", 0) or 0),
        completion_tokens=int(usage_details.get("output", 0) or 0),
        llm_calls=1,
    )
