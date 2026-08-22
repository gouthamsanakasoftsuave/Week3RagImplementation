"""Rewrite messy questions with the LLM before retrieval."""

from __future__ import annotations

import os

_REWRITE_SYSTEM = """Rewrite the user's question as a short search query over a legal contract.
Fix missing spaces and typos. Keep meaning. Under 20 words.
Return only the rewritten query, no quotes or explanation."""


def rewrite_query(question: str) -> str:
    """Turn the user question into a retrieval query via Groq. On failure, use the original."""
    q = (question or "").strip()
    if not q:
        return q
    try:
        rewritten = _llm_rewrite(q)
        if rewritten:
            return rewritten
    except Exception:
        pass
    return q


def _llm_rewrite(question: str) -> str:
    from groq import Groq

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    response = client.chat.completions.create(
        model=model_name,
        temperature=0.0,
        max_tokens=60,
        messages=[
            {"role": "system", "content": _REWRITE_SYSTEM},
            {"role": "user", "content": question},
        ],
    )
    text = (response.choices[0].message.content or "").strip()
    text = text.strip(" \"'")
    if not text or len(text) > 200:
        return ""
    return text
