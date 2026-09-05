"""Shared types for the Week 7 contract agent and fixed workflow."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    """Rough USD from env rates (defaults = Groq gpt-oss-20b public list)."""
    in_rate = float(os.getenv("GROQ_INPUT_USD_PER_MILLION", "0.075"))
    out_rate = float(os.getenv("GROQ_OUTPUT_USD_PER_MILLION", "0.30"))
    return (prompt_tokens / 1_000_000) * in_rate + (completion_tokens / 1_000_000) * out_rate


@dataclass
class UsageMeter:
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, prompt: int = 0, completion: int = 0, calls: int = 1) -> None:
        self.llm_calls += calls
        self.prompt_tokens += int(prompt or 0)
        self.completion_tokens += int(completion or 0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        return estimate_cost_usd(self.prompt_tokens, self.completion_tokens)


@dataclass
class AgentStep:
    index: int
    kind: str  # thought | tool | observe | answer | stop
    title: str
    detail: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentRun:
    question: str
    answer: str
    mode: str  # agent | workflow
    steps: list[AgentStep]
    sources: list[str]
    usage: UsageMeter
    elapsed_ms: int
    stop_reason: str
    grounded: bool
    memory_hits: list[str] = field(default_factory=list)
