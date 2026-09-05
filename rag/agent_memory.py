"""Short-term scratchpad + long-term summarised memory (file + embeddings)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass
class MemoryHit:
    question: str
    summary: str
    score: float


class AgentMemory:
    """In-run notes plus a small JSON log of past task summaries."""

    def __init__(
        self,
        path: str | Path = "eval/agent_memory.json",
        embedder: Any | None = None,
        max_episodes: int = 40,
    ) -> None:
        self.path = Path(path)
        self.embedder = embedder
        self.max_episodes = max_episodes
        self.scratchpad: list[str] = []
        self._episodes: list[dict] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._episodes = []
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._episodes = list(data.get("episodes") or [])
        except Exception:
            self._episodes = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"episodes": self._episodes[-self.max_episodes :]}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def note(self, text: str) -> None:
        text = (text or "").strip()
        if text:
            self.scratchpad.append(text)

    def recall(self, query: str, k: int = 3) -> list[MemoryHit]:
        if not self._episodes or not query.strip():
            return []
        if self.embedder is None:
            q = query.lower()
            scored: list[MemoryHit] = []
            for ep in self._episodes:
                blob = f"{ep.get('question', '')} {ep.get('summary', '')}".lower()
                overlap = sum(1 for t in q.split() if len(t) > 4 and t in blob)
                if overlap:
                    scored.append(
                        MemoryHit(
                            question=str(ep.get("question", "")),
                            summary=str(ep.get("summary", "")),
                            score=float(overlap),
                        )
                    )
            scored.sort(key=lambda h: h.score, reverse=True)
            return scored[:k]

        texts = [
            f"{ep.get('question', '')}\n{ep.get('summary', '')}" for ep in self._episodes
        ]
        q_vec = self.embedder.encode([query], normalize_embeddings=True)
        e_vec = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        sims = (e_vec @ q_vec[0]).tolist()
        ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
        hits: list[MemoryHit] = []
        for idx, score in ranked[:k]:
            if float(score) < 0.35:
                continue
            ep = self._episodes[idx]
            hits.append(
                MemoryHit(
                    question=str(ep.get("question", "")),
                    summary=str(ep.get("summary", "")),
                    score=float(score),
                )
            )
        return hits

    def remember_task(self, question: str, answer: str, sources: list[str]) -> str:
        summary = (answer or "").strip().replace("\n", " ")
        if len(summary) > 400:
            summary = summary[:397] + "..."
        episode = {
            "question": question,
            "summary": summary,
            "sources": sources,
            "ts": int(time.time()),
        }
        self._episodes.append(episode)
        self._save()
        return summary
