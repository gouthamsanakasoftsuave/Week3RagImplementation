"""BM25 keyword search + Reciprocal Rank Fusion (RRF) for hybrid retrieval."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.I)
_QUOTED_RE = re.compile(r'"([^"]+)"|“([^”]+)”')
_HEADING_RE = re.compile(
    r"under the heading\s+(.+?)(?:\s+in\s+the\s+|\s+in\s+|\s*$)",
    re.I,
)
_KNOWN_HEADINGS = [
    "Health and Safety, Security, Fire",
]


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def extract_focus_queries(query: str) -> list[str]:
    """Pull exact headings/phrases out of a messy user question for BM25."""
    found: list[str] = []
    for m in _QUOTED_RE.finditer(query or ""):
        found.append(next(g for g in m.groups() if g))
    for m in _HEADING_RE.finditer(query or ""):
        found.append(m.group(1).strip(" \"'"))
    lower_q = (query or "").lower()
    for heading in _KNOWN_HEADINGS:
        if heading.lower() in lower_q:
            found.append(heading)
    out: list[str] = []
    seen: set[str] = set()
    for item in found:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


@dataclass
class RankedHit:
    chunk_id: str
    content: str
    source: str
    score: float
    metadata: dict


class BM25Index:
    """In-memory BM25 over the same chunks stored in Chroma."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metas: list[dict] = []
        self._bm25: BM25Okapi | None = None

    def rebuild(self, ids: list[str], texts: list[str], metas: list[dict]) -> None:
        self._ids = list(ids)
        self._texts = list(texts)
        self._metas = list(metas)
        tokenized = [tokenize(t) for t in self._texts]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    @property
    def size(self) -> int:
        return len(self._ids)

    def search(self, query: str, top_k: int = 10) -> list[RankedHit]:
        if not self._bm25 or not self._ids:
            return []

        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results: list[RankedHit] = []
        for i in ranked[:top_k]:
            if scores[i] <= 0:
                continue
            meta = self._metas[i] or {}
            results.append(
                RankedHit(
                    chunk_id=self._ids[i],
                    content=self._texts[i],
                    source=meta.get("source", "unknown"),
                    score=float(scores[i]),
                    metadata={**meta, "retrieval": "bm25"},
                )
            )
        return results


def reciprocal_rank_fusion(
    ranked_lists: list[list[RankedHit]],
    *,
    weights: list[float] | None = None,
    rrf_k: int = 60,
    top_k: int = 4,
    phrase_boosts: list[str] | None = None,
) -> list[RankedHit]:
    """Fuse ranked lists: score(d) = Σ w_i / (rrf_k + rank_i(d))."""
    fused: dict[str, float] = defaultdict(float)
    best: dict[str, RankedHit] = {}
    weights = weights or [1.0] * len(ranked_lists)

    for list_idx, results in enumerate(ranked_lists):
        weight = weights[list_idx] if list_idx < len(weights) else 1.0
        for rank, hit in enumerate(results, start=1):
            fused[hit.chunk_id] += weight / (rrf_k + rank)
            prev = best.get(hit.chunk_id)
            if prev is None or len(hit.content) > len(prev.content):
                best[hit.chunk_id] = hit

    for chunk_id, hit in best.items():
        content_l = (hit.content or "").lower()
        for phrase in phrase_boosts or []:
            if phrase.lower() in content_l:
                fused[chunk_id] += 0.08

    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    out: list[RankedHit] = []
    for chunk_id, score in ordered[:top_k]:
        base = best[chunk_id]
        out.append(
            RankedHit(
                chunk_id=base.chunk_id,
                content=base.content,
                source=base.source,
                score=float(score),
                metadata={**base.metadata, "retrieval": "hybrid_rrf"},
            )
        )
    return out
