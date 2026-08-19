"""Embed chunks and store/search them in ChromaDB (+ optional hybrid BM25/RRF)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from rag.chunking import Chunk
from rag.hybrid import (
    BM25Index,
    RankedHit,
    extract_focus_queries,
    reciprocal_rank_fusion,
)


@dataclass
class RetrievedChunk:
    content: str
    source: str
    chunk_id: str
    score: float
    metadata: dict


def _from_hit(hit: RankedHit) -> RetrievedChunk:
    return RetrievedChunk(
        content=hit.content,
        source=hit.source,
        chunk_id=hit.chunk_id,
        score=hit.score,
        metadata=hit.metadata,
    )


class VectorStore:
    """Dense retrieval with a bi-encoder embedding model + Chroma HNSW index."""

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_dir: str | Path = "chroma_db",
        collection_name: str = "documents",
    ) -> None:
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.persist_dir = str(persist_dir)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.bm25 = BM25Index()
        self._refresh_bm25()

    def clear(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.bm25.rebuild([], [], [])

    def _refresh_bm25(self) -> None:
        if self.collection.count() == 0:
            self.bm25.rebuild([], [], [])
            return
        raw = self.collection.get(include=["documents", "metadatas"])
        ids = raw.get("ids") or []
        docs = raw.get("documents") or []
        metas = raw.get("metadatas") or []
        self.bm25.rebuild(ids, docs, metas)

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 32) -> int:
        if not chunks:
            return 0

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [c.content for c in batch]
            embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=False,
                normalize_embeddings=True,
            ).tolist()

            self.collection.add(
                ids=[c.chunk_id for c in batch],
                documents=texts,
                embeddings=embeddings,
                metadatas=[
                    {
                        "source": c.source,
                        "chunk_id": c.chunk_id,
                        "page": str(c.metadata.get("page", "")),
                        "type": str(c.metadata.get("type", "")),
                    }
                    for c in batch
                ],
            )
        self._refresh_bm25()
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 4,
        source_filter: str | None = None,
        mode: str = "semantic",
        candidate_k: int = 20,
    ) -> list[RetrievedChunk]:
        mode = (mode or "semantic").lower()
        if mode == "hybrid":
            return self.search_hybrid(
                query,
                top_k=top_k,
                source_filter=source_filter,
                candidate_k=candidate_k,
            )
        if mode == "bm25":
            hits = self.bm25.search(query, top_k=top_k * 3)
            if source_filter:
                hits = [h for h in hits if h.source == source_filter]
            return [_from_hit(h) for h in hits[:top_k]]
        return self.search_semantic(query, top_k=top_k, source_filter=source_filter)

    def search_semantic(
        self,
        query: str,
        top_k: int = 4,
        source_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
        ).tolist()

        kwargs: dict = {
            "query_embeddings": query_embedding,
            "n_results": min(top_k, self.collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if source_filter:
            kwargs["where"] = {"source": source_filter}

        results = self.collection.query(**kwargs)
        retrieved: list[RetrievedChunk] = []
        if not results["ids"] or not results["ids"][0]:
            return retrieved

        for i, chunk_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            score = max(0.0, 1.0 - float(distance))
            meta = results["metadatas"][0][i] or {}
            retrieved.append(
                RetrievedChunk(
                    content=results["documents"][0][i],
                    source=meta.get("source", "unknown"),
                    chunk_id=chunk_id,
                    score=score,
                    metadata={**meta, "retrieval": "semantic"},
                )
            )
        return retrieved

    def search_hybrid(
        self,
        query: str,
        top_k: int = 4,
        source_filter: str | None = None,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ) -> list[RetrievedChunk]:
        fetch = max(candidate_k, top_k)
        semantic = self.search_semantic(query, top_k=fetch, source_filter=source_filter)
        keyword_hits = self.bm25.search(query, top_k=fetch)
        if source_filter:
            keyword_hits = [h for h in keyword_hits if h.source == source_filter]

        semantic_hits = [
            RankedHit(
                chunk_id=c.chunk_id,
                content=c.content,
                source=c.source,
                score=c.score,
                metadata=c.metadata,
            )
            for c in semantic
        ]

        # Extra BM25 passes on quoted / heading phrases (still one hybrid change)
        focus = extract_focus_queries(query)
        ranked_lists: list[list[RankedHit]] = [semantic_hits, keyword_hits]
        weights: list[float] = [1.0, 1.2]
        for phrase in focus:
            phrase_hits = self.bm25.search(phrase, top_k=fetch)
            if source_filter:
                phrase_hits = [h for h in phrase_hits if h.source == source_filter]
            ranked_lists.append(phrase_hits)
            weights.append(2.5)

        fused = reciprocal_rank_fusion(
            ranked_lists,
            weights=weights,
            rrf_k=rrf_k,
            top_k=top_k,
            phrase_boosts=focus,
        )
        return [_from_hit(h) for h in fused]

    @property
    def count(self) -> int:
        return self.collection.count()
