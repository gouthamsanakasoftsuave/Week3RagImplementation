"""Embed chunks and store/search them in ChromaDB."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from rag.chunking import Chunk


@dataclass
class RetrievedChunk:
    content: str
    source: str
    chunk_id: str
    score: float
    metadata: dict


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
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def clear(self) -> None:
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> int:
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
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 4,
        source_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Similarity search (top-K). Lower cosine distance = better match."""
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
        ).tolist()

        kwargs: dict = {
            "query_embeddings": query_embedding,
            "n_results": top_k,
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
            # Cosine distance → similarity score in [0, 1]
            score = max(0.0, 1.0 - float(distance))
            meta = results["metadatas"][0][i] or {}
            retrieved.append(
                RetrievedChunk(
                    content=results["documents"][0][i],
                    source=meta.get("source", "unknown"),
                    chunk_id=chunk_id,
                    score=score,
                    metadata=meta,
                )
            )
        return retrieved

    @property
    def count(self) -> int:
        return self.collection.count()
