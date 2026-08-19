"""Measure hit-rate@k for semantic vs hybrid retrieval (Week 4)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.pipeline import RagPipeline


def is_hit(chunks, item, k: int) -> bool:
    top = chunks[:k]
    expected_page = str(item.get("expected_page", "")).strip()
    expected_source = item.get("expected_source")
    phrases = [p.lower() for p in item.get("expected_phrases", [])]

    for chunk in top:
        if expected_source and chunk.source != expected_source:
            continue
        page = str(chunk.metadata.get("page", "")).strip()
        content = (chunk.content or "").lower()
        page_ok = (not expected_page) or page == expected_page
        phrase_ok = (not phrases) or any(p in content for p in phrases)
        if page_ok and phrase_ok:
            return True
        # Phrase-only fallback if page metadata missing but text matches
        if phrase_ok and any(p in content for p in phrases):
            if not expected_page or page == expected_page:
                return True
    return False


def evaluate(pipeline: RagPipeline, items: list[dict], mode: str, k: int) -> dict:
    hits = 0
    rows = []
    for item in items:
        chunks = pipeline.retrieve(
            item["question"],
            top_k=k,
            source_filter=item.get("expected_source"),
            mode=mode,
        )
        hit = is_hit(chunks, item, k)
        hits += int(hit)
        rows.append(
            {
                "id": item["id"],
                "hit": hit,
                "pages": [c.metadata.get("page") for c in chunks[:k]],
                "chunk_ids": [c.chunk_id for c in chunks[:k]],
            }
        )
    total = len(items) or 1
    return {
        "mode": mode,
        "k": k,
        "hit_rate": hits / total,
        "hits": hits,
        "total": len(items),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 4 hit-rate@k evaluation")
    parser.add_argument("--eval", default="eval/hr_eval.json")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    pipeline = RagPipeline()
    if args.rebuild or pipeline.store.count == 0:
        print("Ingesting documents...")
        result = pipeline.ingest(rebuild=True)
        print(f"Indexed {result.chunks} chunks from {result.sources}")

    items = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    before = evaluate(pipeline, items, mode="semantic", k=args.k)
    after = evaluate(pipeline, items, mode="hybrid", k=args.k)

    print("\n=== Week 4 · HR Policy · hit-rate@{} ===".format(args.k))
    print(
        f"BEFORE (semantic only): {before['hits']}/{before['total']} "
        f"= {before['hit_rate']:.2%}"
    )
    print(
        f"AFTER  (hybrid BM25+RRF): {after['hits']}/{after['total']} "
        f"= {after['hit_rate']:.2%}"
    )
    print("\nPer-question:")
    for b, a in zip(before["rows"], after["rows"]):
        print(
            f"- {b['id']}: semantic={'HIT' if b['hit'] else 'MISS'} "
            f"pages={b['pages']} | hybrid={'HIT' if a['hit'] else 'MISS'} "
            f"pages={a['pages']}"
        )

    out = {"before": before, "after": after}
    Path("eval/last_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nSaved eval/last_results.json")


if __name__ == "__main__":
    main()
