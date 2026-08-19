"""Streamlit UI: Ask My Documents (RAG) + Week 4 inspection / hybrid search."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag.pipeline import RagPipeline

load_dotenv()

st.set_page_config(
    page_title="Ask My Documents",
    page_icon="📄",
    layout="wide",
)

DOCS_DIR = Path("documents")
DB_DIR = Path("chroma_db")


@st.cache_resource(show_spinner="Loading embedding model & vector store...")
def get_pipeline() -> RagPipeline:
    return RagPipeline(documents_dir=DOCS_DIR, persist_dir=DB_DIR)


def classify_failure(question: str, chunks, answer) -> str:
    """Heuristic label for mentor inspection (retrieval vs generation)."""
    if not chunks:
        return "Retrieval failure — no useful chunks fetched"
    joined = " ".join(c.content.lower() for c in chunks)
    key_terms = [t for t in question.lower().replace('"', " ").split() if len(t) > 4]
    overlap = sum(1 for t in key_terms if t in joined)
    if key_terms and overlap / max(len(key_terms), 1) < 0.25:
        return "Likely retrieval failure — fetched chunks look weakly related"
    if not answer.grounded:
        return "Retrieval may be weak OR evidence incomplete — model refused to answer"
    return "If answer is still wrong: likely generation failure (right-ish docs, bad answer)"


def main() -> None:
    st.title("Ask My Documents")
    st.caption(
        "Week 3–4 RAG — hybrid retrieval (BM25 + semantic + RRF), grounded answers, inspection view."
    )

    if not os.getenv("GROQ_API_KEY"):
        st.error("Missing GROQ_API_KEY. Add it to your `.env` file and restart.")
        st.stop()

    pipeline = get_pipeline()

    with st.sidebar:
        st.header("Documents")
        st.write(f"Folder: `{DOCS_DIR.resolve()}`")
        existing = sorted(
            p.name
            for p in DOCS_DIR.glob("*")
            if p.suffix.lower() in {".pdf", ".txt", ".md"}
        )
        if existing:
            for name in existing:
                st.markdown(f"- `{name}`")
        else:
            st.warning("No documents found. Add PDF/TXT/MD files to `documents/`.")

        uploaded = st.file_uploader(
            "Upload more documents",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        if uploaded:
            DOCS_DIR.mkdir(exist_ok=True)
            for f in uploaded:
                (DOCS_DIR / f.name).write_bytes(f.getbuffer())
            st.success(f"Saved {len(uploaded)} file(s). Click **Rebuild index**.")

        st.divider()
        st.header("Retrieval (Week 4)")
        mode = st.radio(
            "Search mode",
            options=["hybrid", "semantic", "bm25"],
            index=0,
            help="hybrid = BM25 keyword + semantic vectors fused with RRF (one Week-4 improvement)",
        )
        top_k = st.slider("Top-K", min_value=3, max_value=8, value=max(pipeline.top_k, 4))

        st.divider()
        st.header("Index")
        status = pipeline.status()
        st.metric("Indexed chunks", status["indexed_chunks"])
        st.caption(
            f"Chunk {status['chunk_size']}/{status['chunk_overlap']} · "
            f"BM25 docs {status['bm25_docs']} · min semantic score {status['similarity_threshold']}"
        )

        if st.button("Rebuild index", type="primary", use_container_width=True):
            with st.spinner("Loading, chunking, and embedding documents..."):
                result = pipeline.ingest(rebuild=True)
            st.success(
                f"Indexed {result.chunks} chunks from {result.documents} parts "
                f"({', '.join(result.sources) or 'none'})."
            )
            st.cache_resource.clear()
            st.rerun()

        source_options = ["(all documents)"] + existing
        source_choice = st.selectbox("Metadata filter (source)", source_options)
        source_filter = None if source_choice == "(all documents)" else source_choice

    if pipeline.store.count == 0:
        st.info("No index yet. Click **Rebuild index** in the sidebar to ingest documents.")
        return

    question = st.text_input(
        "Your question",
        placeholder='e.g. List numbered items under "Health and Safety, Security, Fire"',
    )
    ask = st.button("Ask", type="primary")

    if ask and question.strip():
        q = question.strip()
        with st.spinner(f"Retrieving with **{mode}** and generating answer..."):
            chunks = pipeline.retrieve(
                q, top_k=top_k, source_filter=source_filter, mode=mode
            )
            answer = pipeline.ask(
                q, top_k=top_k, source_filter=source_filter, mode=mode
            )

        st.subheader("Inspection view (Week 4)")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Question**")
            st.write(q)
            st.caption(f"Mode: `{mode}` · top-K: {top_k}")
        with c2:
            st.markdown("**Fetched chunks**")
            if not chunks:
                st.write("_(none)_")
            else:
                for i, chunk in enumerate(chunks, start=1):
                    page = chunk.metadata.get("page") or "?"
                    st.markdown(
                        f"`#{i}` p{page} · score {chunk.score:.4f} · "
                        f"{chunk.metadata.get('retrieval', mode)}"
                    )
        with c3:
            st.markdown("**Final answer**")
            st.write(answer.answer)

        st.info(classify_failure(q, chunks, answer))

        if answer.grounded:
            st.success("Grounded in retrieved documents")
        else:
            st.warning("Model reported insufficient evidence (no guessing)")

        st.subheader("Retrieved evidence")
        if not answer.chunks_used:
            st.write("No chunks returned.")
        else:
            for i, chunk in enumerate(answer.chunks_used, start=1):
                page = chunk.metadata.get("page") or ""
                label = f"{chunk.source}" + (f" · page {page}" if page else "")
                with st.expander(
                    f"#{i} · {label} · score {chunk.score:.4f}",
                    expanded=i == 1,
                ):
                    st.write(chunk.content or "_(empty chunk)_")
                    st.caption(f"Chunk id: `{chunk.chunk_id}`")

        if answer.sources:
            st.markdown("**Sources:** " + ", ".join(f"`{s}`" for s in answer.sources))

    st.divider()
    with st.expander("Week 4 demo questions (HR Policy)"):
        st.markdown(
            """
- List the numbered items under the heading Health and Safety, Security, Fire in the new employee induction checklist
- what are items that are available in "Health and Safety, Security, Fire"
- Where is the location of fire-fighting equipment covered in induction?
- What does the induction checklist say about keys, passes and ID Badges?
- What is the purpose of the Induction checklist for new employees?
            """
        )
        st.caption("Measure before/after with: `python eval_hit_rate.py --rebuild`")


if __name__ == "__main__":
    main()
