"""Streamlit UI: Ask My Documents (RAG)."""

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


def main() -> None:
    st.title("Ask My Documents")
    st.caption(
        "Retrieval-Augmented Generation — answers come only from your files, with sources."
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
            st.write("Files found:")
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
        st.header("Index")
        status = pipeline.status()
        st.metric("Indexed chunks", status["indexed_chunks"])
        st.caption(
            f"Chunk size {status['chunk_size']} · overlap {status['chunk_overlap']} · "
            f"top-K {status['top_k']} · min score {status['similarity_threshold']}"
        )

        if st.button("Rebuild index", type="primary", use_container_width=True):
            with st.spinner("Loading, chunking, and embedding documents..."):
                result = pipeline.ingest(rebuild=True)
            st.success(
                f"Indexed {result.chunks} chunks from {result.documents} document parts "
                f"({', '.join(result.sources) or 'none'})."
            )
            st.rerun()

        source_options = ["(all documents)"] + existing
        source_choice = st.selectbox("Metadata filter (source)", source_options)
        source_filter = None if source_choice == "(all documents)" else source_choice

    if pipeline.store.count == 0:
        st.info("No index yet. Click **Rebuild index** in the sidebar to ingest documents.")
        return

    question = st.text_input(
        "Your question",
        placeholder="e.g. How many paid sick days do employees get?",
    )
    ask = st.button("Ask", type="primary")

    if ask and question.strip():
        with st.spinner("Retrieving relevant chunks and generating answer..."):
            answer = pipeline.ask(question.strip(), source_filter=source_filter)

        st.subheader("Answer")
        st.write(answer.answer)

        if answer.grounded:
            st.success("Grounded in retrieved documents")
        else:
            st.warning("Model reported insufficient evidence (no guessing)")

        st.subheader("Retrieved evidence")
        if not answer.chunks_used:
            st.write("No chunks passed the similarity threshold.")
        else:
            for i, chunk in enumerate(answer.chunks_used, start=1):
                page = chunk.metadata.get("page") or ""
                label = f"{chunk.source}" + (f" · page {page}" if page else "")
                with st.expander(f"#{i} · {label} · score {chunk.score:.2f}", expanded=i == 1):
                    st.write(chunk.content)
                    st.caption(f"Chunk id: `{chunk.chunk_id}`")

        if answer.sources:
            st.markdown("**Sources:** " + ", ".join(f"`{s}`" for s in answer.sources))

    st.divider()
    with st.expander("Try these questions"):
        st.markdown(
            """
- How many paid sick days do employees get?
- What is the monthly internet stipend for remote work?
- Can unused annual leave carry over?
- What is the hotel limit in tier-1 cities?
- What is our customer refund policy for electronics?
  *(should say it doesn't know — not in these docs)*
            """
        )


if __name__ == "__main__":
    main()
