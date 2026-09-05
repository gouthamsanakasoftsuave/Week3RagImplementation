"""Streamlit UI: Ask Legal Contracts (RAG) + inspection / hybrid search."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag.pipeline import RagPipeline
from rag.ocr import ocr_enabled, tesseract_available
from rag.observability import flush_langfuse, tracing_enabled
from rag.agent_loop import run_agent
from rag.agent_memory import AgentMemory
from rag.agent_types import AgentRun
from rag.agent_workflow import run_workflow

load_dotenv()

st.set_page_config(
    page_title="Ask Legal Contracts",
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


def get_memory(pipeline: RagPipeline) -> AgentMemory:
    return AgentMemory(
        path=Path("eval") / "agent_memory.json",
        embedder=pipeline.store.embedding_model,
    )


def render_run(run: AgentRun) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Time (ms)", run.elapsed_ms)
    c2.metric("LLM calls", run.usage.llm_calls)
    c3.metric("Tokens", run.usage.total_tokens)
    c4.metric("Est. cost USD", f"{run.usage.cost_usd:.5f}")
    st.caption(
        f"Mode `{run.mode}` · stop `{run.stop_reason}` · "
        f"{'grounded' if run.grounded else 'refused / incomplete'}"
    )
    if run.memory_hits:
        st.info("Long-term memory used:\n" + "\n".join(f"- {h}" for h in run.memory_hits[:3]))

    st.subheader("Visible steps")
    for step in run.steps:
        with st.expander(step.title, expanded=step.kind in {"answer", "stop"}):
            st.markdown(f"`{step.kind}`")
            st.write(step.detail)

    st.subheader("Final answer")
    st.write(run.answer)
    if run.sources:
        st.markdown("**Sources:** " + ", ".join(f"`{s}`" for s in run.sources))


def main() -> None:
    st.title("Ask Legal Contracts")
    st.caption(
        "Ask questions over uploaded contracts. Answers are grounded in retrieved clauses "
        "with sources. Hybrid retrieval: BM25 + semantic + RRF."
    )
    st.warning(
        "This is **not legal advice**. The app only summarizes text from your uploaded "
        "contracts. Have a qualified lawyer review any real matter."
    )

    if not os.getenv("GROQ_API_KEY"):
        st.error("Missing GROQ_API_KEY. Add it to your `.env` file and restart.")
        st.stop()

    pipeline = get_pipeline()

    with st.sidebar:
        st.header("Contracts")
        st.write(f"Folder: `{DOCS_DIR.resolve()}`")
        existing = sorted(
            p.name
            for p in DOCS_DIR.glob("*")
            if p.suffix.lower() in {".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
        )
        if existing:
            for name in existing:
                st.markdown(f"- `{name}`")
        else:
            st.warning("No contracts found. Add PDF/TXT/MD files to `documents/`.")

        uploaded = st.file_uploader(
            "Upload more contracts",
            type=["pdf", "txt", "md", "png", "jpg", "jpeg", "webp", "tif", "tiff"],
            accept_multiple_files=True,
        )
        if uploaded:
            DOCS_DIR.mkdir(exist_ok=True)
            for f in uploaded:
                (DOCS_DIR / f.name).write_bytes(f.getbuffer())
            st.success(f"Saved {len(uploaded)} file(s). Click **Rebuild index**.")

        st.divider()
        st.header("OCR")
        if ocr_enabled() and tesseract_available():
            st.success("OCR ready (Tesseract found)")
        elif ocr_enabled():
            st.warning(
                "OCR enabled in `.env`, but Tesseract is not installed/found. "
                "Digital PDF text still works. See README for install steps."
            )
        else:
            st.info("OCR disabled (`ENABLE_OCR=false`)")

        st.divider()
        st.header("Langfuse")
        if tracing_enabled():
            st.success("Tracing on (keys in `.env`)")
            st.caption(
                f"Release `{os.getenv('LANGFUSE_RELEASE', 'week5-error-analysis')}` · "
                "each Ask is a full trace"
            )
        else:
            st.info("Off — add LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to `.env`")

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

        source_options = ["(all contracts)"] + existing
        source_choice = st.selectbox("Metadata filter (source)", source_options)
        source_filter = None if source_choice == "(all contracts)" else source_choice

        st.divider()
        st.header("Agent budgets (Week 7)")
        max_steps = st.slider("Max steps", 2, 12, int(os.getenv("AGENT_MAX_STEPS", "8")))
        max_tokens = st.number_input(
            "Max tokens", min_value=1000, max_value=50000, value=int(os.getenv("AGENT_MAX_TOKENS", "12000")), step=500
        )
        max_seconds = st.slider("Max seconds", 15, 180, int(os.getenv("AGENT_MAX_SECONDS", "90")))

    if pipeline.store.count == 0:
        st.info("No index yet. Click **Rebuild index** in the sidebar to ingest contracts.")
        return

    tab_rag, tab_agent, tab_wf, tab_race = st.tabs(
        ["RAG (one shot)", "Agent loop", "Fixed workflow", "Agent vs workflow race"]
    )

    sample_q = (
        "Who are the two parties to the Service Provider Agreement, "
        "and how much written notice does a party need to give to terminate?"
    )

    with tab_rag:
        st.caption("Single retrieve → generate pass (Weeks 3–6). No loop.")
        question = st.text_input(
            "Your question",
            placeholder="e.g. Who are the two parties to the Service Provider Agreement?",
            key="rag_q",
        )
        ask = st.button("Ask", type="primary", key="rag_ask")

        if ask and question.strip():
            q = question.strip()
            with st.spinner(f"Retrieving with **{mode}** and generating answer..."):
                answer = pipeline.ask(
                    q, top_k=top_k, source_filter=source_filter, mode=mode
                )
                chunks = answer.chunks_used
                if tracing_enabled():
                    flush_langfuse()

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

        with st.expander("Sample contract questions"):
            st.caption(
                "Week 5 set from Service-Provider-Agreement.pdf "
                "(copy into Your question, then Ask)."
            )
            st.markdown(
                """
1. Who are the two parties to the Service Provider Agreement, and what is each party called in the contract?
2. On what date and in which city was the Service Provider Agreement entered into?
3. What is the face value of the Rights Equity Shares and up to what amount is the Issue aggregating?
4. Who is the Lead Manager appointed for the Issue?
5. What professional fee will the Agency be paid for media monitoring under the commercial terms?
6. How much written notice does a party need to give to terminate this agreement?
7. Which country's law governs this agreement?
8. Which courts have exclusive jurisdiction over disputes under this agreement?
9. Within how many days must advertising bills be settled after the month in which ads were released?
10. If the Company asks the Agency to return Confidential Information, how soon must the Agency return it?
11. Can the Company terminate the agreement without notice if it thinks the Agency's services are deficient?
12. What is the maximum aggregate liability of the Agency under the indemnity clause?
13. How long does the confidentiality clause survive after expiry or early termination of the agreement?
14. If the parties have a dispute, what must they try first and when can they go to arbitration?
15. Who owns the creatives, advertisements, reports and other materials produced from the Agency's services?
16. Who is the contact person and email for notices to the Company?
17. What is the GDPR fine in the Data Processing Agreement attached to this contract?
18. What is Jordan Hale's base salary under this Service Provider Agreement?
19. Does this agreement require the Agency to maintain cyber insurance of $2 million?
20. Who must indemnify whom if there is a breach of a third party's intellectual property?
                """
            )
            st.caption("Measure retrieval with: `python eval_hit_rate.py --rebuild`")

    with tab_agent:
        st.caption(
            "Hand-built ReAct loop: think → tool → observe → repeat. Stops on finish, "
            "max steps, max tokens, or max time. Tools: list_contracts, search_contracts, "
            "read_chunk, recall_memory."
        )
        aq = st.text_area("Multi-step question", value=sample_q, key="agent_q", height=90)
        if st.button("Run agent", type="primary", key="agent_run"):
            mem = get_memory(pipeline)
            with st.spinner("Agent loop running..."):
                run = run_agent(
                    aq.strip(),
                    pipeline,
                    memory=mem,
                    max_steps=max_steps,
                    max_tokens=int(max_tokens),
                    max_seconds=float(max_seconds),
                )
            render_run(run)

    with tab_wf:
        st.caption(
            "Fixed sequence (no LLM choosing tools): list contracts → split the question "
            "with rules → search each part → one Groq answer. Same task, no agent loop."
        )
        wq = st.text_area("Multi-step question", value=sample_q, key="wf_q", height=90)
        if st.button("Run workflow", type="primary", key="wf_run"):
            mem = get_memory(pipeline)
            with st.spinner("Fixed workflow running..."):
                run = run_workflow(wq.strip(), pipeline, memory=mem)
            render_run(run)

    with tab_race:
        st.caption(
            "Track F: race the agent against the fixed workflow on 6 multi-hop contract "
            "tasks (speed, Groq tokens/cost, phrase reliability). Writes `eval/week7_race.json`."
        )
        st.warning(
            "This calls Groq twice per task (12 runs). Takes a few minutes. "
            "Rebuild the index first if chunks = 0."
        )
        if st.button("Run race", type="primary", key="race_run"):
            from rag.agent_race import run_race

            with st.spinner("Racing agent vs workflow..."):
                result = run_race(pipeline)
            s = result["summary"]
            left, right = st.columns(2)
            with left:
                st.markdown("**Fixed workflow**")
                st.metric("Reliability", f"{s['workflow']['reliability_pct']}%")
                st.metric("Avg time (ms)", s["workflow"]["avg_ms"])
                st.metric("Avg LLM calls", s["workflow"]["avg_llm_calls"])
                st.metric("Total tokens", s["workflow"]["total_tokens"])
                st.metric("Est. cost USD", f"{s['workflow']['total_cost_usd']:.5f}")
            with right:
                st.markdown("**Agent**")
                st.metric("Reliability", f"{s['agent']['reliability_pct']}%")
                st.metric("Avg time (ms)", s["agent"]["avg_ms"])
                st.metric("Avg LLM calls", s["agent"]["avg_llm_calls"])
                st.metric("Total tokens", s["agent"]["total_tokens"])
                st.metric("Est. cost USD", f"{s['agent']['total_cost_usd']:.5f}")
            st.success(s["ship"])
            st.dataframe(
                [
                    {
                        "id": r["id"],
                        "workflow_pass": r["workflow"]["pass"],
                        "workflow_ms": r["workflow"]["elapsed_ms"],
                        "workflow_tokens": r["workflow"]["total_tokens"],
                        "agent_pass": r["agent"]["pass"],
                        "agent_ms": r["agent"]["elapsed_ms"],
                        "agent_tokens": r["agent"]["total_tokens"],
                        "agent_stop": r["agent"]["stop_reason"],
                    }
                    for r in result["rows"]
                ]
            )
            with st.expander("Raw JSON"):
                st.json(result["summary"])

        existing_race = Path("eval") / "week7_race.json"
        if existing_race.exists():
            st.caption("Last saved race file is `eval/week7_race.json`.")


if __name__ == "__main__":
    main()
