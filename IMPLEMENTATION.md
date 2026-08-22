# Ask Legal Contracts — Implementation Overview

This document summarizes the RAG app after it was retargeted from HR Policy to **Legal Contracts**.

**Topic focus:** Legal Contracts  
**Repo:** Week3RagImplementation  

---

## 1. What the app does

A mini **“Ask Legal Contracts”** RAG app that:

1. Loads contract documents (PDF / TXT / MD / images)
2. Splits them into chunks
3. Embeds and stores them in a vector database
4. Retrieves relevant clauses for a user question
5. Generates an answer **only from those chunks** (with sources)
6. Says **“I don’t know…”** when evidence is missing
7. Shows a **not legal advice** disclaimer in the UI

---

## 2. Current end-to-end flow

```text
Contracts (documents/)
    │
    ▼
Load text
  • pypdf for digital PDF text
  • OCR fallback (Tesseract) if page text is empty/short
  • Direct OCR for image files
    │
    ▼
Chunking
  • RecursiveCharacterTextSplitter
  • size = 500, overlap = 100
    │
    ▼
Embeddings + Index
  • SentenceTransformer: all-MiniLM-L6-v2
  • ChromaDB (HNSW, cosine)
  • BM25 keyword index (same chunks)
    │
    ▼
User question (Streamlit UI)
    │
    ▼
Retrieval mode (sidebar)
  • semantic  → vector similarity only
  • bm25      → keyword search only
  • hybrid    → BM25 + semantic + RRF (+ clause heading boost)  ← default
    │
    ▼
Top-K chunks (+ optional source filter)
    │
    ▼
Grounded generation (Groq LLM)
  • Answer only from retrieved context
  • Do not invent clauses, parties, dates, or obligations
  • Cite sources
  • Refuse if not in documents
    │
    ▼
Inspection view
  • Question | Fetched chunks | Final answer
```

---

## 3. Models & tools in use

| Layer | Choice | Notes |
|--------|--------|--------|
| **LLM (generation)** | Groq `openai/gpt-oss-20b` | Via `GROQ_API_KEY`; temperature `0.1` |
| **Embedding model** | `sentence-transformers/all-MiniLM-L6-v2` | Local bi-encoder |
| **Vector DB** | ChromaDB | Persistent folder `chroma_db/`, HNSW + cosine |
| **Keyword search** | BM25 (`rank-bm25`) | Exact terms / clause headings |
| **Fusion** | Reciprocal Rank Fusion (RRF) | Combines semantic + BM25 rankings |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` | 500 / 100 |
| **PDF text** | `pypdf` | Digital text layer |
| **PDF render for OCR** | PyMuPDF (`fitz`) | Page → image |
| **OCR engine** | Tesseract 5.x + `pytesseract` | Scanned / image pages |
| **UI** | Streamlit | Ask + inspect + rebuild index |

---

## 4. Techniques implemented

### Core RAG
- Document loading (PDF / TXT / MD)
- Chunking with overlap
- Dense embeddings + similarity search (top-K)
- Metadata (source, page)
- Grounded generation with citations
- “I don’t know” when context is insufficient
- Source filter in UI

### Retrieval debugging
- **Retrieval vs generation failure** labeling (inspection hint)
- **Inspection view** (question / fetched chunks / answer)
- **Keyword search (BM25)**
- **Hybrid search** (semantic + BM25 + RRF)
- Phrase / heading boost for common contract clause titles
- **hit-rate@3** evaluation (`eval_hit_rate.py`)

### OCR
- If digital text &lt; `OCR_MIN_CHARS` → OCR that page
- Image uploads (`.png`, `.jpg`, …) OCR’d into the index
- Graceful fallback if Tesseract is missing (text-only still works)

---

## 5. Key configuration (`.env`)

| Variable | Typical value | Meaning |
|----------|---------------|---------|
| `GROQ_API_KEY` | *(secret)* | Groq API access |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Answer model |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Vectors |
| `CHUNK_SIZE` | `500` | Chunk length |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K` | `4` | Chunks sent to the LLM |
| `SIMILARITY_THRESHOLD` | `0.35` | Filter for semantic mode |
| `RETRIEVAL_MODE` | `hybrid` | Default search mode |
| `ENABLE_OCR` | `true` | Turn OCR on/off |
| `OCR_MIN_CHARS` | `40` | Trigger OCR if text shorter than this |
| `TESSERACT_CMD` | `C:\Program Files\Tesseract-OCR\tesseract.exe` | Tesseract path |

---

## 6. Project structure

```text
Week3RagImplementation/
├── app.py                 # Streamlit UI (Ask Legal Contracts)
├── eval_hit_rate.py       # hit-rate@3 before/after script
├── requirements.txt
├── .env / .env.example
├── README.md
├── IMPLEMENTATION.md      # This file
├── documents/             # Contracts to index (sample .txt files included)
├── eval/
│   ├── contracts_eval.json
│   └── last_results.json
└── rag/
    ├── loader.py
    ├── ocr.py
    ├── chunking.py
    ├── store.py
    ├── hybrid.py          # BM25 + RRF + clause heading boost
    ├── generate.py        # Groq grounded contract answers
    └── pipeline.py
```

---

## 7. Sample corpus and eval

Included samples (fictional): `nda.txt`, `msa.txt`, `employment_agreement.txt`.

**Metric:** hit-rate@3 — “Did a chunk from the expected contract with the expected phrases appear in the top 3?”

Measured on the sample `.txt` contracts:

| Mode | Result |
|------|--------|
| BEFORE — semantic only | **80%** (4/5) |
| AFTER — hybrid BM25+RRF | **100%** (5/5) |

The miss on semantic-only was `nda_governing_law` (Governing Law of the NDA); hybrid recovered it.

Re-measure after ingest:

```powershell
python eval_hit_rate.py --k 3 --rebuild
```

If you replace the samples with your own PDFs, update `eval/contracts_eval.json` (`expected_source`, `expected_page`, `expected_phrases`).

---

## 8. Known limitations

- Long clauses can be **split across chunks**
- Pure paraphrase questions can still confuse BM25 / sometimes hybrid
- OCR is slower and can misread noisy scans
- Image-only pages need Tesseract installed locally
- `.env`, `.venv/`, and `chroma_db/` are local (not committed)
- The model is **not a lawyer**; do not treat answers as legal advice

---

## 9. How to run

```powershell
cd C:\Users\SanakaGoutham.N\Downloads\Week3RagImplementation
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

1. Open http://localhost:8501  
2. Confirm sidebar: contracts listed  
3. Click **Rebuild index**  
4. Choose mode: `hybrid` / `semantic` / `bm25`  
5. Ask a clause question and use the **Inspection view**

---

## 10. One-line summary

> We built a Legal Contracts RAG app: ingest contracts, retrieve with hybrid BM25 + semantic + RRF (clause heading boosts), generate grounded answers with citations, and measure hit-rate@3 on sample contract questions.
