# Ask My Documents — Implementation Overview

This document summarizes everything implemented so far for **Week 3** and **Week 4** (Module 2 — Retrieval & RAG), plus the later **OCR** addition.

**Topic focus:** HR Policy (Topic C)  
**Repo:** Week3RagImplementation  

---

## 1. What the app does

A mini **“Ask My Documents”** RAG app that:

1. Loads HR policy documents (PDF / TXT / MD / images)
2. Splits them into chunks
3. Embeds and stores them in a vector database
4. Retrieves relevant chunks for a user question
5. Generates an answer **only from those chunks** (with sources)
6. Says **“I don’t know…”** when evidence is missing

---

## 2. Current end-to-end flow

```text
Documents (documents/)
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
  • hybrid    → BM25 + semantic + RRF (+ phrase boost)  ← default
    │
    ▼
Top-K chunks (+ optional source filter)
    │
    ▼
Grounded generation (Groq LLM)
  • Answer only from retrieved context
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
| **Keyword search** | BM25 (`rank-bm25`) | Exact terms / headings |
| **Fusion** | Reciprocal Rank Fusion (RRF) | Combines semantic + BM25 rankings |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` | 500 / 100 |
| **PDF text** | `pypdf` | Digital text layer |
| **PDF render for OCR** | PyMuPDF (`fitz`) | Page → image |
| **OCR engine** | Tesseract 5.x + `pytesseract` | Scanned / image pages |
| **UI** | Streamlit | Ask + inspect + rebuild index |

---

## 4. Techniques implemented

### Week 3 — Core RAG
- Document loading (PDF / TXT / MD)
- Chunking with overlap
- Dense embeddings + similarity search (top-K)
- Metadata (source, page)
- Grounded generation with citations
- “I don’t know” when context is insufficient
- Source filter in UI

### Week 4 — Debugging retrieval
- **Retrieval vs generation failure** labeling (inspection hint)
- **Inspection view** (question / fetched chunks / answer)
- **Keyword search (BM25)**
- **Hybrid search** (semantic + BM25 + RRF)
- Phrase / heading boost for quoted section titles
- **hit-rate@3** evaluation (`eval_hit_rate.py`)
- Before/after measurement on HR questions

### OCR addition (after Week 4)
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
RagImplementation/
├── app.py                 # Streamlit UI (ask, inspect, modes, OCR status)
├── eval_hit_rate.py       # Week 4 hit-rate@3 before/after script
├── requirements.txt
├── .env / .env.example
├── README.md
├── IMPLEMENTATION.md      # This file
├── documents/             # HR PDFs / text / images to index
├── eval/
│   ├── hr_eval.json       # Eval questions (expected page/phrases)
│   └── last_results.json  # Last measured hit-rate results
└── rag/
    ├── loader.py          # Load docs + OCR fallback
    ├── ocr.py             # Tesseract / PyMuPDF helpers
    ├── chunking.py        # Split into overlapping chunks
    ├── store.py           # Chroma + semantic / bm25 / hybrid search
    ├── hybrid.py          # BM25 index + RRF fusion
    ├── generate.py        # Groq grounded answers
    └── pipeline.py        # ingest → retrieve → ask
```

---

## 7. Week 4 measured results (HR Policy)

**Metric:** hit-rate@3 — “Did the right page/doc appear in the top 3 chunks?”

| Mode | Result |
|------|--------|
| BEFORE — semantic only | **20%** (1/5) |
| AFTER — hybrid BM25+RRF | **60%** (3/5) |

### Demo comparison (same questions)

**Exact heading**  
`List the numbered items under the heading "Health and Safety, Security, Fire"`

| Mode | Outcome |
|------|---------|
| semantic | Wrong pages (≈90–92) |
| bm25 | Page **20** first |
| hybrid | Page **20** first |

**Meaning question**  
`How many annual leave days are full-time staff entitled to?`

| Mode | Outcome |
|------|---------|
| semantic | Page **32**, answer **24 days** |
| bm25 | Page **32**, answer **24 days** |
| hybrid | Page **32**, answer **24 days** |

**Takeaway:** semantic alone fails on exact headings; BM25 helps keywords; hybrid covers both.

---

## 8. Known limitations

- Checklist lists can be **split across chunks** → answer may list only part of items 1–11
- Pure paraphrase questions can still confuse BM25 / sometimes hybrid
- OCR is slower and can misread noisy scans
- Image-only pages need Tesseract installed locally
- `.env`, `.venv/`, and `chroma_db/` are local (not committed)

---

## 9. How to run

```powershell
cd C:\Users\SoftSuave\RagImplementation
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

1. Open http://localhost:8501  
2. Confirm sidebar: documents listed; OCR ready (if Tesseract installed)  
3. Click **Rebuild index**  
4. Choose mode: `hybrid` / `semantic` / `bm25`  
5. Ask a question and use the **Inspection view**

### Re-measure Week 4

```powershell
python eval_hit_rate.py --k 3
```

---

## 10. One-line summary

> We built an HR-policy RAG app (Week 3), then improved retrieval with hybrid BM25 + semantic + RRF, an inspection UI, and hit-rate@3 measurement (Week 4), and added optional Tesseract OCR for scanned/image documents.
