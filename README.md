# Week3RagImplementation

Retrieval-Augmented Generation (RAG) mini app — ask questions over your own documents with grounded answers and sources.

Week 3: load → chunk → embed → retrieve → grounded answer  
Week 4: hybrid search (BM25 + semantic + RRF), inspection view, hit-rate@3  

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Add your `GROQ_API_KEY` to `.env`, then:

```powershell
streamlit run app.py
```

Open http://localhost:8501, click **Rebuild index**, and ask a question.

## OCR (scanned / image PDFs)

Normal PDFs use digital text (`pypdf`). If a page has little/no text (scan/image), the app can OCR it with **Tesseract**.

### 1. Install Tesseract on Windows
1. Download installer: https://github.com/UB-Mannheim/tesseract/wiki  
2. Install (default path is fine): `C:\Program Files\Tesseract-OCR`  
3. Tick English language data during setup  

Or with winget (if available):

```powershell
winget install --id UB-Mannheim.TesseractOCR -e
```

### 2. Python packages
Already in `requirements.txt`: `pymupdf`, `pytesseract`, `Pillow`.

### 3. `.env` settings
```env
ENABLE_OCR=true
OCR_MIN_CHARS=40
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

Then rebuild the index. Image files (`.png`, `.jpg`, …) in `documents/` are also OCR’d when Tesseract is available.

Without Tesseract installed, the app still runs — it just skips OCR and uses digital text only.

## Week 4 eval

```powershell
python eval_hit_rate.py --k 3
```
