# Ask Legal Contracts (RAG)

Retrieval-Augmented Generation mini app — ask questions over **legal contracts** with grounded answers and sources.

This is not legal advice. Answers come only from files you put in `documents/`.

## Sample contracts

The repo includes three fictional sample files you can replace with your own PDFs or text:

- `documents/nda.txt`
- `documents/msa.txt`
- `documents/employment_agreement.txt`

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

Open http://localhost:8501, click **Rebuild index**, and ask a question (for example: *What does the Limitation of Liability clause say?*).

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

## Langfuse traces (Week 5)

Every Ask (Streamlit) and every eval run logs a full trace: question, retrieved chunks, Groq answer. Set keys in `.env` (Project Settings → API Keys at [cloud.langfuse.com](https://cloud.langfuse.com)). Use **EU** (`https://cloud.langfuse.com`) or **US** (`https://us.cloud.langfuse.com`) to match where you created the project.

`LANGFUSE_RELEASE=week5-error-analysis` tags traces so you can filter by release in the Langfuse UI. You do not create a separate “release” object.

Upload the frozen Week 5 20 traces + human OK/FAIL scores (no extra Groq calls):

```powershell
pip install -r requirements.txt
python eval/week5_langfuse.py
```

Then open Langfuse → **Traces** (session `week5-error-analysis`) and **Datasets** → `week5-spa-error-analysis`.

## Agent vs workflow (Week 7 · Track F)

Hand-built ReAct loop (not LangGraph): think → tool → observe → repeat, with max steps / tokens / time. Same multi-hop contract tasks also run as a **fixed** sequence (list contracts → search each part → one answer).

In the app: tabs **Agent loop**, **Fixed workflow**, **Agent vs workflow race**.

```powershell
python eval/week7_race.py
```

Results: `eval/week7_race.json` (speed, Groq tokens/cost, phrase reliability, and which one to ship).

## Retrieval eval (Week 4)

```powershell
python eval_hit_rate.py --k 3 --rebuild
```

Eval questions live in `eval/contracts_eval.json`. If you replace the sample contracts, update `expected_source` and `expected_phrases` to match your files.

## Answer evals (Week 6)

One command — assertion checks, a validated clause-answer judge, and before/after scores per problem type (uses the Service Provider Agreement traces from Week 5):

```powershell
python eval/week6_eval.py
```

Results: `eval/week6_results.json`.

## Contract squad vs single agent (Week 10 · Track F)

Manager plus two specialists (`clause-reader`, `coverage-checker`) raced against the existing single agent on the same six questions in `eval/week7_tasks.json`.

```powershell
python eval/week10_race.py
```

Writes `eval/week10_race.json` and `WEEK_10_REPORT.md` (quality, speed, tokens, cost, and which one to keep). The app tab is **Week 10 squad**.

## Production log, cost, and the failure test (Week 11 · Track F)

Every Ask writes a local trace (`logs/requests.jsonl`) with time and cost on `cache_lookup`, `retrieve`, and `generate`. Repeating the same question is an exact cache hit and does not call Groq again.

The support drill is a 1,200-line log with one planted bad answer: it cites Clause 12.9 and a 60-day notice as the way to terminate. The termination right is Clause 7.1 (30 days). Write-up: `WEEK_11_REPORT.md`. The app tab is **Week 11 production**.

```powershell
python eval/week11_drill.py
python eval/week11_failure_test.py
python eval/week11_cost.py
```
