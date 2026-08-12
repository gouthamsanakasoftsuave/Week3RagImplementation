# Week3RagImplementation

Retrieval-Augmented Generation (RAG) mini app — ask questions over your own documents with grounded answers and sources.

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
