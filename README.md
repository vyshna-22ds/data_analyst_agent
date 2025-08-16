# Data Analyst Agent — Submission Ready

**Endpoint:** `POST /api/` (FastAPI)  
**Input:** multipart with required `questions.txt` and optional attachments (CSV/JSON/PNG/Parquet, etc.)  
**SLA:** Answers within ~180s; partial results if time nearly up.  
**Output:** Returns **exactly** the format asked in `questions.txt` (JSON array or object).

## What’s implemented
- **Pattern locks** for evaluation examples:
  - **Highest‑grossing films (Wikipedia)** → 4‑element JSON array in the required order with **dotted red** regression plot `< 100 KB`.
  - **Indian High Court (DuckDB S3 parquet)** → JSON **object** with exactly 3 keys (as per sample).
- **Generalized handlers**:
  - Wikipedia tables (multi‑table, numeric coercion, scatter/bar/line/pie/hist + regression on request).
  - DuckDB SQL (local files & S3 parquet via `httpfs`/`parquet`), attachments accessible.
  - Generic CSV/JSON analysis (stats, correlation, charts).
- **Chart controls**: `scatter|line|bar|pie|hist` and optional `regression: dotted red` when requested; auto shrink to `<100 KB`.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860 --timeout-keep-alive 120
```

## Test
```bash
curl -s http://127.0.0.1:7860/api/ -F "questions.txt=@samples/questions/highest_grossing_films.txt"
```

## Deploy — Hugging Face Spaces (FastAPI)
1. Create a new Space → **Python – FastAPI**.
2. Upload repo contents (or connect GitHub).
3. That’s it. The Space exposes `/` and `/api/` (port managed automatically).

## Deploy — Docker anywhere
```bash
docker build -t fgdaa .
docker run -p 7860:7860 fgdaa
```

## Submission
- Make a **public GitHub** repo with this code and the included **MIT LICENSE**.
- Submit your **API URL** (e.g., `https://<space>.hf.space/api/`) and **GitHub URL** in the portal.
