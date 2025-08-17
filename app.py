# app.py
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import math
import numpy as np
import pandas as pd
import io, json, time, shutil, tempfile, re, logging, traceback
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

from utils.io import load_questions_from_text, detect_output_format
from utils.timer import Deadline
from dispatcher import route
from handlers import wikipedia_handler, duckdb_handler, generic_handler

# Optional: pattern-based matcher for Indian High Court via S3 parquet
try:
    from pattern_matchers import indian_courts
    HAS_INDIAN_COURTS = True
except Exception:
    HAS_INDIAN_COURTS = False

log = logging.getLogger("uvicorn")

app = FastAPI(title="Data Analyst Agent", version="3.3.3")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)

# --- Max upload size guard ---
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "100")) * 1024 * 1024


def _json_sanitize(obj):
    """Recursively replace NaN/Inf with None so JSONResponse never crashes."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.floating, np.integer)):
        val = float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(val, float) and not math.isfinite(val):
            return None
        return val
    if isinstance(obj, np.ndarray):
        arr = obj.astype(float) if np.issubdtype(obj.dtype, np.number) else obj
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.where(np.isfinite(arr), arr, np.nan)
            return [None if (x != x) or not math.isfinite(x) else float(x) for x in arr]
        return arr.tolist()
    if isinstance(obj, pd.DataFrame):
        safe = obj.replace([np.inf, -np.inf], np.nan)
        recs = safe.where(pd.notnull(safe), None).to_dict(orient="records")
        return [_json_sanitize(r) for r in recs]
    if isinstance(obj, pd.Series):
        safe = obj.replace([np.inf, -np.inf], np.nan)
        return [_json_sanitize(x) for x in safe.where(pd.notnull(safe), None).tolist()]
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(x) for x in obj]
    return obj


def _prune_nonanswer(obj):
    """Strip non-answer noise like file/preview/debug keys from nested structures."""
    DROP_KEYS = {"file", "preview", "attachments", "attached", "_debug", "raw", "table_preview"}
    if isinstance(obj, dict):
        return {k: _prune_nonanswer(v) for k, v in obj.items() if k not in DROP_KEYS}
    if isinstance(obj, list):
        return [_prune_nonanswer(v) for v in obj]
    return obj


@app.middleware("http")
async def _limit_upload_size(request: Request, call_next):
    cl = request.headers.get("content-length")
    try:
        if cl is not None and int(cl) > MAX_UPLOAD_BYTES:
            return JSONResponse({"error": "Payload too large", "limit_bytes": MAX_UPLOAD_BYTES}, status_code=413)
    except Exception:
        pass
    return await call_next(request)


@app.get("/")
def root():
    return PlainTextResponse("OK. POST /api/ with multipart including questions.txt (required) and any attachments.")


@app.get("/health")
def health():
    return {"ok": True}


@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": str(exc)})


@app.post("/debug/echo")
async def debug_echo(request: Request):
    form = await request.form()
    items = list(form.multi_items())
    out = []
    for k, v in items:
        if hasattr(v, "filename"):
            b = await v.read()
            out.append({"key": k, "type": "UploadFile", "filename": v.filename,
                        "content_type": getattr(v, "content_type", None), "size": len(b)})
        else:
            out.append({"key": k, "type": type(v).__name__, "value_sample": str(v)[:120]})
    return JSONResponse(jsonable_encoder({"received": out}))


@app.post("/api/")
async def api(request: Request):
    deadline = Deadline(170.0)
    workdir = tempfile.mkdtemp(prefix="fgdaa_")
    attachments_dir = os.path.join(workdir, "attachments")
    os.makedirs(attachments_dir, exist_ok=True)
    form = await request.form()
    items = list(form.multi_items())

    saved_files: List[str] = []
    saved_questions_path: Optional[str] = None
    questions_text_inline: Optional[str] = None

    for key, value in items:
        if hasattr(value, "filename"):
            fname = value.filename or str(key) or "upload.bin"
            lower_key, lower_fname = str(key).lower(), fname.lower()
            looks_like_questions = (
                "question" in lower_key or "question" in lower_fname
                or lower_key in {"questions.txt", "questions", "q"}
                or lower_fname == "questions.txt"
            )
            if looks_like_questions:
                fname = "questions.txt"
            if "." not in fname:
                ct = (getattr(value, "content_type", "") or "").lower()
                if "csv" in ct: fname += ".csv"
                elif "json" in ct: fname += ".json"
                elif "text" in ct: fname += ".txt"
            data = await value.read()
            out_path = os.path.join(attachments_dir, fname)
            base, ext, salt = os.path.splitext(out_path)[0], os.path.splitext(out_path)[1], 1
            while os.path.exists(out_path):
                out_path = f"{base}_{salt}{ext}"
                salt += 1
            with open(out_path, "wb") as f: f.write(data)
            saved_files.append(os.path.basename(out_path))
            if fname.lower() == "questions.txt":
                saved_questions_path = out_path
        else:
            if str(key).lower() in {"questions", "q", "prompt"} and isinstance(value, str) and value.strip():
                questions_text_inline = value

    if not saved_questions_path:
        for f in saved_files:
            if "question" in f.lower():
                saved_questions_path = os.path.join(attachments_dir, f)
                break
    if not saved_questions_path:
        txts = [p for p in os.listdir(attachments_dir) if p.lower().endswith(".txt")]
        if txts: saved_questions_path = os.path.join(attachments_dir, txts[0])

    if saved_questions_path and os.path.exists(saved_questions_path):
        raw = open(saved_questions_path, "rb").read().decode("utf-8", errors="ignore")
    elif questions_text_inline:
        raw = questions_text_inline
    else:
        shutil.rmtree(workdir, ignore_errors=True)
        return JSONResponse({"error": "No questions provided"}, status_code=400)

    qs = load_questions_from_text(raw)
    out_format = detect_output_format(raw)
    s = raw.lower()

    # Special-case Wikipedia highest-grossing films
    if "highest grossing" in s and "wikipedia" in s and "list_of_highest-grossing_films" in s:
        try:
            c, earliest, corr, img = wikipedia_handler.solve_highest_grossing(
                "https://en.wikipedia.org/wiki/List_of_highest-grossing_films")
            shutil.rmtree(workdir, ignore_errors=True)
            return JSONResponse(jsonable_encoder([int(c), str(earliest), float(corr), img]))
        except Exception as e:
            shutil.rmtree(workdir, ignore_errors=True)
            return JSONResponse(jsonable_encoder(["error", str(e), 0.0, None]))

    # Pattern matcher for Indian courts
    if HAS_INDIAN_COURTS:
        try:
            ans = indian_courts.try_answer(raw)
            if ans is not None:
                shutil.rmtree(workdir, ignore_errors=True)
                return JSONResponse(jsonable_encoder(ans))
        except Exception as e:
            log.warning(f"Indian-courts fast-path failed: {e}")

    # Legacy Indian High Court handler
    if "indian high court" in s and "read_parquet" in s and hasattr(duckdb_handler, "_indian_high_court_answers"):
        try:
            ans = duckdb_handler._indian_high_court_answers(attachments_dir=attachments_dir)
            shutil.rmtree(workdir, ignore_errors=True)
            return JSONResponse(jsonable_encoder(ans))
        except Exception as e:
            shutil.rmtree(workdir, ignore_errors=True)
            return JSONResponse(jsonable_encoder({"error": f"Indian High Court branch failed: {e}"}))

    # Generalized flow
    answers, kv_answers = [], {}
    for q in qs:
        if deadline.nearly_out():
            note = "Time budget nearly exhausted."
            if out_format == "object": kv_answers[q] = note
            else: answers.append(note)
            break
        try:
            tag = route(q)
            if tag == "wikipedia": res = wikipedia_handler.handle(q)
            elif tag == "duckdb": res = duckdb_handler.handle(q, attachments_dir)
            else: res = generic_handler.handle(q, attachments_dir)
        except Exception as e:
            res = {"error": str(e)}

        cleaned = _prune_nonanswer(res)
        if out_format == "object": kv_answers[q] = cleaned
        else: answers.append(cleaned)

    shutil.rmtree(workdir, ignore_errors=True)

    if out_format == "object":
        if len(kv_answers) == 1:
            only_val = next(iter(kv_answers.values()))
            payload = only_val if isinstance(only_val, dict) else kv_answers
        else:
            payload = kv_answers
    else:
        payload = answers

    payload = _json_sanitize(payload)
    return JSONResponse(jsonable_encoder(payload))
