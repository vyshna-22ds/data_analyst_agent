# app.py
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import math
import numpy as np
import pandas as pd
import io, json, time, shutil, tempfile, re, logging, os, traceback
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware  # NEW: CORS

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

# --- NEW: permissive CORS (handy for browser/manual tests) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)

# --- NEW: max upload size guard (default 100 MB; override via env MAX_UPLOAD_MB) ---
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "100")) * 1024 * 1024

def _json_sanitize(obj):
    """
    Recursively replace NaN/±Inf with None so JSONResponse never crashes.
    Handles floats, numpy scalars/arrays, pandas Series/DataFrames, lists, dicts.
    """
    # floats
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    # numpy scalars
    if isinstance(obj, (np.floating, np.integer)):
        val = float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(val, float) and not math.isfinite(val):
            return None
        return val
    # numpy arrays
    if isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.number):
            arr = obj.astype(float)
            arr[~np.isfinite(arr)] = np.nan
            return [None if (isinstance(x, float) and not math.isfinite(x)) or (x != x) else x for x in arr]
        return obj.tolist()
    # pandas
    if isinstance(obj, pd.DataFrame):
        safe = obj.replace([np.inf, -np.inf], np.nan)
        recs = safe.where(pd.notnull(safe), None).to_dict(orient="records")
        return [_json_sanitize(r) for r in recs]
    if isinstance(obj, pd.Series):
        safe = obj.replace([np.inf, -np.inf], np.nan)
        return [_json_sanitize(x) for x in safe.where(pd.notnull(safe), None).tolist()]
    # containers
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(x) for x in obj]
    return obj


@app.middleware("http")
async def _limit_upload_size(request: Request, call_next):
    # If Content-Length is present and too big, fail early with 413
    cl = request.headers.get("content-length")
    try:
        if cl is not None and int(cl) > MAX_UPLOAD_BYTES:
            return JSONResponse(
                {"error": "Payload too large", "limit_bytes": MAX_UPLOAD_BYTES},
                status_code=413,
            )
    except Exception:
        # Malformed or missing header -> continue; server/proxy may still enforce limits
        pass
    return await call_next(request)

@app.get("/")
def root():
    return PlainTextResponse(
        "OK. POST /api/ with multipart including questions.txt (required) and any attachments."
    )

# --- NEW: simple health check for uptime/retries ---
@app.get("/health")
def health():
    return {"ok": True}

# Always respond with JSON on unexpected errors
@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": str(exc)},
    )

# Tiny helper to inspect what arrived (very helpful in Windows + curl.exe)
@app.post("/debug/echo")
async def debug_echo(request: Request):
    form = await request.form()
    items = list(form.multi_items())  # cache iterator
    out = []
    for k, v in items:
        if hasattr(v, "filename"):
            b = await v.read()
            out.append({
                "key": k,
                "type": "UploadFile",
                "filename": v.filename,
                "content_type": getattr(v, "content_type", None),
                "size": len(b),
            })
        else:
            out.append({"key": k, "type": type(v).__name__, "value_sample": str(v)[:120]})
    return JSONResponse(jsonable_encoder({"received": out}))

@app.post("/api/")
async def api(request: Request):
    deadline = Deadline(170.0)

    # Workspace
    workdir = tempfile.mkdtemp(prefix="fgdaa_")
    attachments_dir = os.path.join(workdir, "attachments")
    os.makedirs(attachments_dir, exist_ok=True)

    # Read & cache the multipart ONCE (iterator is one-shot)
    form = await request.form()
    items = list(form.multi_items())

    try:
        log.info(f"Form keys -> {list(form.keys())}")
        log.info(
            "Form items -> " + ", ".join(
                [f"{k}:{'UploadFile' if hasattr(v,'filename') else type(v).__name__}" for k, v in items]
            )
        )
        log.info(
            "Content-Types -> " + ", ".join(
                [f"{k}:{getattr(v,'content_type',type(v).__name__)}" for k, v in items]
            )
        )
    except Exception:
        pass

    saved_files: List[str] = []
    saved_questions_path: Optional[str] = None
    questions_text_inline: Optional[str] = None

    # Save all uploads (duck-typed: any object with .filename is treated as a file)
    for key, value in items:
        if hasattr(value, "filename"):
            fname_in = value.filename or str(key) or "upload.bin"
            fname = os.path.basename(fname_in)
            lower_key = str(key).lower()
            lower_fname = fname.lower()

            # Heuristic: if this looks like a questions file, normalize the name
            looks_like_questions = (
                "question" in lower_key
                or "question" in lower_fname
                or lower_key in {"questions.txt", "questions", "q"}
                or lower_fname == "questions.txt"
            )
            if looks_like_questions:
                fname = "questions.txt"

            # Add extension hints if missing
            if "." not in fname:
                ct = (getattr(value, "content_type", "") or "").lower()
                if "csv" in ct:
                    fname += ".csv"
                elif "json" in ct:
                    fname += ".json"
                elif "text" in ct:
                    fname += ".txt"

            data = await value.read()
            out_path = os.path.join(attachments_dir, fname)
            base, ext = os.path.splitext(out_path)
            salt = 1
            while os.path.exists(out_path):
                out_path = f"{base}_{salt}{ext}"
                salt += 1
            with open(out_path, "wb") as f:
                f.write(data)
            saved_files.append(os.path.basename(out_path))

            if fname.lower() == "questions.txt":
                saved_questions_path = out_path
        else:
            # Accept inline text fields as questions too
            if str(key).lower() in {"questions", "q", "prompt"} and isinstance(value, str) and value.strip():
                questions_text_inline = value

    # Helpful debug
    log.info(f"Saved attachments: {saved_files}")

    # Smart fallbacks for questions
    if not saved_questions_path:
        # Any saved file whose name contains 'question'
        for f in saved_files:
            if "question" in f.lower():
                saved_questions_path = os.path.join(attachments_dir, f)
                break
    if not saved_questions_path:
        # Any .txt we saved
        txts = [p for p in os.listdir(attachments_dir) if p.lower().endswith(".txt")]
        if txts:
            saved_questions_path = os.path.join(attachments_dir, txts[0])

    # Load questions (file first, else inline)
    if saved_questions_path and os.path.exists(saved_questions_path):
        with open(saved_questions_path, "rb") as fh:
            raw = fh.read().decode("utf-8", errors="ignore")
    elif questions_text_inline:
        raw = questions_text_inline
    else:
        shutil.rmtree(workdir, ignore_errors=True)
        return JSONResponse(
            jsonable_encoder({"error": "No questions provided. Upload questions.txt or include a 'questions' text field."}),
            status_code=400,
        )

    # ---- Normal flow from here ----
    qs = load_questions_from_text(raw)
    out_format = detect_output_format(raw)
    s = raw.lower()

    # 1) Highest-grossing films (strict array of 4)
    if "highest grossing" in s and "wikipedia" in s and "list_of_highest-grossing_films" in s:
        try:
            c, earliest, corr, img = wikipedia_handler.solve_highest_grossing(
                "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
            )
            try:
                shutil.rmtree(workdir, ignore_errors=True)
            finally:
                return JSONResponse(jsonable_encoder([int(c), str(earliest), float(corr), img]))
        except Exception as e:
            try:
                shutil.rmtree(workdir, ignore_errors=True)
            finally:
                return JSONResponse(jsonable_encoder(["error", str(e), 0.0, None]))

    # 2) Pattern-based Indian High Court (if present)
    if HAS_INDIAN_COURTS:
        try:
            ans = indian_courts.try_answer(raw)
            if ans is not None:
                try:
                    shutil.rmtree(workdir, ignore_errors=True)
                finally:
                    return JSONResponse(jsonable_encoder(ans))
        except Exception as e:
            log.warning(f"Indian-courts fast-path failed: {e}")

    # 3) Indian High Court (S3) legacy/special-case (runs once, exits early)
    if "indian high court" in s and "read_parquet" in s and hasattr(duckdb_handler, "_indian_high_court_answers"):
        log.info(
            "HC special-case TRIGGERED (attachments_dir=%s, has s3_urls.txt=%s)",
            attachments_dir,
            os.path.exists(os.path.join(attachments_dir, "s3_urls.txt")),
        )
        try:
            # Pass attachments_dir so s3_urls.txt can be used if uploaded
            ans = duckdb_handler._indian_high_court_answers(attachments_dir=attachments_dir)
            try:
                shutil.rmtree(workdir, ignore_errors=True)
            finally:
                return JSONResponse(jsonable_encoder(ans))
        except Exception as e:
            try:
                shutil.rmtree(workdir, ignore_errors=True)
            finally:
                return JSONResponse(jsonable_encoder({
                    "error": f"Indian High Court branch failed: {e}",
                    "hint": "If bucket listing is blocked, attach s3_urls.txt with explicit parquet URLs (one per line).",
                }))

    # 4) Generalized processing (fallback)
    answers: List[Any] = []
    kv_answers: Dict[str, Any] = {}

    for q in qs:
        if deadline.nearly_out():
            note = "Time budget nearly exhausted."
            if out_format == "object":
                kv_answers[q] = note
            else:
                answers.append(note)
            break

        try:
            tag = route(q)
            if tag == "wikipedia":
                res = wikipedia_handler.handle(q)
            elif tag == "duckdb":
                res = duckdb_handler.handle(q, attachments_dir)
            else:
                res = generic_handler.handle(q, attachments_dir)
        except Exception as e:
            res = {"error": str(e)}

        if out_format == "object":
            kv_answers[q] = res
        else:
            # For array outputs, keep items as strings for simple rubric compliance
            answers.append(json.dumps(res, ensure_ascii=False))

    try:
        shutil.rmtree(workdir, ignore_errors=True)
    except Exception:
        pass

    payload = kv_answers if out_format == "object" else answers
    payload = _json_sanitize(payload)
    return JSONResponse(jsonable_encoder(payload))

