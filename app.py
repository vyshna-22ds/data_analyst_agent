import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import math
import numpy as np
import pandas as pd
import re
import io, json, time, shutil, tempfile, logging, traceback
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

from utils.io import load_questions_from_text, detect_output_format
from utils.timer import Deadline
from dispatcher import route
from handlers import wikipedia_handler, duckdb_handler, generic_handler
from handlers import sales_handler, network_handler

try:
    from pattern_matchers import indian_courts
    HAS_INDIAN_COURTS = True
except Exception:
    HAS_INDIAN_COURTS = False

log = logging.getLogger("uvicorn")

app = FastAPI(title="Data Analyst Agent", version="3.3.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "100")) * 1024 * 1024


# ---------- Helpers ----------

def _json_sanitize(obj):
    """Recursively clean NaN/inf for safe JSON encoding (without touching base64 strings)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.floating, np.integer)):
        val = float(obj) if isinstance(obj, np.floating) else int(obj)
        return val if not (isinstance(val, float) and not math.isfinite(val)) else None
    if isinstance(obj, np.ndarray):
        return [None if (isinstance(x, float) and not math.isfinite(x)) or (x != x)
                else float(x) if isinstance(x, np.floating)
                else int(x) if isinstance(x, np.integer)
                else x for x in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        safe = obj.replace([np.inf, -np.inf], np.nan)
        return safe.where(pd.notnull(safe), None).to_dict(orient="records")
    if isinstance(obj, pd.Series):
        safe = obj.replace([np.inf, -np.inf], np.nan)
        return safe.where(pd.notnull(safe), None).tolist()
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(x) for x in obj]
    return obj

def _extract_required_keys_from_prompt(raw_text: str) -> Optional[set]:
    """
    Finds keys listed in a 'Return a JSON object with keys:' section, e.g.
      - `total_sales`
      - `top_region`
    Returns a set of key names, or None if not present.
    """
    s = raw_text
    m = re.search(r"Return a JSON object with keys:(.*?)(?:\n\s*\n|$)", s, flags=re.IGNORECASE|re.DOTALL)
    if not m:
        return None
    block = m.group(1)
    keys = set(re.findall(r"`([A-Za-z0-9_]+)`", block))
    return keys or None


def _prune_nonanswer(obj):
    """Remove non-answer noise (file, preview, debug)."""
    DROP = {"file", "preview", "attachments", "attached", "_debug", "raw", "table_preview"}
    if isinstance(obj, dict):
        return {k: _prune_nonanswer(v) for k, v in obj.items() if k not in DROP}
    if isinstance(obj, list):
        return [_prune_nonanswer(x) for x in obj]
    return obj

def _is_nonempty_dict(d: Any) -> bool:
    return isinstance(d, dict) and bool(d) and any(v is not None for v in d.values())

def _pick_final_object(raw_text: str, per_q_results: List[Any], kv_answers: Dict[str, Any], out_format: str) -> Any:
    """
    Selection priority:
      (A) If prompt contains 'Return a JSON object with keys:' -> pick dict with EXACTLY those keys.
      (B) Else if it matches weather keys (legacy fast-path), return that.
      (C) Else return the last non-empty dict.
      (D) Else dict with most keys.
      (E) Else fall back to legacy behavior.
    """
    required_keys_weather = {
        "average_temp_c", "max_precip_date", "min_temp_c",
        "temp_precip_correlation", "average_precip_mm",
        "temp_line_chart", "precip_histogram"
    }
    candidates = [d for d in per_q_results if isinstance(d, dict)]

    # (A) Exact-key match from prompt
    required_from_prompt = _extract_required_keys_from_prompt(raw_text)
    if required_from_prompt:
        for d in reversed(candidates):  # prefer later
            ks = set(d.keys())
            if ks == required_from_prompt:
                return d
        # allow superset fallback if no exact match
        for d in reversed(candidates):
            if required_from_prompt.issubset(set(d.keys())):
                # strip extras to be safe
                return {k: d[k] for k in required_from_prompt}

    # (B) Weather fast-path (unchanged)
    for d in reversed(candidates):
        try:
            if required_keys_weather.issubset(set(d.keys())):
                return d
        except Exception:
            pass

    # (C) Last non-empty dict
    for d in reversed(candidates):
        if _is_nonempty_dict(d):
            return d

    # (D) Dict with most keys
    if candidates:
        best = max(candidates, key=lambda x: len(x.keys()))
        if best:
            return best

    # (E) Fallback to legacy behavior
    if out_format == "object":
        if len(kv_answers) == 1 and isinstance(next(iter(kv_answers.values())), dict):
            return next(iter(kv_answers.values()))
        return kv_answers
    else:
        only_dicts = [x for x in per_q_results if isinstance(x, dict)]
        if len(only_dicts) == 1:
            return only_dicts[0]
        return per_q_results


# ---------- Middleware ----------

@app.middleware("http")
async def _limit_upload_size(request: Request, call_next):
    cl = request.headers.get("content-length")
    try:
        if cl is not None and int(cl) > MAX_UPLOAD_BYTES:
            return JSONResponse(
                {"error": "Payload too large", "limit_bytes": MAX_UPLOAD_BYTES},
                status_code=413,
            )
    except Exception:
        pass
    return await call_next(request)


# ---------- Routes ----------

@app.get("/")
def root():
    return PlainTextResponse("OK. POST /api/ with multipart including questions.txt and any attachments.")

@app.get("/health")
def health():
    return {"ok": True}

@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": str(exc)})

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
            fname = os.path.basename(value.filename or "upload.bin")
            if "question" in fname.lower():
                fname = "questions.txt"
            data = await value.read()
            out_path = os.path.join(attachments_dir, fname)
            with open(out_path, "wb") as f:
                f.write(data)
            saved_files.append(fname)
            if fname == "questions.txt":
                saved_questions_path = out_path
        else:
            if key.lower() in {"questions", "q", "prompt"} and isinstance(value, str):
                questions_text_inline = value

    if saved_questions_path and os.path.exists(saved_questions_path):
        raw = open(saved_questions_path, "r", encoding="utf-8", errors="ignore").read()
    elif questions_text_inline:
        raw = questions_text_inline
    else:
        shutil.rmtree(workdir, ignore_errors=True)
        return JSONResponse({"error": "No questions provided."}, status_code=400)

    qs = load_questions_from_text(raw)
    out_format = detect_output_format(raw)
    s = raw.lower()

    # Special fast-paths
    if "highest grossing" in s and "wikipedia" in s and "list_of_highest-grossing_films" in s:
        try:
            c, earliest, corr, img = wikipedia_handler.solve_highest_grossing(
                "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
            )
            shutil.rmtree(workdir, ignore_errors=True)
            return JSONResponse(jsonable_encoder([int(c), str(earliest), float(corr), img]))
        except Exception as e:
            shutil.rmtree(workdir, ignore_errors=True)
            return JSONResponse(jsonable_encoder(["error", str(e), 0.0, None]))

    if HAS_INDIAN_COURTS:
        try:
            ans = indian_courts.try_answer(raw)
            if ans is not None:
                shutil.rmtree(workdir, ignore_errors=True)
                return JSONResponse(jsonable_encoder(ans))
        except Exception as e:
            log.warning(f"Indian-courts matcher failed: {e}")

    if "indian high court" in s and "read_parquet" in s and hasattr(duckdb_handler, "_indian_high_court_answers"):
        try:
            ans = duckdb_handler._indian_high_court_answers(attachments_dir=attachments_dir)
            shutil.rmtree(workdir, ignore_errors=True)
            return JSONResponse(jsonable_encoder(ans))
        except Exception as e:
            shutil.rmtree(workdir, ignore_errors=True)
            return JSONResponse({"error": f"Indian High Court branch failed: {e}"})

    # Generalized processing
    answers: List[Any] = []
    kv_answers: Dict[str, Any] = {}
    per_q_results: List[Any] = []  # keep raw results in order

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
            elif tag == "sales":                    # <-- add
                res = sales_handler.handle(q, attachments_dir)
            elif tag == "network":                  # <-- add
                res = network_handler.handle(q, attachments_dir)
            else:
                res = generic_handler.handle(q, attachments_dir)
        except Exception as e:
            res = {"error": str(e)}

        res = _prune_nonanswer(res)
        per_q_results.append(res)

        if out_format == "object":
            kv_answers[q] = res
        else:
            answers.append(res)

    shutil.rmtree(workdir, ignore_errors=True)

    # NEW: Always prefer returning the single final JSON object
    selected_payload = _pick_final_object(raw, per_q_results, kv_answers, out_format)

    payload = _json_sanitize(selected_payload)
    return JSONResponse(jsonable_encoder(payload))
