# handlers/generic_handler.py
import os, io, re, json, math, logging
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from utils.plots import scatter, line, bar, pie, hist
from utils.io import data_uri_png

log = logging.getLogger("uvicorn")

# --- filename patterns we try to extract from the question text ---
FILENAME_RE = re.compile(
    r"""(?:
            [`'"]                                  # quoted filenames
            (?P<qname>[^\s`'"]+\.(?:csv|json))
            [`'"]
         |
            (?P<name>\b[\w\-.]+\.(?:csv|json)\b)   # plain filenames
        )""",
    re.IGNORECASE | re.VERBOSE,
)

# ---------------------- JSON-SAFE HELPERS ---------------------- #
def _safe_float(x):
    """Return a JSON-safe float or None if NaN/Inf/invalid."""
    try:
        f = float(x)
        return f if math.isfinite(f) else None
    except Exception:
        return None

def _df_json_safe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert DataFrame to list-of-dicts with NaN/Inf -> None.
    Also converts pandas.Timestamp to ISO8601 strings.
    """
    # Replace inf with NaN, then turn NaN into None
    df2 = df.replace([np.inf, -np.inf], np.nan)
    # Convert datetimes to ISO strings (avoids non-serializable Timestamps)
    for c in df2.columns:
        if pd.api.types.is_datetime64_any_dtype(df2[c]):
            df2[c] = df2[c].dt.tz_localize(None, nonexistent='shift_forward', ambiguous='NaT').astype('datetime64[ns]')
            df2[c] = df2[c].dt.strftime('%Y-%m-%dT%H:%M:%S')
    # Final None conversion
    df2 = df2.where(pd.notnull(df2), None)
    return df2.to_dict(orient="records")

def _series_list_safe(s: pd.Series) -> List[Any]:
    """Series -> list with NaN/Inf -> None."""
    s2 = s.replace([np.inf, -np.inf], np.nan)
    s2 = s2.where(pd.notnull(s2), None)
    return s2.tolist()

def _coerce_numeric_inplace(df: pd.DataFrame, col: str) -> None:
    try:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    except Exception:
        pass
# --------------------------------------------------------------- #


def _list_files(attachments_dir: str) -> List[str]:
    if not (attachments_dir and os.path.isdir(attachments_dir)):
        return []
    try:
        files = [
            os.path.join(attachments_dir, f)
            for f in os.listdir(attachments_dir)
            if os.path.isfile(os.path.join(attachments_dir, f))
        ]
        return files
    except Exception as e:
        log.warning(f"[generic] _list_files error: {e}")
        return []


def _late_rescan(attachments_dir: str) -> List[str]:
    """Late re-scan (guards against rare Windows temp timing)."""
    try:
        files = [
            os.path.join(attachments_dir, f)
            for f in os.listdir(attachments_dir)
            if os.path.isfile(os.path.join(attachments_dir, f))
        ]
        return files
    except Exception:
        return []


def _prefer_from_question(question: str, files: List[str]) -> Optional[str]:
    """If the question mentions a specific filename, prefer that."""
    if not files:
        return None  # avoid touching files when empty
    s = (question or "").lower()
    m = FILENAME_RE.search(s)
    if not m:
        return None
    mentioned = (m.group("qname") or m.group("name") or "").lower()
    for p in files:
        if os.path.basename(p).lower() == mentioned:
            return p
    return None


def _find_first_with_ext(files: List[str], exts: List[str]) -> Optional[str]:
    for p in files:
        lp = p.lower()
        if any(lp.endswith(e) for e in exts):
            return p
    return None


def _sniff_tabular(files: List[str]) -> Optional[str]:
    """Last resort: pick any file whose first bytes look like CSV/TSV/JSON-ish."""
    for p in files:
        try:
            with open(p, "rb") as fh:
                head = fh.read(1024).lower()
            if b"," in head or b"\t" in head or head.strip().startswith((b"{", b"[")):
                return p
        except Exception:
            continue
    return None


def _load_table(path: str) -> pd.DataFrame:
    lp = path.lower()
    if lp.endswith(".csv"):
        # Try common separators
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, sep=";")
    if lp.endswith(".json"):
        try:
            return pd.read_json(path, lines=False)
        except Exception:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.DataFrame(data)
            return pd.json_normalize(data)
    # As a defensive fallback, attempt CSV parsing
    try:
        return pd.read_csv(path)
    except Exception:
        raise RuntimeError(f"Unsupported or unreadable file: {os.path.basename(path)}")


def _try_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-convert obvious date-like columns (first two object columns)."""
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols[:2]:
        try:
            df[c] = pd.to_datetime(df[c], errors="coerce")
        except Exception:
            pass
    return df


def _choose_numeric(df: pd.DataFrame) -> List[str]:
    nums = df.select_dtypes(include="number").columns.tolist()
    if not nums and len(df.columns) >= 2:
        # coerce first non-numeric if possible
        try:
            col = df.columns[1]
            _coerce_numeric_inplace(df, col)
            nums = df.select_dtypes(include="number").columns.tolist()
        except Exception:
            pass
    return nums


def handle(question: str, attachments_dir: str) -> Dict[str, Any]:
    log.info(f"[generic] attachments_dir={attachments_dir}")
    files = _list_files(attachments_dir)
    log.info(f"[generic] files(initial)={files}")

    # Late re-scan if empty (rare timing case)
    if not files and os.path.isdir(attachments_dir):
        files = _late_rescan(attachments_dir)
        log.info(f"[generic] files(rescan)={files}")

    # If only s3_urls.txt is present, provide a targeted hint instead of the generic message.
    try:
        basenames = [os.path.basename(p).lower() for p in files]
        has_s3_list = "s3_urls.txt" in basenames
        has_tabular = any(p.lower().endswith((".csv", ".json")) for p in files)
        if has_s3_list and not has_tabular:
            log.info("[generic] detected s3_urls.txt without CSV/JSON; returning HC hint.")
            return {
                "answer": (
                    "Detected s3_urls.txt but no CSV/JSON to visualize here.\n"
                    "If this is the Indian High Court task, include the phrase "
                    "'Indian high court' and a read_parquet(...) mention in your question, "
                    "or let the DuckDB route run; the HC handler will read attachments/s3_urls.txt."
                ),
                "attachments_seen": basenames,
            }
    except Exception:
        pass

    # 1) try exact filename mentioned in question
    target = _prefer_from_question(question, files)

    # 2) otherwise any CSV/JSON we have
    if not target:
        target = _find_first_with_ext(files, [".csv", ".json"])

    # 3) last resort: sniff something tabular
    if not target:
        target = _sniff_tabular(files)

    if not target or not os.path.exists(target):
        log.info("[generic] no target file found after all strategies.")
        return {
            "answer": "No data file provided. Upload CSV/JSON or use DuckDB SQL.",
            "attachments_seen": [os.path.basename(p) for p in files],
        }

    log.info(f"[generic] chosen target={target}")

    # Load & clean
    df = _load_table(target)
    df = _try_parse_dates(df).copy()
    # Replace inf with NaN to sanitize later
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Build a JSON-safe head with ISO dates
    out: Dict[str, Any] = {
        "file": os.path.basename(target),
        "rows": int(len(df)),
        "cols": list(map(str, df.columns)),
        "head": _df_json_safe(df.head(10)),
    }

    # --- Plot heuristic ---
    s = (question or "").lower()
    try:
        png = None
        # Prefer explicit chart asks
        if "pie" in s and len(df.columns) >= 2:
            png = pie(df.head(20), df.columns[0], df.columns[1], None, 100_000)
        elif "hist" in s and len(df.columns) >= 1:
            num_cols = df.select_dtypes(include="number").columns
            col = (num_cols[0] if len(num_cols) else df.columns[0])
            png = hist(df, col, 20, None, 100_000)
        elif "line" in s and len(df.columns) >= 2:
            xcol = df.columns[0]
            ycand = _choose_numeric(df)
            ycol = ycand[0] if ycand else (df.columns[1] if len(df.columns) > 1 else df.columns[0])
            png = line(df, xcol, ycol, None, 100_000)
        elif "bar" in s and len(df.columns) >= 2:
            ycand = _choose_numeric(df)
            ycol = ycand[0] if ycand else df.columns[1]
            png = bar(df.head(50), df.columns[0], ycol, None, 100_000)
        else:
            # default: scatter with regression if we have >=2 numeric cols
            nums = _choose_numeric(df)
            if len(nums) >= 2:
                png = scatter(df, nums[0], nums[1], regression=True, title=None, max_bytes=100_000)

        if png is not None:
            out["plot"] = data_uri_png(png)
    except Exception as e:
        out["plot_error"] = str(e)

    # Optional quick stats if the question mentions common terms (JSON-safe)
    try:
        lc = s
        num_cols = df.select_dtypes(include="number").columns
        if ("average" in lc or "mean" in lc) and len(num_cols):
            out["numeric_means"] = {
                c: _safe_float(pd.to_numeric(df[c], errors="coerce").mean()) for c in num_cols[:5]
            }
        if ("correlation" in lc or "corr" in lc) and len(num_cols) >= 2:
            corr_val = df[num_cols[:2]].corr().iloc[0, 1]
            out["pairwise_corr"] = _safe_float(corr_val)
        if any(w in lc for w in ("min", "maximum", "max ")):
            if len(num_cols):
                out["numeric_min"] = {
                    c: _safe_float(pd.to_numeric(df[c], errors="coerce").min()) for c in num_cols[:5]
                }
                out["numeric_max"] = {
                    c: _safe_float(pd.to_numeric(df[c], errors="coerce").max()) for c in num_cols[:5]
                }
    except Exception:
        pass

    return out
