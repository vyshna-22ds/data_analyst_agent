# handlers/generic_handler.py
import os, io, re, json, base64, math, pandas as pd
from typing import Dict, Any, List, Optional
from utils.plots import scatter, line, bar, pie, hist
from utils.io import data_uri_png
import logging

log = logging.getLogger("uvicorn")

FILENAME_RE = re.compile(
    r"""(?:
            [`'"]
            (?P<qname>[^\s`'"]+\.(?:csv|json))
            [`'"]
         |
            (?P<name>\b[\w\-.]+\.(?:csv|json)\b)
        )""",
    re.IGNORECASE | re.VERBOSE,
)

# ---------- NEW: small helpers ----------

def _b64_png(png_bytes: bytes) -> str:
    """Return raw base64 (no data: prefix)."""
    try:
        return base64.b64encode(png_bytes).decode("ascii")
    except Exception:
        return ""

def _to_native(x: Any) -> Any:
    """Cast numpy/pandas scalars to built-ins; sanitize NaN/Inf."""
    try:
        import numpy as np
    except Exception:
        np = None

    # pandas/NumPy scalar -> Python
    if hasattr(x, "item"):
        try:
            x = x.item()
        except Exception:
            pass
    if np is not None and isinstance(x, (np.generic,)):  # extra guard
        try:
            x = x.item()
        except Exception:
            pass

    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return 0.0
    return x

def _sanitize_mapping(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, (list, tuple)):
            out[k] = [_to_native(e) for e in v]
        elif isinstance(v, dict):
            out[k] = _sanitize_mapping(v)
        else:
            out[k] = _to_native(v)
    return out

# ---------- (existing helpers unchanged below) ----------

def _list_files(attachments_dir: str) -> List[str]:
    if not (attachments_dir and os.path.isdir(attachments_dir)):
        return []
    try:
        return [
            os.path.join(attachments_dir, f)
            for f in os.listdir(attachments_dir)
            if os.path.isfile(os.path.join(attachments_dir, f))
        ]
    except Exception as e:
        log.warning(f"[generic] _list_files error: {e}")
        return []

def _late_rescan(attachments_dir: str) -> List[str]:
    try:
        return [
            os.path.join(attachments_dir, f)
            for f in os.listdir(attachments_dir)
            if os.path.isfile(os.path.join(attachments_dir, f))
        ]
    except Exception:
        return []

def _prefer_from_question(question: str, files: List[str]) -> Optional[str]:
    if not files:
        return None
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
        if any(p.lower().endswith(e) for e in exts):
            return p
    return None

def _sniff_tabular(files: List[str]) -> Optional[str]:
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
    try:
        return pd.read_csv(path)
    except Exception:
        raise RuntimeError(f"Unsupported or unreadable file: {os.path.basename(path)}")

def _try_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols[:2]:
        try:
            df[c] = pd.to_datetime(df[c])
        except Exception:
            pass
    return df

def _choose_numeric(df: pd.DataFrame) -> List[str]:
    nums = df.select_dtypes(include="number").columns.tolist()
    if not nums and len(df.columns) >= 2:
        try:
            col = df.columns[1]
            df[col] = pd.to_numeric(df[col], errors="coerce")
            nums = df.select_dtypes(include="number").columns.tolist()
        except Exception:
            pass
    return nums

# ---------- NEW: weather schema exact output ----------

def _maybe_weather_schema(question: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Detect 'Return a JSON object with keys: average_temp_c, ... temp_line_chart, precip_histogram' style prompts."""
    s = (question or "").lower()
    if "return a json object with keys" not in s:
        return None

    needed = {"average_temp_c", "max_precip_date", "min_temp_c",
              "temp_precip_correlation", "average_precip_mm",
              "temp_line_chart", "precip_histogram"}
    if not needed.issubset(set(w.strip("`*_- ") for w in re.findall(r"[a-z0-9_]+", s))):
        # keys don’t match the known weather schema -> skip
        return None

    # Try to locate expected columns; allow slight casing differences
    cols = {c.lower(): c for c in df.columns}
    if not {"date", "temperature_c", "precip_mm"}.issubset(cols.keys()):
        return None

    date_col = cols["date"]
    t_col = cols["temperature_c"]
    p_col = cols["precip_mm"]

    # Ensure numeric for stats
    t = pd.to_numeric(df[t_col], errors="coerce")
    p = pd.to_numeric(df[p_col], errors="coerce")

    avg_temp = float(t.mean())
    min_temp = float(t.min())
    # idxmax may be NaN if all NaN, guard it
    try:
        idx = int(p.idxmax())
        max_precip_dt = pd.to_datetime(df.loc[idx, date_col]).isoformat()
    except Exception:
        max_precip_dt = pd.to_datetime(df[date_col].iloc[0]).isoformat()

    try:
        corr = float(pd.concat([t, p], axis=1).corr().iloc[0, 1])
        if math.isnan(corr) or math.isinf(corr):
            corr = 0.0
    except Exception:
        corr = 0.0

    avg_precip = float(p.mean())

    # Charts as **raw base64** under 100k
    temp_line_png = line(df, date_col, t_col, None, 100_000)
    precip_hist_png = hist(df, p_col, 20, None, 100_000)

    out = {
        "average_temp_c": _to_native(avg_temp),
        "max_precip_date": max_precip_dt,
        "min_temp_c": _to_native(min_temp),
        "temp_precip_correlation": _to_native(corr),
        "average_precip_mm": _to_native(avg_precip),
        "temp_line_chart": _b64_png(temp_line_png),
        "precip_histogram": _b64_png(precip_hist_png),
    }
    return _sanitize_mapping(out)

# ---------- main handler ----------

def handle(question: str, attachments_dir: str) -> Dict[str, Any]:
    log.info(f"[generic] attachments_dir={attachments_dir}")
    files = _list_files(attachments_dir)
    log.info(f"[generic] files(initial)={files}")

    if not files and os.path.isdir(attachments_dir):
        files = _late_rescan(attachments_dir)
        log.info(f"[generic] files(rescan)={files}")

    # s3_urls.txt hint
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

    target = _prefer_from_question(question, files)
    if not target:
        target = _find_first_with_ext(files, [".csv", ".json"])
    if not target:
        target = _sniff_tabular(files)

    if not target or not os.path.exists(target):
        log.info("[generic] no target file found after all strategies.")
        return {
            "answer": "No data file provided. Upload CSV/JSON or use DuckDB SQL.",
            "attachments_seen": [os.path.basename(p) for p in files],
        }

    log.info(f"[generic] chosen target={target}")

    df = _load_table(target)
    df = _try_parse_dates(df).copy()

    # ---------- NEW: schema fast-paths ----------
    special = _maybe_weather_schema(question, df)
    if special is not None:
        return special

    # ---------- existing generic response ----------
    head_json = df.head(10).to_json(orient="records", date_format="iso")
    out: Dict[str, Any] = {
        "file": os.path.basename(target),
        "rows": int(len(df)),
        "cols": list(map(str, df.columns)),
        "head": json.loads(head_json),
    }

    s = (question or "").lower()
    try:
        png = None
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
            nums = _choose_numeric(df)
            if len(nums) >= 2:
                png = scatter(df, nums[0], nums[1], regression=True, title=None, max_bytes=100_000)

        if png is not None:
            # keep generic path as Data URI (back-compat),
            # but schema fast-path uses raw base64 above
            out["plot"] = data_uri_png(png)
    except Exception as e:
        out["plot_error"] = str(e)

    try:
        lc = s
        num_cols = df.select_dtypes(include="number").columns
        if ("average" in lc or "mean" in lc) and len(num_cols):
            out["numeric_means"] = {c: _to_native(pd.to_numeric(df[c], errors="coerce").mean()) for c in num_cols[:5]}
        if ("correlation" in lc or "corr" in lc) and len(num_cols) >= 2:
            out["pairwise_corr"] = _to_native(df[num_cols[:2]].corr().iloc[0, 1])
        if any(w in lc for w in ("min", "maximum", "max ")):
            if len(num_cols):
                out["numeric_min"] = {c: _to_native(pd.to_numeric(df[c], errors="coerce").min()) for c in num_cols[:5]}
                out["numeric_max"] = {c: _to_native(pd.to_numeric(df[c], errors="coerce").max()) for c in num_cols[:5]}
    except Exception:
        pass

    return _sanitize_mapping(out)
