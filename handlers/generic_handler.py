# handlers/generic_handler.py
import os, io, re, json, base64, math, logging
from typing import Dict, Any, List, Optional, Tuple
from fastapi.responses import JSONResponse

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from io import BytesIO

log = logging.getLogger("uvicorn")

# ----------------------- filename sniffing -----------------------
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

# ----------------------- JSON sanitization -----------------------
def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(v) for v in obj]
    elif isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj

# Optional convenience if your router wants a Response directly
def handle_response(question: str, attachments_dir: str):
    cleaned = sanitize(handle(question, attachments_dir))
    return JSONResponse(content=cleaned)

# ----------------------- small, styled plots -----------------------
def _encode_png(fig, max_bytes: int = 100_000, dpi_start: int = 100) -> str:
    """
    Save figure as PNG (no data: prefix), attempt to keep under max_bytes by lowering dpi.
    """
    try:
        dpi = dpi_start
        for _ in range(6):  # a few attempts
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
            data = buf.getvalue()
            buf.close()
            if len(data) <= max_bytes:
                plt.close(fig)
                return base64.b64encode(data).decode("utf-8")
            dpi = max(50, int(dpi * 0.75))
        # give up but return something
        plt.close(fig)
        return base64.b64encode(data).decode("utf-8")
    except Exception:
        plt.close(fig)
        return ""

def _plot_line_red(dates, values, title: Optional[str] = None, max_bytes: int = 100_000) -> str:
    fig = plt.figure(figsize=(6, 3.2))
    ax = fig.add_subplot(111)
    ax.plot(dates, values, color="red")
    if title:
        ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    fig.tight_layout()
    return _encode_png(fig, max_bytes=max_bytes, dpi_start=90)

def _plot_hist_orange(values, bins: int = 20, title: Optional[str] = None, max_bytes: int = 100_000) -> str:
    fig = plt.figure(figsize=(6, 3.2))
    ax = fig.add_subplot(111)
    ax.hist(values, bins=bins, color="orange")
    if title:
        ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return _encode_png(fig, max_bytes=max_bytes, dpi_start=90)

def _plot_scatter_with_fit(x, y, title: Optional[str] = None, max_bytes: int = 100_000) -> str:
    fig = plt.figure(figsize=(6, 3.2))
    ax = fig.add_subplot(111)
    ax.scatter(x, y)
    # regression line if enough points
    try:
        if len(x) >= 2 and len(y) >= 2:
            coeffs = np.polyfit(x, y, 1)
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            ax.plot(xs, coeffs[0] * xs + coeffs[1])
    except Exception:
        pass
    if title:
        ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    return _encode_png(fig, max_bytes=max_bytes, dpi_start=90)

def _plot_bar(x, y, title: Optional[str] = None, max_bytes: int = 100_000) -> str:
    fig = plt.figure(figsize=(6.5, 3.2))
    ax = fig.add_subplot(111)
    ax.bar(x, y)
    if title:
        ax.set_title(title)
    ax.set_xlabel("Category")
    ax.set_ylabel("Value")
    fig.tight_layout()
    return _encode_png(fig, max_bytes=max_bytes, dpi_start=90)

def _plot_pie(labels, sizes, title: Optional[str] = None, max_bytes: int = 100_000) -> str:
    fig = plt.figure(figsize=(5.6, 5.6))
    ax = fig.add_subplot(111)
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    if title:
        ax.set_title(title)
    ax.axis("equal")
    fig.tight_layout()
    return _encode_png(fig, max_bytes=max_bytes, dpi_start=90)

# ----------------------- basic file helpers -----------------------
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

# ----------------------- Weather "exact keys" fast-path -----------------------
def _maybe_weather_schema(question: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Detect 'Return a JSON object with keys: average_temp_c, max_precip_date, ... temp_line_chart, precip_histogram'
    and produce EXACT keys with strict styles:
      - line: red
      - histogram: orange
      - base64 raw (no data: prefix), under 100kB
    """
    s = (question or "").lower()
    if "return a json object with keys" not in s:
        return None

    needed = {
        "average_temp_c", "max_precip_date", "min_temp_c",
        "temp_precip_correlation", "average_precip_mm",
        "temp_line_chart", "precip_histogram"
    }
    # quick check that all names appear somewhere
    keys_in_q = set(re.findall(r"[a-z0-9_]+", s))
    if not needed.issubset(keys_in_q):
        return None

    cols = {c.lower(): c for c in df.columns}
    if not {"date", "temperature_c", "precip_mm"}.issubset(cols.keys()):
        return None

    date_col = cols["date"]
    t_col = cols["temperature_c"]
    p_col = cols["precip_mm"]

    t = pd.to_numeric(df[t_col], errors="coerce")
    p = pd.to_numeric(df[p_col], errors="coerce")
    d = pd.to_datetime(df[date_col], errors="coerce")

    avg_temp = float(t.mean())
    min_temp = float(np.nanmin(t.values))
    try:
        idx = int(np.nanargmax(p.values))
        max_precip_dt = pd.to_datetime(d.iloc[idx]).isoformat()
    except Exception:
        max_precip_dt = pd.to_datetime(d.iloc[0]).isoformat()

    try:
        corr = float(pd.concat([t, p], axis=1).corr().iloc[0, 1])
        if math.isnan(corr) or math.isinf(corr):
            corr = 0.0
    except Exception:
        corr = 0.0

    avg_precip = float(p.mean())

    # exact styles
    temp_line_b64 = _plot_line_red(d, t, title=None, max_bytes=100_000)
    precip_hist_b64 = _plot_hist_orange(p.dropna().values, bins=10, title=None, max_bytes=100_000)

    out = {
        "average_temp_c": avg_temp,
        "max_precip_date": max_precip_dt,
        "min_temp_c": min_temp,
        "temp_precip_correlation": corr,
        "average_precip_mm": float(avg_precip),
        "temp_line_chart": temp_line_b64,
        "precip_histogram": precip_hist_b64,
    }
    return sanitize(out)

# ----------------------- Operator registry + phrase parser -----------------------
def _first_date_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return c
    return None

def _col_by_name_loose(df: pd.DataFrame, name_like: str) -> Optional[str]:
    if not name_like:
        return None
    nl = name_like.strip().lower()
    # exact / contains
    for c in df.columns:
        if c.lower() == nl:
            return c
    for c in df.columns:
        if nl in c.lower():
            return c
    return None

def _num_col_or_first(df: pd.DataFrame, hint: Optional[str] = None) -> Optional[str]:
    if hint:
        c = _col_by_name_loose(df, hint)
        if c is not None:
            return c
    nums = df.select_dtypes(include="number").columns.tolist()
    return nums[0] if nums else None

# --- operators ---
def op_median(df: pd.DataFrame, args: Dict[str, Any]) -> Dict[str, Any]:
    col = _num_col_or_first(df, args.get("column"))
    if not col:
        return {"median": None}
    return {"median": float(pd.to_numeric(df[col], errors="coerce").median())}

def op_stddev(df: pd.DataFrame, args: Dict[str, Any]) -> Dict[str, Any]:
    col = _num_col_or_first(df, args.get("column"))
    if not col:
        return {"stddev": None}
    return {"stddev": float(pd.to_numeric(df[col], errors="coerce").std(ddof=1))}

def op_topn(df: pd.DataFrame, args: Dict[str, Any]) -> Dict[str, Any]:
    n = int(args.get("n") or 5)
    col = _num_col_or_first(df, args.get("by"))
    if not col:
        return {"topn": []}
    tmp = df.copy()
    tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
    top = tmp.sort_values(col, ascending=False).head(n)
    return {"topn_by_"+col: json.loads(top.head(50).to_json(orient="records", date_format="iso"))}

def op_groupby(df: pd.DataFrame, args: Dict[str, Any]) -> Dict[str, Any]:
    group_col = _col_by_name_loose(df, args.get("group"))
    agg_col = _num_col_or_first(df, args.get("of"))
    fn = (args.get("fn") or "sum").lower()
    if not group_col or not agg_col:
        return {"groupby": []}
    g = df.copy()
    g[agg_col] = pd.to_numeric(g[agg_col], errors="coerce")
    if fn in ("avg", "mean"):
        res = g.groupby(group_col, dropna=False)[agg_col].mean().reset_index()
    elif fn == "count":
        res = g.groupby(group_col, dropna=False)[agg_col].count().reset_index(name=agg_col)
    elif fn == "max":
        res = g.groupby(group_col, dropna=False)[agg_col].max().reset_index()
    elif fn == "min":
        res = g.groupby(group_col, dropna=False)[agg_col].min().reset_index()
    else:  # sum
        res = g.groupby(group_col, dropna=False)[agg_col].sum().reset_index()
    return {"groupby_"+group_col+"_"+fn+"_"+agg_col: json.loads(res.head(200).to_json(orient="records", date_format="iso"))}

def op_rolling_mean(df: pd.DataFrame, args: Dict[str, Any]) -> Dict[str, Any]:
    window = int(args.get("window") or 7)
    date_col = _first_date_col(df) or df.columns[0]
    ycol = _num_col_or_first(df, args.get("column")) or (_choose_numeric(df)[0] if _choose_numeric(df) else None)
    if not ycol:
        return {"rolling_mean": []}
    t = df.copy()
    # ensure date sorted
    try:
        t[date_col] = pd.to_datetime(t[date_col], errors="coerce")
        t = t.sort_values(date_col)
    except Exception:
        pass
    t[ycol] = pd.to_numeric(t[ycol], errors="coerce")
    t["rolling_mean"] = t[ycol].rolling(window=window, min_periods=max(1, window // 2)).mean()
    # plot (line: raw + rolling)
    fig = plt.figure(figsize=(7.2, 3.6))
    ax = fig.add_subplot(111)
    try:
        ax.plot(t[date_col], t[ycol], alpha=0.5)  # base series
        ax.plot(t[date_col], t["rolling_mean"], linewidth=2)  # smooth line
    except Exception:
        ax.plot(range(len(t)), t[ycol], alpha=0.5)
        ax.plot(range(len(t)), t["rolling_mean"], linewidth=2)
    ax.set_title(f"Rolling mean ({window}) of {ycol}")
    ax.set_xlabel(str(date_col))
    ax.set_ylabel(str(ycol))
    fig.tight_layout()
    b64 = _encode_png(fig, max_bytes=100_000, dpi_start=100)
    return {
        "rolling_mean_preview": b64,
        "rolling_mean_table": json.loads(t[[date_col, "rolling_mean"]].tail(50).to_json(orient="records", date_format="iso")),
    }

OP_REGISTRY = {
    "median": op_median,
    "stddev": op_stddev,
    "topn": op_topn,
    "groupby": op_groupby,
    "rolling_mean": op_rolling_mean,
}

# Tiny phrase parser (best-effort, non-blocking)
def _parse_phrases(question: str) -> List[Tuple[str, Dict[str, Any]]]:
    s = (question or "").lower()

    ops: List[Tuple[str, Dict[str, Any]]] = []

    # median / stddev
    m = re.search(r"\bmedian(?:\s+of\s+([a-z0-9_ \-]+))?", s)
    if m:
        ops.append(("median", {"column": (m.group(1) or "").strip()}))
    m = re.search(r"\bstd(?:dev)?(?:\s+of\s+([a-z0-9_ \-]+))?", s)
    if m:
        ops.append(("stddev", {"column": (m.group(1) or "").strip()}))

    # top-N by <col>
    m = re.search(r"\btop\s*(\d+)\s*(?:by\s+([a-z0-9_ \-]+))?", s)
    if m:
        ops.append(("topn", {"n": int(m.group(1)), "by": (m.group(2) or "").strip()}))

    # group-by <col> <fn> of <col2>
    m = re.search(r"\bgroup(?:\s*by)?\s+([a-z0-9_ \-]+)\s+(sum|avg|mean|count|max|min)\s+of\s+([a-z0-9_ \-]+)", s)
    if m:
        ops.append(("groupby", {"group": m.group(1).strip(), "fn": m.group(2).strip(), "of": m.group(3).strip()}))

    # rolling mean N (or "rolling average") of <col>
    m = re.search(r"\b(rolling|moving)\s+(average|mean)\s*(?:over|of)?\s*(\d+)", s)
    if m:
        # optional column mention
        mcol = re.search(r"\b(?:of|for)\s+([a-z0-9_ \-]+)\b", s)
        ops.append(("rolling_mean", {"window": int(m.group(3)), "column": (mcol.group(1).strip() if mcol else None)}))

    return ops

def _apply_ops(df: pd.DataFrame, question: str) -> Dict[str, Any]:
    ops = _parse_phrases(question)
    out: Dict[str, Any] = {}
    for op_name, args in ops:
        fn = OP_REGISTRY.get(op_name)
        if not fn:
            continue
        try:
            res = fn(df, args)
            if isinstance(res, dict):
                out.update(sanitize(res))
        except Exception as e:
            out[f"{op_name}_error"] = str(e)
    return out

# ----------------------- main handler -----------------------
# ----------------------- main handler (core) -----------------------
def _handle_core(question: str, attachments_dir: str) -> Dict[str, Any]:
    log.info(f"[generic] attachments_dir={attachments_dir}")
    files = _list_files(attachments_dir)
    log.info(f"[generic] files(initial)={files}")

    if not files and os.path.isdir(attachments_dir):
        files = _late_rescan(attachments_dir)
        log.info(f"[generic] files(rescan)={files}")

    # special hint path (kept from earlier behavior)
    try:
        basenames = [os.path.basename(p).lower() for p in files]
        has_s3_list = "s3_urls.txt" in basenames
        has_tabular = any(p.lower().endswith((".csv", ".json")) for p in files)
        if has_s3_list and not has_tabular:
            log.info("[generic] detected s3_urls.txt without CSV/JSON; returning HC hint.")
            return sanitize({
                "answer": (
                    "Detected s3_urls.txt but no CSV/JSON to visualize here.\n"
                    "If this is the Indian High Court task, include the phrase "
                    "'Indian high court' and a read_parquet(...) mention in your question, "
                    "or let the DuckDB route run; the HC handler will read attachments/s3_urls.txt."
                ),
                "attachments_seen": basenames,
            })
    except Exception:
        pass

    target = _prefer_from_question(question, files) or \
             _find_first_with_ext(files, [".csv", ".json"]) or \
             _sniff_tabular(files)

    if not target or not os.path.exists(target):
        log.info("[generic] no target file found after all strategies.")
        return sanitize({
            "answer": "No data file provided. Upload CSV/JSON or use DuckDB SQL.",
            "attachments_seen": [os.path.basename(p) for p in files],
        })

    log.info(f"[generic] chosen target={target}")

    df = _load_table(target)
    df = _try_parse_dates(df).copy()

    # ---------- schema fast-paths ----------
    special = _maybe_weather_schema(question, df)
    if special is not None:
        return special  # already sanitized

    # ---------- generic response ----------
    head_json = df.head(10).to_json(orient="records", date_format="iso")
    out: Dict[str, Any] = {
        "file": os.path.basename(target),
        "rows": int(len(df)),
        "cols": list(map(str, df.columns)),
        "head": json.loads(head_json),
    }

    # phrase-ops enrichers (non-blocking)
    try:
        out.update(_apply_ops(df, question))
    except Exception as e:
        out["ops_error"] = str(e)

    # opportunistic visualization chosen from question keywords
    try:
        s = (question or "").lower()
        png_b64 = None

        if "pie" in s and len(df.columns) >= 2:
            png_b64 = _plot_pie(df.iloc[:20, 0].astype(str).values, pd.to_numeric(df.iloc[:20, 1], errors="coerce").values)

        elif "hist" in s:
            num_cols = df.select_dtypes(include="number").columns
            col = (num_cols[0] if len(num_cols) else df.columns[0])
            png_b64 = _plot_hist_orange(pd.to_numeric(df[col], errors="coerce").dropna().values, bins=20)

        elif "line" in s and len(df.columns) >= 2:
            xcol = _first_date_col(df) or df.columns[0]
            nums = _choose_numeric(df)
            ycol = nums[0] if nums else (df.columns[1] if len(df.columns) > 1 else df.columns[0])
            x = df[xcol]
            y = pd.to_numeric(df[ycol], errors="coerce")
            png_b64 = _plot_line_red(x, y)

        elif "bar" in s and len(df.columns) >= 2:
            nums = _choose_numeric(df)
            ycol = nums[0] if nums else df.columns[1]
            x = df.iloc[:50, 0].astype(str).values
            y = pd.to_numeric(df.iloc[:50][ycol], errors="coerce").values
            png_b64 = _plot_bar(x, y)

        else:
            nums = _choose_numeric(df)
            if len(nums) >= 2:
                x = pd.to_numeric(df[nums[0]], errors="coerce").values
                y = pd.to_numeric(df[nums[1]], errors="coerce").values
                png_b64 = _plot_scatter_with_fit(x, y)

        if png_b64:
            # keep generic path as data URI for backward-compat with your UI
            out["plot"] = f"data:image/png;base64,{png_b64}"
    except Exception as e:
        out["plot_error"] = str(e)

    # quick stats if asked
    try:
        lc = (question or "").lower()
        num_cols = df.select_dtypes(include="number").columns
        if ("average" in lc or "mean" in lc) and len(num_cols):
            out["numeric_means"] = {c: float(pd.to_numeric(df[c], errors="coerce").mean()) for c in num_cols[:5]}
        if ("correlation" in lc or "corr" in lc) and len(num_cols) >= 2:
            out["pairwise_corr"] = float(df[num_cols[:2]].corr().iloc[0, 1])
        if any(w in lc for w in ("min", "maximum", "max ")):
            if len(num_cols):
                out["numeric_min"] = {c: float(pd.to_numeric(df[c], errors="coerce").min()) for c in num_cols[:5]}
                out["numeric_max"] = {c: float(pd.to_numeric(df[c], errors="coerce").max()) for c in num_cols[:5]}
    except Exception:
        pass

    return sanitize(out)

# keep your earlier helper
def handle_response(question: str, attachments_dir: str):
    cleaned = sanitize(_handle_core(question, attachments_dir))
    return JSONResponse(content=cleaned)

# ----------------------- compatibility entrypoint -----------------------
def handle(*args, **kwargs):
    """
    Compatible with both:
      1) handle(question: str, attachments_dir: str)
      2) handle(attachments_dir: str, files: list[str] | None, form_text: str | None)
         (some routers pass (attachments_dir, files, form_text))
    Returns a plain dict (sanitized).
    """
    # Keyword-style call
    if "question" in kwargs and "attachments_dir" in kwargs:
        return _handle_core(kwargs["question"], kwargs["attachments_dir"])

    # Positional-style dispatch
    if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], str):
        # (question, attachments_dir)
        return _handle_core(args[0], args[1])

    if len(args) >= 1 and isinstance(args[0], str) and os.path.isdir(args[0]):
        # (attachments_dir, [files], [form_text])
        attachments_dir = args[0]
        form_text = ""
        if len(args) >= 3 and isinstance(args[2], str):
            form_text = args[2]
        # try to extract a question from form_text if present; else use a generic prompt
        question = form_text or "Analyze the attached data."
        return _handle_core(question, attachments_dir)

    # Fallback if the router mis-called us
    raise TypeError("generic_handler.handle called with unexpected signature")
