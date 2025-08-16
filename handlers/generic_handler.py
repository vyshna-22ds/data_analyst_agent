# app/handlers/generic_handler.py
from __future__ import annotations

import io
import json
import re
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ========== logging util ==========
def log(logger, msg: str) -> None:
    try:
        logger.info(msg)
    except Exception:
        print(msg)


# ========== base64 fig util (<= ~100kB) ==========
def _to_png_b64(fig: plt.Figure, max_kb: int = 95) -> str:
    import base64, io
    attempts = [(8, 4, 120), (7, 3.5, 110), (6, 3, 100), (6, 3, 90), (5.5, 2.8, 90), (5, 2.5, 80)]
    for w, h, dpi in attempts:
        fig.set_size_inches(w, h)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        b = buf.getvalue()
        if len(b) // 1024 <= max_kb:
            return base64.b64encode(b).decode("utf-8")
    # final downscale pass
    try:
        im = Image.open(io.BytesIO(b))
        for _ in range(6):
            im = im.resize((max(1, int(im.width * 0.9)), max(1, int(im.height * 0.9))), Image.LANCZOS)
            b2 = io.BytesIO()
            im.save(b2, format="PNG", optimize=True)
            if len(b2.getvalue()) // 1024 <= max_kb:
                return base64.b64encode(b2.getvalue()).decode("utf-8")
    except Exception:
        pass
    return base64.b64encode(b).decode("utf-8")


def _safe_iso8601(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return ""
    return ts.to_pydatetime().strftime("%Y-%m-%dT%H:%M:%S")


# ========== file helpers ==========
def _choose_target_file(paths: List[str]) -> Optional[str]:
    if not paths:
        return None
    prio = [".csv", ".json", ".parquet", ".xlsx", ".xls", ".tsv", ".txt"]
    buckets = {e: [] for e in prio}; rest = []
    for p in paths:
        ext = "." + p.rsplit(".", 1)[-1].lower() if "." in p else ""
        (buckets if ext in buckets else rest).append(p)
    for e in prio:
        if buckets[e]:
            return buckets[e][0]
    return rest[0] if rest else None


def _load_table(path: str, logger) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    try:
        if ext == "csv": return pd.read_csv(path), "csv"
        if ext == "tsv": return pd.read_csv(path, sep="\t"), "tsv"
        if ext in ("xlsx","xls"): return pd.read_excel(path), "excel"
        if ext == "parquet": return pd.read_parquet(path), "parquet"
        if ext == "json":
            try: return pd.read_json(path, orient="records"), "json"
            except Exception: return pd.read_json(path, lines=True), "json"
        if ext in ("txt","md"): return None, "text"
        return pd.read_csv(path), "csv"
    except Exception as e:
        log(logger, f"[generic] load failed for {path}: {e}")
        return None, None


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


# ========== questions.txt parsing ==========
@dataclass
class ParsedSpec:
    keys: List[str]          # exact keys to return
    body: str                # the full “Return a JSON object …” block
    answer_lines: List[str]  # lines after "Answer:" (if any)


def _extract_spec(questions: str) -> Optional[ParsedSpec]:
    if not questions:
        return None

    # Locate the spec block
    start_anchors = [r"Return a JSON object with keys:", r"Return a JSON with keys:", r"Return JSON object with keys:"]
    start = -1; anchor = None
    for a in start_anchors:
        m = re.search(re.escape(a), questions, flags=re.IGNORECASE)
        if m: start, anchor = m.start(), a; break
    if start == -1:
        return None

    # Stop before first blank line or "Answer:"
    tail = questions[start:]
    m_end = re.search(r"\n\s*\n|^Answer:", tail, flags=re.IGNORECASE | re.MULTILINE)
    end = m_end.start() if m_end else len(tail)
    keys_block = tail[:end]

    # exact keys in backticks OR bullet format "- name:"
    keys = re.findall(r"`([^`]+)`", keys_block)
    if not keys:
        keys = [m.group(1).strip() for m in re.finditer(r"-\s*([A-Za-z0-9_]+)\s*:", keys_block)]

    # answer lines (instructions)
    answer_lines = []
    m_ans = re.search(r"Answer:\s*(.*)$", questions, flags=re.IGNORECASE | re.DOTALL)
    if m_ans:
        answer_text = m_ans.group(1)
        answer_lines = [ln.strip() for ln in answer_text.splitlines() if ln.strip()]

    return ParsedSpec(keys=keys, body=keys_block, answer_lines=answer_lines) if keys else None


# ========== column mapping ==========
def _maybe_parse_dates(df: pd.DataFrame, logger) -> pd.DataFrame:
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in ["date", "time", "timestamp", "datetime"]):
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
            except Exception as e:
                log(logger, f"date-parse warn '{c}': {e}")
    return df


def _find_col(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    cols = {str(c).lower(): c for c in df.columns}
    for pat in patterns:
        rx = re.compile(pat, re.I)
        for lc, orig in cols.items():
            if rx.search(lc): return orig
    return None


def _column_synonyms(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        "date": _find_col(df, [r"^date$", r"datetime", r"timestamp", r"\btime\b"]),
        "temp": _find_col(df, [r"temp(erature)?(_?c)?\b", r"\btemp\b"]),
        "precip": _find_col(df, [r"precip(itation)?", r"\brain\b"]),
        "revenue": _find_col(df, [r"revenue|sales|amount|total(_?sales)?"]),
        "qty": _find_col(df, [r"qty|quantity|units"]),
        "price": _find_col(df, [r"price|unit[_ ]?price|avg[_ ]?price"]),
        "category": _find_col(df, [r"category|segment|class|type"]),
        "value": _find_col(df, [r"value|score|metric|measure"]),
        "x": _find_col(df, [r"^x$", r"x_value", r"feature1"]),
        "y": _find_col(df, [r"^y$", r"y_value", r"feature2"]),
    }


# ========== intent extraction ==========
@dataclass
class KeyIntent:
    name: str
    op: str                    # mean|min|max|sum|count|nunique|corr|date_of_max|date_of_min|std|median|plot_*|topn_by|groupby_agg|rolling_mean(_plot)
    col_a: Optional[str] = None
    col_b: Optional[str] = None
    color_hint: Optional[str] = None
    over_time: bool = False
    n: Optional[int] = None           # for top-N or rolling window
    group_col: Optional[str] = None   # for group-by
    agg: Optional[str] = None         # group-by agg name


def _infer_color(line: str) -> Optional[str]:
    for c in ["red", "orange", "blue", "green", "purple", "black", "gray"]:
        if re.search(rf"\b{c}\b", line, re.I):
            return c
    return None


def _infer_plot_op(key: str, line: str) -> Optional[str]:
    if re.search(r"line chart|plot .* over time|line graph", line, re.I) or re.search(r"line", key, re.I):
        return "plot_line"
    if re.search(r"\bhist(?:ogram)?\b", line, re.I) or re.search(r"histogram|hist", key, re.I):
        return "plot_hist"
    if re.search(r"\bbar\b|bar chart", line, re.I):
        return "plot_bar"
    if re.search(r"scatter", line, re.I) or re.search(r"scatter", key, re.I):
        return "plot_scatter"
    return None


def _infer_stat_op(key: str, line: str) -> Optional[str]:
    # central tendency & dispersion
    if re.search(r"\bmedian\b", line, re.I) or re.search(r"\bmedian\b", key, re.I):
        return "median"
    if re.search(r"\bstd(?:dev|\. deviation|andard deviation)?\b", line, re.I) or re.search(r"\bstd\b", key, re.I):
        return "std"
    if re.search(r"\baverage|mean\b", line, re.I) or re.search(r"\baverage|mean\b", key, re.I):
        return "mean"
    if re.search(r"\bminimum|min\b", line, re.I) or re.search(r"\bmin\b", key, re.I):
        return "min"
    if re.search(r"\bmaximum|max\b", line, re.I) or re.search(r"\bmax\b", key, re.I):
        return "max"
    if re.search(r"\btotal|sum\b", line, re.I) or re.search(r"\bsum\b", key, re.I):
        return "sum"
    if re.search(r"\bcount\b", line, re.I):
        return "count"
    if re.search(r"\bunique\b", line, re.I):
        return "nunique"
    if re.search(r"correlation|corr\b", line, re.I) or re.search(r"correlation|corr\b", key, re.I):
        return "corr"
    if re.search(r"date .* (highest|max)", line, re.I) or re.search(r"date.*max|max.*date|max_.*date", key, re.I):
        return "date_of_max"
    if re.search(r"date .* (lowest|min)", line, re.I) or re.search(r"date.*min|min.*date|min_.*date", key, re.I):
        return "date_of_min"
    # rolling average
    if re.search(r"(rolling|moving)\s+average", line, re.I) or re.search(r"(rolling|moving)_?avg", key, re.I):
        return "rolling_mean"
    # top-N
    if re.search(r"\btop\s+\d+\b", line, re.I) or re.search(r"top\d+", key, re.I):
        return "topn_by"
    # group-by
    if re.search(r"\bgroup by\b", line, re.I) or re.search(r"groupby", key, re.I):
        return "groupby_agg"
    return None


def _parse_int(value: Optional[str]) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _pick_column_from_phrase(df: pd.DataFrame, phrase: str, synonyms: Dict[str, Optional[str]]) -> Optional[str]:
    # explicit column mentions
    for c in df.columns:
        if re.search(rf"\b{re.escape(str(c))}\b", phrase, re.I): return c
    # semantic words
    if re.search(r"temp|temperature", phrase, re.I) and synonyms["temp"]: return synonyms["temp"]
    if re.search(r"precip|rain", phrase, re.I) and synonyms["precip"]: return synonyms["precip"]
    if re.search(r"revenue|sales|amount|total", phrase, re.I) and synonyms["revenue"]: return synonyms["revenue"]
    if re.search(r"qty|quantity|units", phrase, re.I) and synonyms["qty"]: return synonyms["qty"]
    if re.search(r"price", phrase, re.I) and synonyms["price"]: return synonyms["price"]
    if re.search(r"date|time|timestamp|datetime", phrase, re.I) and synonyms["date"]: return synonyms["date"]
    if synonyms["value"]: return synonyms["value"]
    return None


def _infer_intents(df: pd.DataFrame, spec: ParsedSpec) -> Dict[str, KeyIntent]:
    syn = _column_synonyms(df)
    intents: Dict[str, KeyIntent] = {key: KeyIntent(name=key, op="unknown") for key in spec.keys}

    for line in spec.answer_lines:
        # correlation
        m = re.search(r"correlation between ([^,;]+?) and ([^.,;]+)", line, re.I)
        if m:
            a_phrase, b_phrase = m.group(1).strip(), m.group(2).strip()
            a = _pick_column_from_phrase(df, a_phrase, syn) or syn["x"] or syn["temp"] or syn["revenue"]
            b = _pick_column_from_phrase(df, b_phrase, syn) or syn["y"] or syn["precip"] or syn["qty"]
            for k in spec.keys:
                if re.search(r"corr|correlation", k, re.I) and intents[k].op == "unknown":
                    intents[k] = KeyIntent(name=k, op="corr", col_a=a, col_b=b)
                    break

        # date-of-max/min
        m2 = re.search(r"date .* (highest|max) (?:for|of)?\s*([^?.]+)", line, re.I)
        if m2:
            col_phrase = m2.group(2).strip()
            col = _pick_column_from_phrase(df, col_phrase, syn) or syn["precip"] or syn["revenue"] or syn["value"]
            date_col = syn["date"]
            for k in spec.keys:
                if re.search(r"date.*max|max.*date|max_.*date", k, re.I) and intents[k].op == "unknown":
                    intents[k] = KeyIntent(name=k, op="date_of_max", col_a=col, col_b=date_col)
                    break

        m3 = re.search(r"date .* (lowest|min) (?:for|of)?\s*([^?.]+)", line, re.I)
        if m3:
            col_phrase = m3.group(2).strip()
            col = _pick_column_from_phrase(df, col_phrase, syn) or syn["value"]
            date_col = syn["date"]
            for k in spec.keys:
                if re.search(r"date.*min|min.*date|min_.*date", k, re.I) and intents[k].op == "unknown":
                    intents[k] = KeyIntent(name=k, op="date_of_min", col_a=col, col_b=date_col)
                    break

        # rolling average (window)
        m_roll = re.search(r"(rolling|moving)\s+average(?:\s*(\d+)\s*[- ]?(?:day|week|month|point|window)?)?", line, re.I)
        roll_window = _parse_int(m_roll.group(2)) if m_roll else None

        # top-N
        m_top = re.search(r"\btop\s+(\d+)\s+(.+?)\s+by\s+([A-Za-z0-9_ ]+)", line, re.I)
        if m_top:
            N = _parse_int(m_top.group(1))
            label_phrase = m_top.group(2)
            metric_phrase = m_top.group(3)
            label_col = _pick_column_from_phrase(df, label_phrase, syn) or syn["category"]
            metric_col = _pick_column_from_phrase(df, metric_phrase, syn) or syn["revenue"] or syn["value"] or syn["qty"]
            for k in spec.keys:
                if intents[k].op == "unknown" and re.search(r"top", k, re.I):
                    intents[k] = KeyIntent(name=k, op="topn_by", col_a=metric_col, group_col=label_col, n=N)
                    break

        # group-by
        m_gb = re.search(r"group by\s+([A-Za-z0-9_ ]+)\s+(?:and\s+)?(sum|mean|avg|median|min|max|count|std)\s+of\s+([A-Za-z0-9_ ]+)", line, re.I)
        if m_gb:
            g_phrase = m_gb.group(1)
            agg = m_gb.group(2).lower()
            agg = "mean" if agg == "avg" else agg
            val_phrase = m_gb.group(3)
            gcol = _pick_column_from_phrase(df, g_phrase, syn) or syn["category"]
            vcol = _pick_column_from_phrase(df, val_phrase, syn) or syn["value"] or syn["revenue"] or syn["qty"]
            for k in spec.keys:
                if intents[k].op == "unknown" and (re.search(r"groupby|group by|by_", k, re.I)):
                    intents[k] = KeyIntent(name=k, op="groupby_agg", group_col=gcol, col_a=vcol, agg=agg)
                    break

        # plot hints + generic stats per line
        for k in spec.keys:
            if intents[k].op != "unknown":
                continue
            op = _infer_stat_op(k, line) or _infer_plot_op(k, line)
            if not op:
                continue
            color = _infer_color(line)
            col = _pick_column_from_phrase(df, line, syn)
            ki = KeyIntent(name=k, op=op, col_a=col, color_hint=color, over_time=bool(re.search(r"over time", line, re.I)))
            if op == "rolling_mean":
                ki.n = roll_window or 7  # sensible default
            intents[k] = ki

    # final pass: fall back by key name
    for k, it in intents.items():
        if it.op != "unknown":
            continue
        op = _infer_stat_op(k, k) or _infer_plot_op(k, k)
        color = _infer_color(k)
        col = None
        toks = re.split(r"[_\W]+", k.lower())
        syn = _column_synonyms(df)
        if any(t in toks for t in ["temp","temperature"]) and syn["temp"]:
            col = syn["temp"]
        elif any(t in toks for t in ["precip","rain"]) and syn["precip"]:
            col = syn["precip"]
        elif any(t in toks for t in ["rev","revenue","sales","amount","total"]) and syn["revenue"]:
            col = syn["revenue"]
        elif syn["value"]:
            col = syn["value"]

        # rolling window by key name e.g. rolling7, ma30
        m_key_roll = re.search(r"(?:rolling|ma|moving)[_-]?(\d+)", k, re.I)
        roll_n = _parse_int(m_key_roll.group(1)) if m_key_roll else None
        intents[k] = KeyIntent(name=k, op=op or "unknown", col_a=col, color_hint=color, n=roll_n)

    return intents


# ========== stat/plot executors ==========
def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _exec_stat(df: pd.DataFrame, it: KeyIntent) -> Any:
    if it.op == "mean" and it.col_a:
        return float(_num(df[it.col_a]).dropna().mean())
    if it.op == "median" and it.col_a:
        return float(_num(df[it.col_a]).dropna().median())
    if it.op == "std" and it.col_a:
        return float(_num(df[it.col_a]).dropna().std(ddof=0))
    if it.op == "sum" and it.col_a:
        return float(_num(df[it.col_a]).dropna().sum())
    if it.op == "min" and it.col_a:
        return float(_num(df[it.col_a]).dropna().min())
    if it.op == "max" and it.col_a:
        return float(_num(df[it.col_a]).dropna().max())
    if it.op == "count":
        return int(df.shape[0])
    if it.op == "nunique" and it.col_a:
        return int(df[it.col_a].nunique(dropna=True))
    if it.op == "corr" and it.col_a and it.col_b and it.col_a in df and it.col_b in df:
        return float(_num(df[it.col_a]).corr(_num(df[it.col_b])))
    if it.op == "date_of_max" and it.col_a and it.col_b and it.col_a in df and it.col_b in df:
        idx = _num(df[it.col_a]).idxmax()
        if pd.isna(idx): return ""
        return _safe_iso8601(pd.to_datetime(df.loc[idx, it.col_b], errors="coerce"))
    if it.op == "date_of_min" and it.col_a and it.col_b and it.col_a in df and it.col_b in df:
        idx = _num(df[it.col_a]).idxmin()
        if pd.isna(idx): return ""
        return _safe_iso8601(pd.to_datetime(df.loc[idx, it.col_b], errors="coerce"))

    # top-N: return list of dict rows with label + metric (and date if present)
    if it.op == "topn_by" and it.col_a:
        metric = it.col_a
        N = it.n or 5
        df2 = df.copy()
        if it.group_col and it.group_col in df2.columns:
            # if label column provided, return top rows grouped by label by aggregating metric
            agg = df2.groupby(it.group_col)[metric].sum(numeric_only=True).sort_values(ascending=False).head(N)
            return [{str(it.group_col): str(k), str(metric): float(v)} for k, v in agg.items()]
        # else just top N rows by metric
        df2 = df2.sort_values(by=metric, ascending=False).head(N)
        keep = [c for c in [it.group_col, metric, _find_col(df2, [r"^date$", r"datetime", r"timestamp"])] if c]
        if not keep:
            keep = [metric]
        recs = df2[keep].to_dict(orient="records")
        # coerce numerics to float
        for r in recs:
            for k2,v in list(r.items()):
                if isinstance(v, (int,float,np.floating,np.integer)):
                    r[k2] = float(v)
                elif isinstance(v, pd.Timestamp):
                    r[k2] = _safe_iso8601(v)
        return recs

    # group-by: return list of {group: ..., value: ...}
    if it.op == "groupby_agg" and it.group_col and it.col_a:
        g, val, agg = it.group_col, it.col_a, (it.agg or "sum")
        ser = _num(df[val]) if pd.api.types.is_numeric_dtype(df[val]) is False else df[val]
        if agg == "mean": out = df.groupby(g)[val].mean(numeric_only=True)
        elif agg == "median": out = df.groupby(g)[val].median(numeric_only=True)
        elif agg == "min": out = df.groupby(g)[val].min(numeric_only=True)
        elif agg == "max": out = df.groupby(g)[val].max(numeric_only=True)
        elif agg == "count": out = df.groupby(g)[val].count()
        elif agg == "std": out = df.groupby(g)[val].std(ddof=0, numeric_only=True)
        else: out = df.groupby(g)[val].sum(numeric_only=True)
        out = out.sort_values(ascending=False)
        return [{str(g): str(k), str(val): (float(v) if pd.notna(v) else None)} for k, v in out.items()]

    # rolling mean (scalar fallback): return latest rolling value
    if it.op == "rolling_mean" and it.col_a:
        n = max(2, it.n or 7)
        s = _num(df[it.col_a]).rolling(n, min_periods=max(1, n//2)).mean()
        val = s.iloc[-1]
        return float(val) if pd.notna(val) else None

    return None


def _exec_plot(df: pd.DataFrame, it: KeyIntent) -> Optional[str]:
    # choose x
    date_col = _find_col(df, [r"^date$", r"datetime", r"timestamp", r"\btime\b"])
    if it.over_time and date_col is None:
        df = df.reset_index(drop=True).reset_index().rename(columns={"index": "idx__"})
        xcol = "idx__"
    else:
        xcol = date_col if date_col else None

    # choose y
    ycol = it.col_a
    if ycol is None:
        numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        ycol = numcols[0] if numcols else None
    if ycol is None:
        return None

    fig, ax = plt.subplots()
    try:
        if it.op == "plot_line":
            x = df[xcol] if xcol else range(len(df))
            ax.plot(x, df[ycol])
            if it.color_hint:
                for ln in ax.get_lines(): ln.set_color(it.color_hint)
            ax.set_xlabel(xcol if xcol else "index")
            ax.set_ylabel(ycol)
            ax.set_title(f"{ycol} over time" if xcol else ycol)

        elif it.op == "plot_hist":
            ax.hist(pd.to_numeric(df[ycol], errors="coerce").dropna().values, bins="auto")
            if it.color_hint:
                for p in ax.patches: p.set_facecolor(it.color_hint)
            ax.set_xlabel(ycol); ax.set_title(f"{ycol} histogram")

        elif it.op == "plot_bar":
            # if group column is specified, plot aggregated bars
            if it.group_col and it.col_a:
                agg = "sum" if not it.agg else it.agg
                grp = df.groupby(it.group_col)[it.col_a]
                if agg == "mean": s = grp.mean(numeric_only=True)
                elif agg == "median": s = grp.median(numeric_only=True)
                elif agg == "min": s = grp.min(numeric_only=True)
                elif agg == "max": s = grp.max(numeric_only=True)
                elif agg == "count": s = grp.count()
                elif agg == "std": s = grp.std(ddof=0, numeric_only=True)
                else: s = grp.sum(numeric_only=True)
                s = s.sort_values(ascending=False).head(50)
                ax.bar(s.index.astype(str), s.values)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                if it.color_hint:
                    for p in ax.patches: p.set_facecolor(it.color_hint)
                ax.set_ylabel(it.col_a)
            else:
                # bar of categories or index
                if not pd.api.types.is_numeric_dtype(df[ycol]):
                    vc = df[ycol].value_counts().head(20)
                    ax.bar(vc.index.astype(str), vc.values)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                else:
                    vals = pd.to_numeric(df[ycol], errors="coerce").dropna().head(50)
                    ax.bar(range(len(vals)), vals.values)
                if it.color_hint:
                    for p in ax.patches: p.set_facecolor(it.color_hint)
                ax.set_ylabel("count" if not pd.api.types.is_numeric_dtype(df[ycol]) else ycol)

        elif it.op == "plot_scatter":
            x = df[xcol] if xcol else range(len(df))
            ax.scatter(x, pd.to_numeric(df[ycol], errors="coerce"))
            ax.set_xlabel(xcol if xcol else "index"); ax.set_ylabel(ycol)

        elif it.op in ("rolling_mean", "rolling_mean_plot"):
            # rolling mean line plot
            n = max(2, it.n or 7)
            x = df[xcol] if xcol else range(len(df))
            y = pd.to_numeric(df[ycol], errors="coerce")
            roll = y.rolling(n, min_periods=max(1, n//2)).mean()
            ax.plot(x, roll)
            if it.color_hint:
                for ln in ax.get_lines(): ln.set_color(it.color_hint)
            ax.set_xlabel(xcol if xcol else "index"); ax.set_ylabel(f"rolling_mean_{n}({ycol})")
            ax.set_title(f"{ycol} {n}-period rolling mean")

        else:
            plt.close(fig)
            return None

        fig.tight_layout()
        b64 = _to_png_b64(fig)
        return b64
    finally:
        plt.close(fig)


# ========== generic summary fallback ==========
def _generic_summary(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    return {
        "file": target.rsplit("/", 1)[-1],
        "rows": int(df.shape[0]),
        "columns": [str(c) for c in df.columns],
        "head": df.head(10).to_dict(orient="records"),
    }


# ========== main entry ==========
def handle_generic(attachments_dir: str, files: List[str], logger, form_keys: List[str]) -> Dict[str, Any]:
    log(logger, f"[generic] attachments_dir={attachments_dir}")
    log(logger, f"[generic] files(initial)={files}")

    # pick questions.txt if present
    questions_path = next((p for p in files if p.lower().endswith("questions.txt")), None)
    questions_text = _read_text(questions_path) if questions_path else ""

    # choose target data file (exclude questions.txt)
    target = _choose_target_file([p for p in files if p != questions_path])
    if not target:
        log(logger, "[generic] no target file found after all strategies.")
        return {"error": "no target file found"}

    log(logger, f"[generic] chosen target={target}")
    df, fmt = _load_table(target, logger)
    if df is None:
        payload = {"message": "No tabular file found.", "attachments": [p.rsplit('/',1)[-1] for p in files]}
        if questions_text: payload["questions.txt"] = questions_text[:2000]
        return payload

    # basic cleaning
    df.columns = [str(c) for c in df.columns]
    df = _maybe_parse_dates(df, logger)

    spec = _extract_spec(questions_text) if questions_text else None
    if not spec:
        return {"note": "No exact key spec found; generic analysis returned.", **_generic_summary(df, target)}

    # infer intents from the spec
    intents = _infer_intents(df, spec)

    # build exact output key set in the order listed
    out: Dict[str, Any] = {}
    for k in spec.keys:
        it = intents.get(k)
        if not it or it.op == "unknown":
            out[k] = None
            continue

        # decide if plotting variant for rolling
        if it.op == "rolling_mean" and re.search(r"(plot|chart|line)", k, re.I):
            it.op = "rolling_mean_plot"

        if it.op.startswith("plot_") or it.op.endswith("_plot"):
            out[k] = _exec_plot(df, it)
        else:
            out[k] = _exec_stat(df, it)

    return out
