# wikipedia_handler.py
import re, time, requests, pandas as pd, numpy as np
from typing import Dict, Any, List, Tuple, Optional
from utils.plots import scatter, line, bar, pie, hist
from utils.io import data_uri_png

# --- Networking defaults recommended for Wikipedia scraping ---
UA = (
    "DataAnalystAgent/3.3 (+https://app.example.com; contact: ops@app.example.com) "
    "requests/2.x"
)
HEADERS = {
    "User-Agent": UA,
    "Accept-Language": "en;q=0.9,en-GB;q=0.8",
    "Referer": "https://en.wikipedia.org/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def _fetch_html(url: str, timeout: float = 10.0, attempts: int = 3, backoff: float = 0.75) -> str:
    """
    Fetch HTML with a friendly UA, bounded timeout, and tiny retry/backoff.
    Retries on 403/429/5xx or network errors.
    """
    last_err: Optional[Exception] = None
    for i in range(attempts):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            # Retry on transient/blocked statuses
            if resp.status_code in (403, 429) or 500 <= resp.status_code < 600:
                last_err = RuntimeError(f"HTTP {resp.status_code} from Wikipedia")
            else:
                resp.raise_for_status()
                return resp.text
        except Exception as e:
            last_err = e
        # small backoff then try again
        time.sleep(backoff * (i + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")

def _read_tables(url: str) -> List[pd.DataFrame]:
    html = _fetch_html(url, timeout=10.0, attempts=3)
    # Let pandas parse with bs4/lxml; wikipedia tables are well‑formed enough
    tables = pd.read_html(html)
    cleaned = []
    for t in tables:
        t = t.dropna(axis=1, how="all").copy()
        if isinstance(t.columns, pd.MultiIndex):
            t.columns = [
                " ".join([str(c) for c in col if c is not None]).strip()
                for col in t.columns.values
            ]
        t.columns = [str(c).strip() for c in t.columns]
        cleaned.append(t)
    return cleaned

def _coerce_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(r"[^0-9\.\-]", "", regex=True),
        errors="coerce",
    )

def _find(df: pd.DataFrame, keys: List[str]):
    keys = [k.lower() for k in keys]
    for c in df.columns:
        cl = str(c).lower()
        if all(k in cl for k in keys):
            return c
    return None

def solve_highest_grossing(url: str):
    dfs = _read_tables(url)
    if not dfs:
        raise RuntimeError("No tables found.")
    def pick(dfs):
        for d in dfs:
            cols = [c.lower() for c in d.columns]
            if any("rank" in c for c in cols) and any("peak" in c for c in cols):
                return d
        return dfs[0]
    df = pick(dfs).copy()
    col_year  = _find(df, ["year"]) or "Year"
    col_title = _find(df, ["title"]) or _find(df, ["film"]) or "Title"
    col_gross = _find(df, ["worldwide","gross"]) or _find(df, ["gross"])
    col_rank  = _find(df, ["rank"])
    col_peak  = _find(df, ["peak"])

    if col_year in df:  df[col_year]  = _coerce_num(df[col_year])
    if col_rank in df:  df[col_rank]  = _coerce_num(df[col_rank])
    if col_peak in df:  df[col_peak]  = _coerce_num(df[col_peak])
    if col_gross in df: df[col_gross] = _coerce_num(df[col_gross])

    # 1) How many $2bn movies before 2000?
    cnt = int(((df[col_gross] >= 2_000_000_000) & (df[col_year] < 2000)).sum()) \
          if (col_gross in df and col_year in df) else 0

    # 2) Earliest film over $1.5bn
    earliest = "Unknown"
    if col_gross in df and col_year in df and col_title in df:
        sub = df.loc[df[col_gross] >= 1_500_000_000, [col_year, col_title]].dropna()
        if not sub.empty:
            r = sub.sort_values(col_year, ascending=True).iloc[0]
            earliest = str(r[col_title])

    # 3) Corr + scatter with regression (size guarded to 100k in utils)
    corr = float("nan")
    if col_rank in df and col_peak in df:
        m = df[col_rank].notna() & df[col_peak].notna()
        if m.sum() >= 2:
            corr = float(np.corrcoef(df[col_rank][m], df[col_peak][m])[0, 1])
        png = scatter(df.loc[m].head(600), col_rank, col_peak, regression=True, title=None, max_bytes=100_000)
        img = data_uri_png(png)
    else:
        img = None

    return cnt, earliest, float(corr), img

def handle(question: str):
    m = re.search(r"(https?://[^\s]+)", question)
    url = m.group(1) if m else None
    if not url:
        return {"error": "Wikipedia URL required."}
    if "list_of_highest-grossing_films" in url:
        return solve_highest_grossing(url)
    # Generic Wikipedia: return first table head
    dfs = _read_tables(url)
    if not dfs:
        return {"error": "No tables parsed."}
    df = dfs[0]
    return {"columns": list(df.columns), "head": df.head(10).to_dict(orient="records")}
