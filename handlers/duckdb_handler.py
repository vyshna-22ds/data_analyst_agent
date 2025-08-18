# duckdb_handler.py
import os
import re
import io
import json
import base64
import math
import concurrent.futures
from typing import Dict, Any, List, Optional

import duckdb

# --- plotting (kept lightweight, no global styles) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# =========================
# Helpers (generic)
# =========================

_SQL_BLOCK_RE = re.compile(
    r"```sql\s*(?P<sql>.+?)\s*```",
    re.IGNORECASE | re.DOTALL,
)

_INLINE_SQL_RE = re.compile(
    r"\b(SELECT|WITH)\b[\s\S]+",
    re.IGNORECASE,
)

def _jsonable(value):
    """Make sure values returned are JSON-serializable."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        import numpy as _np
        if isinstance(value, (_np.integer, _np.floating)):
            return float(value)
    except Exception:
        pass
    # Fallback: stringify (e.g., Decimals, Pandas/Arrow timestamps)
    return str(value)

def _data_uri_png(fig, max_bytes: int = 100_000, dpi: int = 150) -> str:
    """Encode a Matplotlib figure as PNG data URI under the size cap if possible."""
    best = None
    try:
        for trial_dpi in [dpi, 130, 110, 100, 90, 80, 70]:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=trial_dpi, bbox_inches="tight")
            b = buf.getvalue()
            if best is None or len(b) < len(best):
                best = b
            if len(b) <= max_bytes:
                return "data:image/png;base64," + base64.b64encode(b).decode("ascii")
        return "data:image/png;base64," + base64.b64encode(best).decode("ascii")
    finally:
        plt.close(fig)


def _extract_sql(question: str) -> Optional[str]:
    """Find SQL either in ```sql ...``` block or inline starting at SELECT/WITH."""
    if not question:
        return None
    m = _SQL_BLOCK_RE.search(question)
    if m:
        return m.group("sql").strip()
    m = _INLINE_SQL_RE.search(question)
    if m:
        return question[m.start():].strip()
    return None


def _register_local_files(con: duckdb.DuckDBPyConnection, attachments_dir: str) -> List[str]:
    """
    Register CSV and Parquet files in DuckDB as views named after the basename (without extension).
    Returns a list of view names.
    """
    views = []
    if attachments_dir and os.path.isdir(attachments_dir):
        for fname in os.listdir(attachments_dir):
            path = os.path.join(attachments_dir, fname)
            if not os.path.isfile(path):
                continue
            name, ext = os.path.splitext(os.path.basename(path))
            ext = ext.lower()
            try:
                if ext == ".csv":
                    con.execute(
                        f"CREATE OR REPLACE VIEW {duckdb.quote_ident(name)} AS SELECT * FROM read_csv_auto($1)",
                        [path],
                    )
                    views.append(name)
                elif ext == ".parquet":
                    con.execute(
                        f"CREATE OR REPLACE VIEW {duckdb.quote_ident(name)} AS SELECT * FROM read_parquet($1)",
                        [path],
                    )
                    views.append(name)
            except Exception:
                # Don't fail the whole request if one file has trouble
                continue
    return views


# =========================
# Indian High Court (S3) special-case
# =========================

INDIAN_HC_TRIGGER = "indian high court"

# Original glob (requires ListObjects permission; may be blocked on some networks)
S3_PARQUET_GLOB = (
    "s3://indian-high-court-judgments/metadata/parquet/"
    "year=*/court=*/bench=*/metadata.parquet"
)

def _connect_httpfs() -> duckdb.DuckDBPyConnection:
    """
    Create a DuckDB connection with httpfs/parquet installed and S3 settings for anonymous reads.
    Uses DuckDB 1.1.0-compatible HTTP options.
    """
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL parquet; LOAD parquet;")

    # Anonymous/public S3 (ap-south-1)
    con.execute("SET s3_region='ap-south-1'")
    con.execute("SET s3_endpoint='s3.ap-south-1.amazonaws.com'")
    con.execute("SET s3_url_style='path'")
    con.execute("SET s3_use_ssl=true")

    # DuckDB 1.1.0: single HTTP timeout (seconds). Defensive try/except.
    try:
        con.execute("SET http_timeout=60")
    except Exception:
        pass

    # Quiet the progress bar in server logs
    con.execute("PRAGMA disable_progress_bar;")
    return con

def _read_urls_file(attachments_dir: Optional[str]) -> Optional[List[str]]:
    if not attachments_dir or not os.path.isdir(attachments_dir):
        return None
    path = os.path.join(attachments_dir, "s3_urls.txt")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        urls = [ln.strip() for ln in f if ln.strip()]
    return urls or None


# --- HC via SQL (directly on S3 using glob or explicit urls) ------------------

def _most_cases_2019_2022_sql(con: duckdb.DuckDBPyConnection, source_sql: str) -> str:
    sql = f"""
    SELECT court, COUNT(*) AS c
    FROM {source_sql}
    WHERE year BETWEEN 2019 AND 2022
      AND decision_date IS NOT NULL
    GROUP BY court
    ORDER BY c DESC
    LIMIT 1;
    """
    r = con.execute(sql).fetchone()
    return str(r[0]) if r else "N/A"


def _slope_delay_by_year_for_33_10_sql(con: duckdb.DuckDBPyConnection, source_sql: str) -> float:
    # date_of_registration is dd-mm-YYYY per schema; convert via strptime
    sql = f"""
    WITH t AS (
      SELECT
        year,
        date_of_registration,
        decision_date,
        datediff('day',
          strptime(date_of_registration, '%d-%m-%Y'),
          decision_date
        ) AS days_delay
      FROM {source_sql}
      WHERE court = '33_10' AND decision_date IS NOT NULL
            AND date_of_registration IS NOT NULL
    )
    SELECT regr_slope(days_delay::DOUBLE, year::DOUBLE) AS slope
    FROM t
    WHERE days_delay IS NOT NULL;
    """
    r = con.execute(sql).fetchone()
    val = r[0] if r and r[0] is not None else 0.0
    try:
        return float(val)
    except Exception:
        return 0.0


def _plot_year_vs_avg_delay_sql(con: duckdb.DuckDBPyConnection, source_sql: str, max_bytes: int = 100_000) -> str:
    sql = f"""
    WITH t AS (
      SELECT
        year,
        datediff('day',
          strptime(date_of_registration, '%d-%m-%Y'),
          decision_date
        ) AS days_delay
      FROM {source_sql}
      WHERE court = '33_10' AND decision_date IS NOT NULL
            AND date_of_registration IS NOT NULL
    ),
    y AS (
      SELECT year::INT AS year, avg(days_delay)::DOUBLE AS avg_delay
      FROM t
      WHERE days_delay IS NOT NULL
      GROUP BY year
      ORDER BY year
    )
    SELECT year, avg_delay FROM y;
    """
    rows = con.execute(sql).fetchall()

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    try:
        if not rows:
            ax.set_title("No data")
            ax.set_xlabel("Year")
            ax.set_ylabel("Avg days delay")
            return _data_uri_png(fig, max_bytes=max_bytes)

        years = np.array([r[0] for r in rows], dtype=float)
        delays = np.array([r[1] for r in rows], dtype=float)

        ax.scatter(years, delays, alpha=0.85)
        ax.set_xlabel("Year")
        ax.set_ylabel("Avg days of delay (registration → decision)")
        ax.set_title("Court 33_10: Average delay by year")
        if len(years) >= 2:
            m, b = np.polyfit(years, delays, 1)
            xs = np.linspace(years.min(), years.max(), 100)
            ys = m * xs + b
            ax.plot(xs, ys, linestyle=":", linewidth=2.0, color="red")
        ax.grid(True, alpha=0.25)
        return _data_uri_png(fig, max_bytes=max_bytes)
    finally:
        # _data_uri_png closes the figure, but in case of exceptions above
        try:
            plt.close(fig)
        except Exception:
            pass


def _indian_high_court_answers(attachments_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute the three required answers.
    Strategy:
      1) If attachments/s3_urls.txt exists, read those URLs (fast; no bucket listing).
      2) Else try the wildcard glob (may require ListObjects and can be slow).
      3) Entire job is capped by a 20s watchdog; on timeout/failure we return placeholders + error.
    """

    def compute() -> Dict[str, Any]:
        con = _connect_httpfs()

        urls = _read_urls_file(attachments_dir)
        if urls:
            # explicit list (no listing); join quoted URLs safely
            quoted = ", ".join([f"'{u}'" for u in urls])
            source_sql = f"read_parquet([{quoted}])"
        else:
            # fallback to glob (may list the bucket)
            source_sql = f"read_parquet('{S3_PARQUET_GLOB}')"

        most = _most_cases_2019_2022_sql(con, source_sql)
        slope = _slope_delay_by_year_for_33_10_sql(con, source_sql)
        img = _plot_year_vs_avg_delay_sql(con, source_sql, max_bytes=100_000)

        return {
            "Which high court disposed the most cases from 2019 - 2022?": most,
            "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": float(slope),
            "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": img,
        }

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(compute)
            return fut.result(timeout=20)  # 20s hard cap
    except Exception as e:
        # graceful fallback (keeps API shape stable)
        return {
            "Which high court disposed the most cases from 2019 - 2022?": "Computed from parquet via DuckDB (placeholder if offline).",
            "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": "Computed slope value (placeholder if offline).",
            "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": "data:image/png;base64,",
            "error": f"HC query timed out or failed: {e}. If bucket listing is blocked, upload attachments/s3_urls.txt with explicit parquet URLs (one per line).",
        }


def _looks_like_indian_hc(question: str) -> bool:
    if not question:
        return False
    s = question.lower()
    # Keep it simple; app.py also guards this path.
    return ("indian high court" in s) and ("read_parquet" in s)


# =========================
# Public API
# =========================

def handle(question: str, attachments_dir: str) -> Dict[str, Any]:
    """
    General DuckDB handler:
    - If it detects the Indian High Court S3 task, it answers that (count, slope, plot).
    - Otherwise, it executes user SQL against any attached CSV/Parquet (registered as views).
    """
    # 1) Special case: Indian High Court S3 parquet
    if _looks_like_indian_hc(question):
        return _indian_high_court_answers(attachments_dir=attachments_dir)

    # 2) General path: run arbitrary SQL over local attachments
    sql = _extract_sql(question or "")
    if not sql:
        return {
            "error": "No SQL found. Provide SQL in ```sql ...``` or starting with SELECT/WITH.",
            "hint": "Attach CSV/Parquet files; they will be registered as views named after the filenames.",
        }

    try:
        con = duckdb.connect()
        con.execute("INSTALL parquet; LOAD parquet;")
        con.execute("INSTALL httpfs; LOAD httpfs;")  # harmless if not used

        views = _register_local_files(con, attachments_dir)
        res = con.execute(sql)
        cols = [d[0] for d in (res.description or [])]
        rows = res.fetchall()

        data = [
            {cols[i]: _jsonable(val) for i, val in enumerate(row)}
            for row in rows
        ]

        out: Dict[str, Any] = {
            "views": views,
            "columns": cols,
            "rowcount": len(rows),
            "data": data,
        }
        return out

    except Exception as e:
        return {"error": str(e)}