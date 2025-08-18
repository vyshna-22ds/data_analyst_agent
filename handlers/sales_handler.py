# CURRENT_PROJECT/handlers/sales_handler.py
from __future__ import annotations

import io
import math
from typing import Any, Dict, Optional, Union

import pandas as pd
import matplotlib.pyplot as plt

from utils.images import fig_to_base64_png_under_100kb

DataLike = Union[pd.DataFrame, bytes, bytearray, str]

def _load_df(data: DataLike) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        if isinstance(data, (bytes, bytearray)):
            buf = io.BytesIO(data)
        elif isinstance(data, str):
            buf = io.BytesIO(data.encode("utf-8"))
        else:
            raise TypeError("Unsupported input for sales handler")
        df = pd.read_csv(buf)
    # normalize headers
    cols = {c.lower().strip(): c for c in df.columns}
    # try to coerce column names
    def pick(cands):
        for c in cands:
            if c in cols:
                return cols[c]
        return None
    c_date   = pick(["date", "order_date"])
    c_region = pick(["region", "area"])
    c_sales  = pick(["sales", "amount", "revenue", "sale"])
    if c_date is None or c_region is None or c_sales is None:
        raise ValueError("Expected columns like date/region/sales")
    # types
    df[c_date] = pd.to_datetime(df[c_date], errors="coerce")
    df[c_sales] = pd.to_numeric(df[c_sales], errors="coerce")
    df = df.dropna(subset=[c_date, c_sales])
    return df.rename(columns={c_date: "date", c_region: "region", c_sales: "sales"})

def _bar_chart_by_region(df: pd.DataFrame) -> str:
    by_r = df.groupby("region", dropna=False)["sales"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(by_r.index.astype(str), by_r.values, color="blue")  # blue bars per spec
    ax.set_xlabel("Region")
    ax.set_ylabel("Total Sales")
    ax.set_title("Total Sales by Region")
    ax.grid(axis="y", alpha=0.25)
    out = fig_to_base64_png_under_100kb(fig)
    plt.close(fig)
    return out

def _cumulative_chart(df: pd.DataFrame) -> str:
    d2 = df.sort_values("date").copy()
    d2["cum_sales"] = d2["sales"].cumsum()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(d2["date"], d2["cum_sales"], linewidth=2.0, color="red")  # red line per spec
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Sales")
    ax.set_title("Cumulative Sales Over Time")
    ax.grid(alpha=0.25)
    out = fig_to_base64_png_under_100kb(fig)
    plt.close(fig)
    return out

def solve_sales(data: DataLike, question: Optional[str] = None) -> Dict[str, Any]:
    """
    Core solver for the public testcase asking:
    - total_sales
    - top_region
    - day_sales_correlation (day-of-month vs sales)
    - bar_chart (base64 PNG under 100kB)
    - median_sales
    - total_sales_tax (10% rate)
    - cumulative_sales_chart (base64 PNG under 100kB)
    """
    df = _load_df(data)

    total_sales = float(df["sales"].sum())
    top_region = (
        df.groupby("region", dropna=False)["sales"].sum().sort_values(ascending=False).index[0]
        if not df.empty else None
    )
    # correlation day-of-month vs sales
    df["_day"] = df["date"].dt.day
    try:
        day_sales_correlation = float(df[["_day", "sales"]].corr().iloc[0, 1])
        if math.isnan(day_sales_correlation):
            day_sales_correlation = 0.0
    except Exception:
        day_sales_correlation = 0.0

    median_sales = float(df["sales"].median()) if not df.empty else 0.0
    total_sales_tax = total_sales * 0.10  # 10% as per prompt

    bar_chart = _bar_chart_by_region(df)
    cumulative_sales_chart = _cumulative_chart(df)

    # IMPORTANT: wrap so the app can unwrap to answer-only when needed
    return {"__answer_only__": {
        "total_sales": total_sales,
        "top_region": str(top_region) if top_region is not None else None,
        "day_sales_correlation": day_sales_correlation,
        "bar_chart": bar_chart,
        "median_sales": median_sales,
        "total_sales_tax": total_sales_tax,
        "cumulative_sales_chart": cumulative_sales_chart,
    }}
