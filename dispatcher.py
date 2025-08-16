def route(question: str) -> str:
    """
    Classify a question into a handler category:
    - wikipedia: any mention of Wikipedia or a wikipedia.org URL
    - duckdb: contains SQL patterns, mentions 'duckdb', or uses read_parquet/read_csv_auto
    - generic: everything else
    """
    s = (question or "").lower()

    # Wikipedia handler
    if "wikipedia" in s or "https://en.wikipedia.org" in s:
        return "wikipedia"

    # DuckDB handler
    if (
        ("select " in s and " from " in s)
        or "duckdb" in s
        or "read_parquet(" in s
        or "read_csv_auto(" in s
    ):
        return "duckdb"

    # Fallback
    return "generic"
