# dispatcher.py
from typing import Dict, Any
from handlers import generic_handler, wikipedia_handler, duckdb_handler
from handlers import network_handler
from handlers import sales_handler  # <-- add this

def route(question: str) -> str:
    s = (question or "").lower()

    if "wikipedia" in s or "https://en.wikipedia.org" in s:
        return "wikipedia"

    if (("select " in s and " from " in s)
        or "duckdb" in s
        or "read_parquet(" in s
        or "read_csv_auto(" in s):
        return "duckdb"

    # sales (match both file-name and task words)
    if any(k in s for k in [
        "sample-sales.csv", "sales.csv", "total_sales",
        "top_region", "median_sales", "sales tax", "cumulative_sales",
        "analyze `sample-sales.csv`", "analyze `sales.csv`"
    ]):
        return "sales"

    # network / graph
    if any(k in s for k in [
        "edges.csv", "network", "graph", "degree", "centrality",
        "density", "shortest_path", "betweenness", "closeness",
        "pagerank", "connected components", "communities"
    ]):
        return "network"

    return "generic"

def dispatch(question: str, attachments_dir: str) -> Dict[str, Any]:
    tag = route(question)

    if tag == "wikipedia":
        return wikipedia_handler.handle(question, attachments_dir)
    if tag == "duckdb":
        return duckdb_handler.handle(question, attachments_dir)
    if tag == "sales":
        return sales_handler.handle(question, attachments_dir)     # <-- add
    if tag == "network":
        return network_handler.handle(question, attachments_dir)

    return generic_handler.handle(question, attachments_dir)
