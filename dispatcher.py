# dispatcher.py
from typing import Dict, Any
from handlers import generic_handler
from handlers import wikipedia_handler
from handlers import duckdb_handler
from handlers import network_handler  # << new

def route(question: str) -> str:
    """
    Classify a question into a handler category:
    - wikipedia: any mention of Wikipedia or a wikipedia.org URL
    - duckdb: contains SQL patterns, mentions 'duckdb', or uses read_parquet/read_csv_auto
    - network: graph/network analysis cues or explicit edges.csv
    - generic: everything else
    """
    s = (question or "").lower()

    # Wikipedia handler
    if "wikipedia" in s or "https://en.wikipedia.org" in s:
        return "wikipedia"

    # DuckDB handler
    if (
        ("select " in s and " from " in s)  # simple SQL heuristic
        or "duckdb" in s
        or "read_parquet(" in s
        or "read_csv_auto(" in s
    ):
        return "duckdb"

    # Network/graph handler (NEW)
    if any(k in s for k in [
        "edges.csv", "network", "graph", "degree", "centrality",
        "density", "shortest_path", "betweenness", "closeness",
        "pagerank", "connected components", "communities"
    ]):
        return "network"

    # Fallback
    return "generic"


def dispatch(question: str, attachments_dir: str) -> Dict[str, Any]:
    """
    Dispatch to the routed handler and return its result dict.
    """
    tag = route(question)

    if tag == "wikipedia":
        return wikipedia_handler.handle(question, attachments_dir)
    if tag == "duckdb":
        return duckdb_handler.handle(question, attachments_dir)
    if tag == "network":
        return network_handler.handle(question, attachments_dir)

    # default
    return generic_handler.handle(question, attachments_dir)
