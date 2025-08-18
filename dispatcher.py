# dispatcher.py
from typing import Dict, Any

# dispatcher.py
def route(q: str) -> str:
    s = q.lower()
    if "use the undirected network" in s or "degree_histogram" in s or "edges.csv" in s:
        return "network"
    if "analyze" in s and "sales" in s or "sample-sales.csv" in s or "cumulative sales" in s:
        return "sales"
    if "wikipedia" in s:
        return "wikipedia"
    if "duckdb" in s or "read_parquet" in s:
        return "duckdb"
    return "generic"


def dispatch(question: str, attachments_dir: str) -> Dict[str, Any]:
    """
    Dispatch to the routed handler and return its result dict.
    Lazy-import handlers here to avoid circular imports at module import time.
    """
    tag = route(question)

    if tag == "wikipedia":
        from handlers import wikipedia_handler
        return wikipedia_handler.handle(question, attachments_dir)

    if tag == "duckdb":
        from handlers import duckdb_handler
        return duckdb_handler.handle(question, attachments_dir)

    if tag == "sales":
        from handlers import sales_handler
        return sales_handler.handle(question, attachments_dir)

    if tag == "network":
        from handlers import network_handler
        return network_handler.handle(question, attachments_dir)

    # default
    from handlers import generic_handler
    return generic_handler.handle(question, attachments_dir)
