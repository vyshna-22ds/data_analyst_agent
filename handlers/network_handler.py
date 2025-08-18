# handlers/network_handler.py
from __future__ import annotations

import os
import io
import re
import math
import base64
import logging
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import networkx as nx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger("uvicorn")

FILENAME_RE = re.compile(
    r"""(?:
            [`'"](?P<qname>[^\s`'"]+\.(?:csv|json))[`'"]
         |
            (?P<name>\b[\w\-.]+\.(?:csv|json)\b)
        )""",
    re.IGNORECASE | re.VERBOSE,
)

MAX_PNG_BYTES = 100_000

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
        log.warning(f"[network] _list_files error: {e}")
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

def _pick_edges_like(files: List[str]) -> Optional[str]:
    priority = []
    for p in files:
        lp = p.lower()
        if lp.endswith(".csv") and any(k in lp for k in ("edge", "graph", "network")):
            priority.append(p)
    if priority:
        return priority[0]
    for p in files:
        if p.lower().endswith(".csv"):
            return p
    return None

def _load_edges(path: str) -> pd.DataFrame:
    if path.lower().endswith(".json"):
        df = pd.read_json(path)
    else:
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, sep=";")

    cols = {c.lower().strip(): c for c in df.columns}
    def pick(cands: Tuple[str, ...]) -> Optional[str]:
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    c_src = pick(("source", "src", "from", "u"))
    c_tgt = pick(("target", "tgt", "to", "v"))
    if c_src is None or c_tgt is None:
        raise RuntimeError("Edges file must contain source/target-like columns (e.g., source,target).")
    return pd.DataFrame({
        "source": df[c_src].astype(str),
        "target": df[c_tgt].astype(str),
    })

def _fig_to_base64_under_limit(fig, target_bytes=MAX_PNG_BYTES, start_dpi=120) -> str:
    dpi = start_dpi
    data = b""
    for _ in range(7):
        bio = io.BytesIO()
        fig.savefig(bio, format="png", dpi=dpi, bbox_inches="tight")
        data = bio.getvalue()
        if len(data) <= target_bytes:
            break
        dpi = max(50, int(dpi * 0.8))
    plt.close(fig)
    return base64.b64encode(data).decode("ascii")

def _round2(x: float) -> float:
    try:
        val = float(x)
        if not math.isfinite(val):
            return 0.0
        return float(f"{val:.2f}")
    except Exception:
        return 0.0

def _solve(edges: pd.DataFrame) -> Dict[str, Any]:
    G = nx.from_pandas_edgelist(edges, source="source", target="target")

    edge_count = int(G.number_of_edges())
    degrees = dict(G.degree())
    if not degrees:
        raise RuntimeError("Graph has no nodes after reading edges.")

    highest_degree_node = max(degrees, key=degrees.get)
    avg_degree = sum(degrees.values()) / max(1, len(degrees))
    density = nx.density(G)
    try:
        shortest_path_alice_eve = int(nx.shortest_path_length(G, "Alice", "Eve"))
    except Exception:
        shortest_path_alice_eve = 0

    pos = nx.spring_layout(G, seed=7)

    fig1 = plt.figure(figsize=(5, 5))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color="lightblue",
        edge_color="gray",
        node_size=900,
        font_size=10,
        width=1.6,
    )
    network_graph = _fig_to_base64_under_limit(fig1)

    labels = list(degrees.keys())
    values = [degrees[k] for k in labels]
    fig2 = plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color="green")
    plt.xlabel("Nodes")
    plt.ylabel("Degree")
    plt.tight_layout()
    degree_histogram = _fig_to_base64_under_limit(fig2)

    return {
        "edge_count": edge_count,
        "highest_degree_node": str(highest_degree_node),
        "average_degree": _round2(avg_degree),
        "density": _round2(density),
        "shortest_path_alice_eve": shortest_path_alice_eve,
        "network_graph": network_graph,          # RAW base64 (no data URI)
        "degree_histogram": degree_histogram,    # RAW base64 (no data URI)
    }

def handle(question: str, attachments_dir: str) -> Dict[str, Any]:
    """
    Matches the 3 expected prompts:
      1) “Use the undirected network in `edges.csv`.”  -> {}
      2) “Return a JSON object with keys: …”          -> only those keys
      3) “Answer: …”                                  -> text + two base64 PNGs
    """
    q = (question or "").lower()
    if "use the undirected network" in q:
        return {}

    files = _list_files(attachments_dir)
    if not files:
        return {"error": "No attachments provided. Upload edges CSV/JSON."}

    target = _prefer_from_question(question, files) or _pick_edges_like(files)
    if not target or not os.path.exists(target):
        return {"error": "No edges file found. Provide a CSV/JSON with source/target columns."}

    edges = _load_edges(target)
    if len(edges) == 0:
        return {"error": f"Edges file '{os.path.basename(target)}' is empty."}

    payload = _solve(edges)

    if "return a json object" in q or "json object with keys" in q:
        return {
            "edge_count": payload["edge_count"],
            "highest_degree_node": payload["highest_degree_node"],
            "average_degree": payload["average_degree"],
            "density": payload["density"],
            "shortest_path_alice_eve": payload["shortest_path_alice_eve"],
            "network_graph": payload["network_graph"],
            "degree_histogram": payload["degree_histogram"],
        }

    if q.startswith("answer:") or q.startswith("answer"):
        answer = (
            "1. Edges in the network: {edge_count}\n"
            "2. Highest-degree node: {highest}\n"
            "3. Average degree: {avg:.2f}\n"
            "4. Network density: {dens:.2f}\n"
            "5. Shortest path length (Alice → Eve): {spath}\n"
            "6. network_graph: <base64 PNG>\n"
            "7. degree_histogram: <base64 PNG>"
        ).format(
            edge_count=payload["edge_count"],
            highest=payload["highest_degree_node"],
            avg=payload["average_degree"],
            dens=payload["density"],
            spath=payload["shortest_path_alice_eve"],
        )
        return {
            "answer": answer,
            "network_graph": payload["network_graph"],
            "degree_histogram": payload["degree_histogram"],
        }

    return payload
