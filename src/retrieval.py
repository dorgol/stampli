#!/usr/bin/env python3
"""
retrieval.py
Query Chroma with a question embedding + metadata filters and return standardized snippets.

- Works with enriched metadata:
  * cluster_* (e.g., cluster_mbk / cluster_micro / cluster_kmeans / cluster_hdbscan)
  * *_label (e.g., cluster_mbk_label ...)
  * sentiment (optional)
  * keywords_str (optional)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import os
import chromadb
from openai import OpenAI

# ---------- Embedding ----------
def embed_query_openai(text: str, model: str = "text-embedding-3-small") -> List[float]:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set.")
    client = OpenAI()
    resp = client.embeddings.create(model=model, input=[text])
    return resp.data[0].embedding

# ---------- Helpers for flexible filters over dynamic columns ----------
_CLUSTER_ID_KEYS = ["cluster_mbk", "cluster_micro", "cluster_kmeans", "cluster_hdbscan"]
_CLUSTER_LABEL_KEYS = [f"{k}_label" for k in _CLUSTER_ID_KEYS]

def _one_or_in(field: str, vals):
    if not vals: return None
    vals = [v for v in vals if v is not None]
    if not vals: return None
    if len(vals) == 1: return {field: vals[0]}
    return {field: {"$in": vals}}

def _or_contains(fields: List[str], needle: Optional[str]) -> Optional[Dict[str, Any]]:
    if not needle:
        return None
    return {"$or": [{f: {"$contains": needle}} for f in fields]}

def _or_equals(fields, value):
    if value is None: return None
    fields = [f for f in fields if f]
    if not fields: return None
    if len(fields) == 1: return {fields[0]: value}
    return {"$or": [{f: value} for f in fields]}

def _compact_and(clauses):
    clauses = [c for c in clauses if c]
    if not clauses: return None
    if len(clauses) == 1: return clauses[0]
    return {"$and": clauses}

def build_where(
    branch=None, countries=None, months=None, seasons=None, years=None, rating_gte=None,
    cluster_label_in=None,   # NEW: list of exact labels (not "contains")
    cluster_id=None,
    sentiment_in=None,
    keywords_in=None         # if you keep tokens/keywords_str exact matches
):
    clauses = []
    v = _one_or_in("Branch", branch);                v and clauses.append(v)
    v = _one_or_in("Reviewer_Location", countries);  v and clauses.append(v)
    v = _one_or_in("Month", months);                 v and clauses.append(v)
    v = _one_or_in("Season", seasons);               v and clauses.append(v)
    v = _one_or_in("Year", years);                   v and clauses.append(v)
    if rating_gte is not None:
        clauses.append({"Rating": {"$gte": float(rating_gte)}})

    # cluster id across any known id columns
    v = _or_equals(_CLUSTER_ID_KEYS, int(cluster_id)) if cluster_id is not None else None
    v and clauses.append(v)

    # cluster label exact match across any known label columns
    if cluster_label_in:
        ors = [{lbl_col: {"$in": cluster_label_in}} for lbl_col in _CLUSTER_LABEL_KEYS]
        clauses.append({"$or": ors} if len(ors) > 1 else ors[0])

    # sentiment exact match
    v = _one_or_in("sentiment", sentiment_in); v and clauses.append(v)

    # keywords exact match (only if you indexed tokens/keywords_str as exact strings)
    if keywords_in:
        clauses.append({"keywords_str": {"$in": keywords_in}})

    return _compact_and(clauses)


# ---------- Retrieval ----------
def retrieve_snippets(
    chroma_path: str,
    collection: str,
    query_embedding: List[float],
    where: Optional[Dict[str, Any]],
    top_k: int = 12,
    snippet_chars: int = 420,
) -> List[Dict[str, Any]]:
    """
    Execute the query against Chroma and return normalized snippets including enriched metadata.
    """
    client = chromadb.PersistentClient(path=chroma_path)
    col = client.get_or_create_collection(collection)

    res = col.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    ids   = res.get("ids", [[]])[0]
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    out: List[Dict[str, Any]] = []
    for rid, doc, meta, dist in zip(ids, docs, metas, dists):
        text = doc or ""
        snip = text[:snippet_chars] + ("…" if len(text) > snippet_chars else "")

        # pick whichever cluster id/label exists in this record
        cluster_id = None
        cluster_label = None
        for k in _CLUSTER_ID_KEYS:
            if k in meta:
                cluster_id = meta.get(k)
                break
        for k in _CLUSTER_LABEL_KEYS:
            if k in meta:
                cluster_label = meta.get(k)
                break

        out.append({
            "id": str(rid),
            "Review_ID": meta.get("Review_ID", rid),
            "Branch": meta.get("Branch"),
            "Reviewer_Location": meta.get("Reviewer_Location"),
            "Year": meta.get("Year"),
            "Month": meta.get("Month"),
            "Season": meta.get("Season"),
            "Rating": meta.get("Rating"),
            "cluster_id": cluster_id,
            "cluster_label": cluster_label,
            "sentiment": meta.get("sentiment"),
            "keywords_str": meta.get("keywords_str"),
            "distance": float(dist),
            "snippet": snip,
        })
    return out
