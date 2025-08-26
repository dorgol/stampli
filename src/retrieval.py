#!/usr/bin/env python3
# src/retrieval.py
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional

import chromadb

# -------------------- Embedding --------------------

def embed_query_openai(text: str,
                       model: str = os.getenv("OPENAI_MODEL_EMB", "text-embedding-3-small")) -> List[float]:
    """
    Get an embedding vector for the query text using OpenAI embeddings.
    """
    try:
        from openai import OpenAI  # lazy import
    except Exception as e:
        raise RuntimeError("OpenAI SDK not installed. Try: pip install openai") from e

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI()
    resp = client.embeddings.create(model=model, input=text)
    vec = resp.data[0].embedding
    # Chroma expects a list[float]
    return [float(x) for x in vec]


# -------------------- Where filter --------------------

def _eq_or_in(field: str, values: Optional[List[Any]]) -> Optional[Dict[str, Any]]:
    """
    Turn a list/None into a Chroma equality filter. (Single -> $eq, Multi -> $in)
    """
    if not values:
        return None
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return {field: {"$in": vals}} if len(vals) > 1 else {field: {"$eq": vals[0]}}

def build_where(branch: Optional[List[str]] = None,
                country: Optional[List[str]] = None,
                month: Optional[List[int]] = None,
                season: Optional[List[str]] = None,
                year: Optional[List[int]] = None,
                rating_gte: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """
    Build a Chroma 'where' filter dict using equality / $in and numeric threshold on Rating.
    Only includes fields that are provided.
    """
    clauses: List[Dict[str, Any]] = []

    for part in (
        _eq_or_in("Branch", branch),
        _eq_or_in("Reviewer_Location", country),
        _eq_or_in("Month", month),
        _eq_or_in("Season", season),
        _eq_or_in("Year", year),
    ):
        if part:
            clauses.append(part)

    if rating_gte is not None:
        clauses.append({"Rating": {"$gte": float(rating_gte)}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


# -------------------- Retrieval --------------------

def _pick_cluster_label(meta: Dict[str, Any]) -> Optional[str]:
    """
    Inspect a metadata dict and return the first non-empty *_label value if present.
    Prefer the most 'active' convention: cluster_*_label keys.
    """
    label_keys = [k for k in meta.keys() if k.endswith("_label")]
    for k in sorted(label_keys):  # deterministic order
        v = meta.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None

def _to_float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def _truncate(txt: str, max_chars: int = 600) -> str:
    t = (txt or "").strip()
    return t if len(t) <= max_chars else (t[:max_chars] + "…")

def retrieve_snippets(chroma_path: str,
                      collection: str,
                      query_embedding: Iterable[float],
                      where: Optional[Dict[str, Any]] = None,
                      top_k: int = 8) -> List[Dict[str, Any]]:
    """
    Query a persistent Chroma collection and return a normalized list of snippets with metadata:
      - Review_ID, Branch, Reviewer_Location, Rating
      - cluster_label (if available), sentiment (if available)
      - snippet (text), distance (float)
    """
    client = chromadb.PersistentClient(path=chroma_path)
    col = client.get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})

    res = col.query(
        query_embeddings=[list(query_embedding)],
        n_results=int(top_k),
        where=where,
        include=["metadatas", "documents", "distances"],
    )

    # Normalize Chroma response into a list of dicts
    out: List[Dict[str, Any]] = []
    docs = res.get("documents", [[]])[0]
    mets = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0] or []

    for i in range(min(len(docs), len(mets))):
        meta = mets[i] or {}
        doc = docs[i] or ""
        dist = dists[i] if i < len(dists) else None

        # Pick cluster label if available
        clabel = _pick_cluster_label(meta)

        out.append({
            "Review_ID": meta.get("Review_ID"),
            "Branch": meta.get("Branch"),
            "Reviewer_Location": meta.get("Reviewer_Location"),
            "Rating": meta.get("Rating"),
            "cluster_label": clabel,
            "sentiment": meta.get("sentiment"),
            "snippet": _truncate(str(doc)),
            "distance": _to_float_or_none(dist),
        })

    return out
