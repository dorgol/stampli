#!/usr/bin/env python3
"""
retrieval.py
Query Chroma with a question embedding + metadata filters and return standardized snippets.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import chromadb
from openai import OpenAI

def embed_query_openai(text: str, model: str = "text-embedding-3-small") -> List[float]:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set.")
    client = OpenAI()
    resp = client.embeddings.create(model=model, input=[text])
    return resp.data[0].embedding

def build_where(branch=None, countries=None, months=None, seasons=None, years=None, rating_gte=None) -> Optional[Dict[str, Any]]:
    clauses: List[Dict[str, Any]] = []
    def one_or_in(field, vals):
        if not vals: return None
        vals = [v for v in vals if v is not None]
        if not vals: return None
        if len(vals) == 1: return {field: vals[0]}
        return {field: {"$in": vals}}

    if branch:    clauses.append(one_or_in("Branch", branch))
    if countries: clauses.append(one_or_in("Reviewer_Location", countries))
    if months:    clauses.append(one_or_in("Month", months))
    if seasons:   clauses.append(one_or_in("Season", seasons))
    if years:     clauses.append(one_or_in("Year", years))
    if rating_gte is not None:
        clauses.append({"Rating": {"$gte": float(rating_gte)}})

    clauses = [c for c in clauses if c]
    if not clauses: return None
    if len(clauses) == 1: return clauses[0]
    return {"$and": clauses}

def retrieve_snippets(chroma_path: str,
                      collection: str,
                      query_embedding: List[float],
                      where: Optional[Dict[str, Any]],
                      top_k: int = 12) -> List[Dict[str, Any]]:
    client = chromadb.PersistentClient(path=chroma_path)
    col = client.get_or_create_collection(collection)
    res = col.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents","metadatas","distances"]
    )
    ids   = res.get("ids", [[]])[0]
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    out = []
    for rid, doc, meta, dist in zip(ids, docs, metas, dists):
        text = doc or ""
        snip = text[:420] + ("…" if len(text) > 420 else "")
        out.append({
            "id": str(rid),
            "Review_ID": meta.get("Review_ID", rid),
            "Branch": meta.get("Branch"),
            "Reviewer_Location": meta.get("Reviewer_Location"),
            "Year": meta.get("Year"),
            "Month": meta.get("Month"),
            "Rating": meta.get("Rating"),
            "distance": float(dist),
            "snippet": snip
        })
    return out
