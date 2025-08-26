#!/usr/bin/env python3
"""
retrieval.py
Query the Disney reviews index with semantic search + metadata filters.

Example:
  python src/retrieval.py --question "What do visitors from Australia say about Disneyland in HongKong?" `
    --chroma_path data/index_tmp/chroma_db --branch Disneyland_HongKong --country Australia --top_k 5
"""

import argparse
import os
from typing import Dict, Any, List, Optional
import chromadb

# --- Embedding the query with OpenAI ---
def embed_query_openai(text: str, model: str = "text-embedding-3-small") -> List[float]:
    from openai import OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set.")
    client = OpenAI()
    resp = client.embeddings.create(model=model, input=[text])
    return resp.data[0].embedding

# --- Build metadata filter ("where") for Chroma ---
def build_where(branch=None, countries=None, months=None, seasons=None, years=None, rating_gte=None):
    clauses = []

    def one_or_in(field, values):
        values = [v for v in values if v is not None]
        if not values:
            return None
        if len(values) == 1:
            return {field: values[0]}
        return {field: {"$in": values}}

    if branch:
        c = one_or_in("Branch", branch);            clauses.append(c)
    if countries:
        c = one_or_in("Reviewer_Location", countries); clauses.append(c)
    if months:
        c = one_or_in("Month", months);            clauses.append(c)
    if seasons:
        c = one_or_in("Season", seasons);          clauses.append(c)
    if years:
        c = one_or_in("Year", years);              clauses.append(c)
    if rating_gte is not None:
        clauses.append({"Rating": {"$gte": float(rating_gte)}})

    clauses = [c for c in clauses if c is not None]
    if not clauses:
        return None
    if len(clauses) == 1:
        # single condition: pass it directly (no $and) to keep Chroma happy
        return clauses[0]
    return {"$and": clauses}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--chroma_path", default="data/index/chroma_db")
    ap.add_argument("--collection", default="reviews")
    ap.add_argument("--oa_model", default="text-embedding-3-small")
    ap.add_argument("--top_k", type=int, default=15)
    ap.add_argument("--branch", nargs="*", help="Disneyland_HongKong / Disneyland_Paris / Disneyland_California")
    ap.add_argument("--country", nargs="*", help="Australia, United States, etc.")
    ap.add_argument("--month", type=int, nargs="*", help="1-12")
    ap.add_argument("--season", nargs="*", help="Spring Summer Fall Winter")
    ap.add_argument("--year", type=int, nargs="*")
    ap.add_argument("--rating_gte", type=float, default=None)
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=args.chroma_path)
    col = client.get_or_create_collection(args.collection)

    q_emb = embed_query_openai(args.question, model=args.oa_model)

    where = build_where(
        branch=args.branch,
        countries=args.country,
        months=args.month,
        seasons=args.season,
        years=args.year,
        rating_gte=args.rating_gte,
    )

    print("WHERE filter:", where)  # debug

    res = col.query(
        query_embeddings=[q_emb],
        n_results=args.top_k,
        where=where,
        include=["documents", "metadatas", "distances"]  # <-- no "ids"
    )

    ids = res.get("ids", [[]])[0]  # IDs come separately; not part of include
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    print(f"\nTop {len(ids)} results for: {args.question}")
    for i, (rid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
        snip = (doc[:200] + "…") if len(doc) > 200 else doc
        print(f"\n#{i} id={rid} dist={dist:.4f}")
        print(f"   {meta.get('Branch')} | {meta.get('Reviewer_Location')} | "
              f"{meta.get('Year')}-{meta.get('Month')} | Rating={meta.get('Rating')}")
        print(f"   {snip}")

if __name__ == "__main__":
    main()
