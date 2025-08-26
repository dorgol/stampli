#!/usr/bin/env python3
"""
synthesis.py
Combine analysis (full filtered stats) + retrieval (top-k snippets) into a grounded answer using gpt-4o-mini.

Usage:
  $env:OPENAI_API_KEY="sk-..."
  python src/synthesis.py ^
    --question "Is Disneyland California usually crowded in June?" ^
    --meta data/embeddings_tmp/reviews_with_embeddings.parquet ^
    --chroma_path data/index_tmp/chroma_db ^
    --branch Disneyland_California --month 6 --top_k 12

Outputs JSON to stdout:
{
  "answer": "...",
  "filters": {...},
  "stats": {...},            # from analysis_tool
  "citations": [Review_ID...],
  "snippets_used": [{id, branch, country, year, month, rating, snippet}]
}
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import chromadb

# ---- OpenAI (chat) ----
from openai import OpenAI

# ---- Local imports (optional) ----
# If you exported helpers in your retrieval.py / analysis_tool.py, you can import them.
# from retrieval import embed_query_openai, build_where
# from analysis_tool import add_derived_cols, apply_filters, compute_metrics

# ------------- Minimal fallbacks so this file works standalone -------------
def _embed_query_openai(text: str, model: str = "text-embedding-3-small") -> List[float]:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set.")
    client = OpenAI()
    resp = client.embeddings.create(model=model, input=[text])
    return resp.data[0].embedding

def _build_where(branch=None, countries=None, months=None, seasons=None, years=None, rating_gte=None):
    clauses = []

    def one_or_in(field, values):
        if not values:
            return None
        values = [v for v in values if v is not None]
        if not values:
            return None
        if len(values) == 1:
            return {field: values[0]}
        return {field: {"$in": values}}

    if branch:    clauses.append(one_or_in("Branch", branch))
    if countries: clauses.append(one_or_in("Reviewer_Location", countries))
    if months:    clauses.append(one_or_in("Month", months))
    if seasons:   clauses.append(one_or_in("Season", seasons))
    if years:     clauses.append(one_or_in("Year", years))
    if rating_gte is not None:
        clauses.append({"Rating": {"$gte": float(rating_gte)}})

    clauses = [c for c in clauses if c]
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}

def _add_derived_cols(df: pd.DataFrame) -> pd.DataFrame:
    # (re)derive Year/Month/Season if missing
    if "Year" not in df.columns or "Month" not in df.columns:
        years, months = [], []
        for v in df["Year_Month"].astype(str):
            y, m = None, None
            if "-" in v:
                parts = v.split("-")
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    y, m = int(parts[0]), int(parts[1])
            years.append(y); months.append(m)
        df["Year"], df["Month"] = years, months
    if "Season" not in df.columns:
        def season(m):
            if m in (3,4,5):   return "Spring"
            if m in (6,7,8):   return "Summer"
            if m in (9,10,11): return "Fall"
            if m in (12,1,2):  return "Winter"
            return "Unknown"
        df["Season"] = [season(m) for m in df["Month"]]
    return df

def _apply_filters(df: pd.DataFrame,
                   branch: Optional[List[str]], countries: Optional[List[str]],
                   months: Optional[List[int]], seasons: Optional[List[str]],
                   years: Optional[List[int]], rating_gte: Optional[float]) -> pd.DataFrame:
    m = df
    if branch:    m = m[m["Branch"].isin(branch)]
    if countries: m = m[m["Reviewer_Location"].isin(countries)]
    if months:    m = m[m["Month"].isin(months)]
    if seasons:   m = m[m["Season"].isin(seasons)]
    if years:     m = m[m["Year"].isin(years)]
    if rating_gte is not None: m = m[m["Rating"] >= rating_gte]
    return m.copy()

def _compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    n = len(df)
    out["n"] = int(n)
    if n == 0:
        out["avg_rating"] = None
        out["pos_share"] = None
        out["by_month"] = []
        out["by_season"] = []
        out["topic_counts"] = []
        return out
    out["avg_rating"] = float(df["Rating"].mean())
    out["pos_share"] = float((df["Rating"] >= 4).mean())
    # month breakdown
    bym = (df.groupby("Month")
             .agg(avg_rating=("Rating","mean"),
                  pos_share=("Rating", lambda s: (s>=4).mean()),
                  count=("Rating","size"))
             .reset_index()
             .sort_values("Month"))
    out["by_month"] = bym.to_dict(orient="records")
    # season breakdown
    bys = (df.groupby("Season")
             .agg(pos_share=("Rating", lambda s: (s>=4).mean()),
                  count=("Rating","size"))
             .reset_index())
    order = {"Spring":1,"Summer":2,"Fall":3,"Winter":4,"Unknown":5}
    bys["order"] = bys["Season"].map(order).fillna(99)
    out["by_season"] = bys.sort_values("order").drop(columns=["order"]).to_dict(orient="records")
    # lightweight topic counts (can be extended)
    import re
    topics = [
        r"(crowd|queue|queues|line|lines|wait|waiting)",
        r"(staff|friendly|rude|cast\s*member|employees?)",
        r"(food|meal|lunch|dinner|restaurant|price|expensive|overpriced)"
    ]
    texts = df["Review_Text"].astype(str).values
    rows = []
    for patt in topics:
        rx = re.compile(patt, flags=re.IGNORECASE)
        hits = int(sum(1 for t in texts if rx.search(t)))
        rows.append({"topic_regex": patt, "count": hits, "share": (hits/n) if n else 0.0})
    out["topic_counts"] = rows
    return out
# ---------------------------------------------------------------------------

def synthesize_answer(question: str,
                      stats: Dict[str, Any],
                      snippets: List[Dict[str, Any]],
                      model: str = "gpt-4o-mini") -> str:
    """
    Ask the LLM to write a concise, grounded answer using provided stats + snippets.
    """
    client = OpenAI()

    # Build a compact, citation-friendly context
    preview_snips = []
    for s in snippets[:8]:  # cap to keep prompt small
        rid = s.get("Review_ID") or s.get("id")
        br = s.get("Branch")
        loc = s.get("Reviewer_Location") or s.get("country")
        yr = s.get("Year"); mo = s.get("Month")
        rating = s.get("Rating")
        text = s.get("snippet", "")
        preview_snips.append({
            "id": rid, "branch": br, "country": loc, "year": yr, "month": mo, "rating": rating,
            "quote": text[:280]
        })

    system_msg = (
        "You are a precise analyst. Answer the user's question ONLY using the provided statistics and review snippets. "
        "Start with a crisp 1–2 sentence verdict. Then add 2–4 bullet points with key numbers. "
        "Cite 2–3 review IDs inline like [ID: 670772142]. Do not invent facts not present in stats/snippets. "
        "If evidence is weak or n<20, state uncertainty."
    )

    user_payload = {
        "question": question,
        "stats": stats,
        "snippets": preview_snips
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": json.dumps(user_payload)}
        ],
        temperature=0.2,
        max_tokens=450,
    )
    return resp.choices[0].message.content.strip()

def run(question: str,
        meta_path: str,
        chroma_path: str,
        collection: str,
        branch: Optional[List[str]],
        country: Optional[List[str]],
        month: Optional[List[int]],
        season: Optional[List[str]],
        year: Optional[List[int]],
        rating_gte: Optional[float],
        top_k: int = 12,
        oa_embed_model: str = "text-embedding-3-small") -> Dict[str, Any]:

    # ---------- ANALYSIS over FULL filtered set ----------
    df = pd.read_parquet(meta_path)
    need = {"Review_ID","Review_Text","Branch","Reviewer_Location","Year_Month","Rating"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{meta_path} missing columns: {missing}")

    df = _add_derived_cols(df)
    filtered = _apply_filters(df, branch, country, month, season, year, rating_gte)
    stats = _compute_metrics(filtered)

    # ---------- RETRIEVAL (top-k evidence) ----------
    client = chromadb.PersistentClient(path=chroma_path)
    col = client.get_or_create_collection(collection)
    q_emb = _embed_query_openai(question, model=oa_embed_model)
    where = _build_where(branch, country, month, season, year, rating_gte)

    res = col.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        where=where,
        include=["documents","metadatas","distances"]
    )

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    # Build snippet objects (trim text)
    snippets = []
    for rid, doc, meta, dist in zip(ids, docs, metas, dists):
        snip = doc[:420] + ("…" if len(doc) > 420 else "")
        snippets.append({
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

    # ---------- Synthesize answer ----------
    answer_text = synthesize_answer(question, stats, snippets)

    # ---------- Assemble payload ----------
    citations = [s.get("Review_ID") for s in snippets[:3] if s.get("Review_ID") is not None]
    payload = {
        "answer": answer_text,
        "filters": {
            "branch": branch, "country": country, "month": month,
            "season": season, "year": year, "rating_gte": rating_gte
        },
        "stats": stats,
        "citations": citations,
        "snippets_used": snippets[:8]  # first few for transparency
    }
    return payload

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--meta", required=True, help="Parquet with full metadata")
    ap.add_argument("--chroma_path", required=True)
    ap.add_argument("--collection", default="reviews")
    ap.add_argument("--branch", nargs="*")
    ap.add_argument("--country", nargs="*")
    ap.add_argument("--month", type=int, nargs="*")
    ap.add_argument("--season", nargs="*")
    ap.add_argument("--year", type=int, nargs="*")
    ap.add_argument("--rating_gte", type=float, default=None)
    ap.add_argument("--top_k", type=int, default=12)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    out = run(
        question=args.question,
        meta_path=args.meta,
        chroma_path=args.chroma_path,
        collection=args.collection,
        branch=args.branch,
        country=args.country,
        month=args.month,
        season=args.season,
        year=args.year,
        rating_gte=args.rating_gte,
        top_k=args.top_k
    )
    print(json.dumps(out, indent=2))
