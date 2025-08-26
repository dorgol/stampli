#!/usr/bin/env python3
"""
orchestrator.py
One-shot CLI that wires: parse → analyze → retrieve → synthesize (+ coverage backoff).

Usage:
  python src/orchestrator.py ^
    --question "Is the staff in Paris friendly?" ^
    --meta data/embeddings_out/reviews_with_embeddings.parquet ^
    --chroma_path data/index/chroma_db ^
    --collection reviews ^
    --top_k 12

(Optionally override filters explicitly; if omitted, they’re parsed from the question.)
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from src.analysis_tool import add_derived_cols, apply_filters, compute_metrics
from src.query_parser import parse_question
from src.retrieval import embed_query_openai, build_where, retrieve_snippets
from src.synthesis import run as synthesize


def _tolist(x):
    if x is None: return None
    return x if isinstance(x, list) else [x]

def _filters_from_args_or_parse(args) -> Dict[str, Any]:
    """
    If user provided any explicit filters, use those; otherwise parse from question.
    Returns a dict with normalized filters (lists where appropriate).
    """
    any_override = any([
        args.branch, args.country, args.month, args.season, args.year, args.rating_gte is not None
    ])
    if any_override:
        return {
            "branch":  _tolist(args.branch) if args.branch else None,
            "country": _tolist(args.country) if args.country else None,
            "month":   _tolist(args.month)  if args.month  else None,
            "season":  _tolist(args.season) if args.season else None,
            "year":    _tolist(args.year)   if args.year   else None,
            "rating_gte": args.rating_gte
        }
    parsed = parse_question(args.question)
    return {
        "branch":  _tolist(parsed.get("branch")),
        "country": _tolist(parsed.get("country")),
        "month":   _tolist(parsed.get("month"))   if parsed.get("month")   else None,
        "season":  _tolist(parsed.get("season"))  if parsed.get("season")  else None,
        "year":    _tolist(parsed.get("year"))    if parsed.get("year")    else None,
        "rating_gte": parsed.get("rating_gte"),
        "parsed": parsed
    }

def _run_once(df: pd.DataFrame,
              question: str,
              chroma_path: str,
              collection: str,
              branch: Optional[List[str]],
              country: Optional[List[str]],
              month: Optional[List[int]],
              season: Optional[List[str]],
              year: Optional[List[int]],
              rating_gte: Optional[float],
              top_k: int) -> Dict[str, Any]:
    """Compute stats + retrieve snippets + synthesize (no backoff)."""
    # Analysis
    filtered = apply_filters(df, branch, country, month, season, year, rating_gte)
    stats = compute_metrics(filtered)

    # Retrieval
    q_emb = embed_query_openai(question)
    where = build_where(branch, country, month, season, year, rating_gte)
    snippets = retrieve_snippets(chroma_path, collection, q_emb, where, top_k)

    # Synthesis
    synth = synthesize(question, stats, snippets)

    return {
        "stats": stats,
        "snippets": snippets,
        "synth": synth,
        "filters_used": {
            "branch": branch, "country": country, "month": month,
            "season": season, "year": year, "rating_gte": rating_gte
        }
    }

def run(args) -> Dict[str, Any]:
    t0 = time.time()
    # Load + prepare metadata
    df = pd.read_parquet(args.meta)
    need = {"Review_ID","Review_Text","Branch","Reviewer_Location","Year_Month","Rating"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{args.meta} missing columns: {miss} — re-run embeddings to retain metadata.")
    df = add_derived_cols(df)

    # Get filters (explicit overrides or parsed)
    filt = _filters_from_args_or_parse(args)
    parsed = filt.pop("parsed", None)  # optional: only present if we parsed

    # Pass #1: exact filters
    res = _run_once(df, args.question, args.chroma_path, args.collection,
                    filt["branch"], filt["country"], filt["month"], filt["season"], filt["year"],
                    filt["rating_gte"], args.top_k)

    relaxed = None

    # Backoff #1: if empty population, drop time filters (month/season/year)
    if res["stats"]["n"] == 0 and not args.no_relax:
        res2 = _run_once(df, args.question, args.chroma_path, args.collection,
                         filt["branch"], filt["country"], None, None, None,
                         filt["rating_gte"], args.top_k)
        if res2["stats"]["n"] > 0:
            res = res2
            relaxed = "time"

    # Backoff #2: still empty, and we had a branch → drop branch
    if res["stats"]["n"] == 0 and filt["branch"] and not args.no_relax:
        res3 = _run_once(df, args.question, args.chroma_path, args.collection,
                         None, filt["country"], None, None, None,
                         filt["rating_gte"], args.top_k)
        if res3["stats"]["n"] > 0:
            res = res3
            relaxed = "branch" if relaxed is None else f"{relaxed}+branch"

    # Assemble final
    answer_text = res["synth"]["answer"]
    if relaxed:
        answer_text = f"Note: no results with the exact filters; relaxed {relaxed} constraint(s).\n\n" + answer_text

    out = {
        "answer": answer_text,
        "filters": res["filters_used"],
        "stats": res["stats"],
        "citations": res["synth"]["citations"],
        "snippets_used": res["snippets"][:8],
        "parsed": parsed if parsed else None,
        "_meta": {"latency_ms": int((time.time()-t0)*1000), "relaxed": relaxed or ""}
    }
    return out

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--chroma_path", required=True)
    ap.add_argument("--collection", default="reviews")
    ap.add_argument("--top_k", type=int, default=12)

    # Optional explicit overrides (if omitted, we parse from question)
    ap.add_argument("--branch", nargs="*")
    ap.add_argument("--country", nargs="*")
    ap.add_argument("--month", type=int, nargs="*")
    ap.add_argument("--season", nargs="*")
    ap.add_argument("--year", type=int, nargs="*")
    ap.add_argument("--rating_gte", type=float, default=None)

    # Disable relaxation (for testing)
    ap.add_argument("--no_relax", action="store_true")

    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))
