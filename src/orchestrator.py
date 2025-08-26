#!/usr/bin/env python3
import argparse, json
from typing import List, Optional
from query_parser import parse_question
from synthesis import run as synth_run

def _to_list(x):
    if x is None: return None
    return x if isinstance(x, list) else [x]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--chroma_path", required=True)
    ap.add_argument("--collection", default="reviews")
    ap.add_argument("--top_k", type=int, default=12)
    args = ap.parse_args()

    parsed = parse_question(args.question)  # JSON Schema enforced
    branch = _to_list(parsed.get("branch"))
    country = _to_list(parsed.get("country"))
    season  = parsed.get("season")
    month   = parsed.get("month")
    year    = parsed.get("year")
    rating  = parsed.get("rating_gte")
    focus   = parsed.get("focus")

    # 1st pass: exact filters
    out = synth_run(
        question=args.question,
        meta_path=args.meta,
        chroma_path=args.chroma_path,
        collection=args.collection,
        branch=branch, country=country,
        month=[month] if month else None,
        season=[season] if season else None,
        year=[year] if year else None,
        rating_gte=rating,
        top_k=args.top_k
    )

    # Coverage backoff if empty
    relaxed = None
    if out["stats"]["n"] == 0:
        # drop time filters
        out = synth_run(args.question, args.meta, args.chroma_path, args.collection,
                        branch=branch, country=country,
                        month=None, season=None, year=None,
                        rating_gte=rating, top_k=args.top_k)
        relaxed = "time" if out["stats"]["n"] > 0 else None

    if out["stats"]["n"] == 0 and branch:
        # drop branch last (broaden to all parks)
        out = synth_run(args.question, args.meta, args.chroma_path, args.collection,
                        branch=None, country=country,
                        month=None, season=None, year=None,
                        rating_gte=rating, top_k=args.top_k)
        relaxed = "branch" if out["stats"]["n"] > 0 else relaxed

    # annotate disclosure if we relaxed
    if relaxed:
        out["answer"] = (f"Note: no results with the exact filters; relaxed {relaxed} constraint(s).\n\n"
                         + out["answer"])

    # pass focus back (useful for UI/telemetry)
    out["parsed"] = {**parsed, "focus": focus}
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
