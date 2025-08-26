#!/usr/bin/env python3
"""
analysis_tool.py
Aggregate Disney reviews with metadata filters over the FULL filtered subset.

Example:
  python src/analysis_tool.py ^
    --meta data/embeddings_out/reviews_with_embeddings.parquet ^
    --branch Disneyland_HongKong --country Australia
"""
from __future__ import annotations

import argparse
import json
import re
from typing import List, Dict, Any, Optional

import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True,
                    help="Parquet with Review_ID, Review_Text, Branch, Reviewer_Location, Year_Month, Rating (+ Year/Month/Season if present)")
    ap.add_argument("--branch", nargs="*", help="Disneyland_HongKong / Disneyland_Paris / Disneyland_California")
    ap.add_argument("--country", nargs="*", help="e.g., Australia 'United States'")
    ap.add_argument("--month", type=int, nargs="*", help="1-12")
    ap.add_argument("--season", nargs="*", help="Spring Summer Fall Winter")
    ap.add_argument("--year", type=int, nargs="*")
    ap.add_argument("--rating_gte", type=float, default=None)
    ap.add_argument("--topics", nargs="*", default=[
        r"(crowd|queue|queues|line|lines|wait|waiting)",
        r"(staff|friendly|rude|cast\s*member|employees?)",
        r"(food|meal|lunch|dinner|restaurant|price|expensive|overpriced)"
    ], help="Regex groups for theme counts")
    return ap.parse_args()

def add_derived_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Coerce Rating to numeric (drops bad rows later if needed)
    if "Rating" in df.columns:
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

    # Derive Year/Month from Year_Month if missing
    if "Year" not in df.columns or "Month" not in df.columns:
        years, months = [], []
        for v in df["Year_Month"].astype(str):
            y = m = None
            try:
                parts = v.split("-")
                if len(parts) == 2:
                    y = int(float(parts[0]))
                    m = int(float(parts[1]))
            except Exception:
                y = m = None
            years.append(y); months.append(m)
        df["Year"], df["Month"] = years, months

    # Season from Month
    if "Season" not in df.columns:
        def season(m):
            if m in (3, 4, 5):   return "Spring"
            if m in (6, 7, 8):   return "Summer"
            if m in (9, 10, 11): return "Fall"
            if m in (12, 1, 2):  return "Winter"
            return "Unknown"
        df["Season"] = [season(m) for m in df["Month"]]
    return df

def apply_filters(df: pd.DataFrame,
                  branch: Optional[List[str]] = None,
                  countries: Optional[List[str]] = None,
                  months: Optional[List[int]] = None,
                  seasons: Optional[List[str]] = None,
                  years: Optional[List[int]] = None,
                  rating_gte: Optional[float] = None) -> pd.DataFrame:
    m = df
    if branch:    m = m[m["Branch"].isin(branch)]
    if countries: m = m[m["Reviewer_Location"].isin(countries)]
    if months:    m = m[m["Month"].isin(months)]
    if seasons:   m = m[m["Season"].isin(seasons)]
    if years:     m = m[m["Year"].isin(years)]
    if rating_gte is not None: m = m[m["Rating"] >= rating_gte]
    # Drop rows with missing critical fields after filtering
    return m.dropna(subset=["Rating"]).copy()

def compute_metrics(df: pd.DataFrame, topic_regexes: Optional[List[str]] = None) -> Dict[str, Any]:
    topic_regexes = topic_regexes or [
        r"(crowd|queue|queues|line|lines|wait|waiting)",
        r"(staff|friendly|rude|cast\s*member|employees?)",
        r"(food|meal|lunch|dinner|restaurant|price|expensive|overpriced)"
    ]
    out: Dict[str, Any] = {}
    n = int(len(df))
    out["n"] = n
    if n == 0:
        out.update({"avg_rating": None, "pos_share": None, "by_month": [], "by_season": [], "topic_counts": []})
        return out

    out["avg_rating"] = float(df["Rating"].mean())
    out["pos_share"]  = float((df["Rating"] >= 4).mean())

    # Month breakdown (ignore rows with missing Month)
    bym = (df.dropna(subset=["Month"])
             .groupby("Month", dropna=True)
             .agg(avg_rating=("Rating","mean"),
                  pos_share=("Rating", lambda s: (s>=4).mean()),
                  count=("Rating","size"))
             .reset_index().sort_values("Month"))
    out["by_month"] = bym.to_dict(orient="records")

    # Season breakdown
    bys = (df.groupby("Season")
             .agg(pos_share=("Rating", lambda s: (s>=4).mean()),
                  count=("Rating","size"))
             .reset_index())
    order = {"Spring":1,"Summer":2,"Fall":3,"Winter":4,"Unknown":5}
    bys["order"] = bys["Season"].map(order).fillna(99)
    out["by_season"] = (bys.sort_values("order")
                          .drop(columns=["order"])
                          .to_dict(orient="records"))

    # Topic counts
    texts = df["Review_Text"].astype(str).values
    rows = []
    for patt in topic_regexes:
        rx = re.compile(patt, flags=re.IGNORECASE)
        hits = int(sum(1 for t in texts if rx.search(t)))
        rows.append({"topic_regex": patt, "count": hits, "share": (hits/n) if n else 0.0})
    out["topic_counts"] = rows
    return out

def main():
    args = parse_args()
    df = pd.read_parquet(args.meta)
    need = {"Review_ID","Review_Text","Branch","Reviewer_Location","Year_Month","Rating"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{args.meta} missing columns: {missing} — re-run embedding to retain metadata.")
    df = add_derived_cols(df)
    fdf = apply_filters(
        df,
        branch=args.branch,
        countries=args.country,
        months=args.month,
        seasons=args.season,
        years=args.year,
        rating_gte=args.rating_gte
    )
    result = compute_metrics(fdf, topic_regexes=args.topics)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
