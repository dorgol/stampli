#!/usr/bin/env python3
"""
analysis_tool.py
Aggregate Disney reviews with metadata filters over the FULL filtered subset.

Example:
  python src/analysis_tool.py ^
    --meta data/embeddings_tmp/reviews_with_embeddings.parquet ^
    --branch Disneyland_HongKong --country Australia
"""

import argparse
import json
import re
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="Parquet with Review_ID, Review_Text, Branch, Reviewer_Location, Year_Month, Rating (+ Year/Month/Season if present)")
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
    # If Year/Month/Season missing, derive from Year_Month
    if "Year" not in df.columns or "Month" not in df.columns:
        years, months = [], []
        for v in df["Year_Month"].astype(str):
            y, m = None, None
            if re.fullmatch(r"\d{4}-\d{1,2}", v):
                y, m = v.split("-")
                y, m = int(y), int(m)
            years.append(y); months.append(m)
        df["Year"] = years
        df["Month"] = months
    if "Season" not in df.columns:
        def season(m):
            if m in (3,4,5):   return "Spring"
            if m in (6,7,8):   return "Summer"
            if m in (9,10,11): return "Fall"
            if m in (12,1,2):  return "Winter"
            return "Unknown"
        df["Season"] = [season(m) for m in df["Month"]]
    return df


def apply_filters(df: pd.DataFrame,
                  branch: Optional[List[str]], countries: Optional[List[str]],
                  months: Optional[List[int]], seasons: Optional[List[str]],
                  years: Optional[List[int]], rating_gte: Optional[float]) -> pd.DataFrame:
    m = df
    if branch:   m = m[m["Branch"].isin(branch)]
    if countries: m = m[m["Reviewer_Location"].isin(countries)]
    if months:   m = m[m["Month"].isin(months)]
    if seasons:  m = m[m["Season"].isin(seasons)]
    if years:    m = m[m["Year"].isin(years)]
    if rating_gte is not None: m = m[m["Rating"] >= rating_gte]
    return m.copy()


def compute_metrics(df: pd.DataFrame, topic_regexes: List[str]) -> Dict[str, Any]:
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

    # monthly
    bym = (df.groupby("Month")
             .agg(avg_rating=("Rating","mean"),
                  pos_share=("Rating", lambda s: (s>=4).mean()),
                  count=("Rating","size"))
             .reset_index()
             .sort_values("Month"))
    out["by_month"] = bym.to_dict(orient="records")

    # seasonal
    bys = (df.groupby("Season")
             .agg(pos_share=("Rating", lambda s: (s>=4).mean()),
                  count=("Rating","size"))
             .reset_index())
    # stable season order
    season_order = {"Spring":1,"Summer":2,"Fall":3,"Winter":4,"Unknown":5}
    bys["order"] = bys["Season"].map(season_order).fillna(99)
    out["by_season"] = bys.sort_values("order").drop(columns=["order"]).to_dict(orient="records")

    # topic counts (regex hit in Review_Text, case-insensitive)
    texts = df["Review_Text"].astype(str).values
    topic_rows = []
    for patt in topic_regexes:
        rx = re.compile(patt, flags=re.IGNORECASE)
        hits = int(sum(1 for t in texts if rx.search(t)))
        topic_rows.append({"topic_regex": patt, "count": hits, "share": (hits / n) if n else 0.0})
    out["topic_counts"] = topic_rows

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

    result = compute_metrics(fdf, args.topics)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
