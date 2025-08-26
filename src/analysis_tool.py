#!/usr/bin/env python3
"""
analysis_tool.py
Aggregate Disney reviews with metadata filters over the FULL filtered subset.
Now cluster-aware: detects active cluster id/label columns (cluster_* / *_label).

Examples:
  python src/analysis_tool.py ^
    --meta clustering_out/reviews_with_clusters_labeled.parquet ^
    --branch Disneyland_HongKong --country Australia

  # cluster filters (exact match)
  python src/analysis_tool.py --meta clustering_out/reviews_with_clusters_labeled.parquet --cluster_label "Staff Experience"
  python src/analysis_tool.py --meta clustering_out/reviews_with_clusters_labeled.parquet --cluster_id 27
"""
from __future__ import annotations

import argparse
import json
import re
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd

# ---------------- args ----------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True,
                    help="Parquet with Review_ID, Review_Text, Branch, Reviewer_Location, Year_Month, Rating (+ derived & cluster cols)")
    ap.add_argument("--branch", nargs="*", help="Disneyland_HongKong / Disneyland_Paris / Disneyland_California")
    ap.add_argument("--country", nargs="*", help="e.g., Australia 'United States'")
    ap.add_argument("--month", type=int, nargs="*", help="1-12")
    ap.add_argument("--season", nargs="*", help="Spring Summer Fall Winter")
    ap.add_argument("--year", type=int, nargs="*")
    ap.add_argument("--sentiment", nargs="*", help="e.g., negative positive mixed 'very positive'")
    ap.add_argument("--rating_gte", type=float, default=None)

    # NEW: cluster filters
    ap.add_argument("--cluster_id", type=int, help="Exact cluster id (in active cluster column)")
    ap.add_argument("--cluster_label", type=str, help="Exact cluster label (in active *_label column)")

    ap.add_argument("--topics", nargs="*", default=[
        r"(crowd|queue|queues|line|lines|wait|waiting)",
        r"(staff|friendly|rude|cast\s*member|employees?)",
        r"(food|meal|lunch|dinner|restaurant|price|expensive|overpriced)"
    ], help="Regex groups for theme counts")
    return ap.parse_args()

# ---------------- helpers ----------------

def detect_active_cluster_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (cluster_id_col, cluster_label_col) by picking the *_label column
    with the most non-nulls, and its base id column if present.
    """
    label_cols = [c for c in df.columns if c.endswith("_label") and df[c].notna().any()]
    if not label_cols:
        return None, None
    label_col = max(label_cols, key=lambda c: df[c].notna().sum())
    base = label_col[:-6]  # drop "_label"
    id_col = base if base in df.columns else None
    return id_col, label_col

def add_derived_cols(df: pd.DataFrame) -> pd.DataFrame:
    if "Rating" in df.columns:
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    if "Year" not in df.columns or "Month" not in df.columns:
        years, months = [], []
        for v in df["Year_Month"].astype(str):
            y = m = None
            try:
                parts = v.split("-")
                if len(parts) == 2:
                    y = int(float(parts[0])); m = int(float(parts[1]))
            except Exception:
                pass
            years.append(y); months.append(m)
        df["Year"], df["Month"] = years, months
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
                  rating_gte: Optional[float] = None,
                  sentiment: Optional[List[str]] = None,
                  cluster_id_col: Optional[str] = None,
                  cluster_label_col: Optional[str] = None,
                  cluster_id: Optional[int] = None,
                  cluster_label: Optional[str] = None) -> pd.DataFrame:
    m = df
    if branch:    m = m[m["Branch"].isin(branch)]
    if countries: m = m[m["Reviewer_Location"].isin(countries)]
    if months:    m = m[m["Month"].isin(months)]
    if seasons:   m = m[m["Season"].isin(seasons)]
    if years:     m = m[m["Year"].isin(years)]
    if rating_gte is not None: m = m[m["Rating"] >= rating_gte]
    if sentiment and "sentiment" in m.columns:
        m = m[m["sentiment"].isin(sentiment)]
    if cluster_id_col and cluster_id is not None and cluster_id_col in m.columns:
        m = m[m[cluster_id_col] == int(cluster_id)]
    if cluster_label_col and cluster_label and cluster_label_col in m.columns:
        m = m[m[cluster_label_col] == cluster_label]
    return m.dropna(subset=["Rating"]).copy()

def compute_metrics(df: pd.DataFrame,
                    topic_regexes: Optional[List[str]] = None,
                    cluster_label_col: Optional[str] = None) -> Dict[str, Any]:
    topic_regexes = topic_regexes or [
        r"(crowd|queue|queues|line|lines|wait|waiting)",
        r"(staff|friendly|rude|cast\s*member|employees?)",
        r"(food|meal|lunch|dinner|restaurant|price|expensive|overpriced)"
    ]
    out: Dict[str, Any] = {}
    n = int(len(df))
    out["n"] = n
    if n == 0:
        out.update({
            "avg_rating": None, "pos_share": None,
            "by_month": [], "by_season": [],
            "by_cluster_label": [], "by_cluster_rating": [],
            "topic_counts": []
        })
        return out

    out["avg_rating"] = float(df["Rating"].mean())
    out["pos_share"]  = float((df["Rating"] >= 4).mean())

    # Month breakdown
    if "Month" in df.columns:
        bym = (df.dropna(subset=["Month"])
                 .groupby("Month", dropna=True)
                 .agg(avg_rating=("Rating","mean"),
                      pos_share=("Rating", lambda s: (s>=4).mean()),
                      count=("Rating","size"))
                 .reset_index().sort_values("Month"))
        out["by_month"] = bym.to_dict(orient="records")
    else:
        out["by_month"] = []

    # Season breakdown
    if "Season" in df.columns:
        bys = (df.groupby("Season")
                 .agg(pos_share=("Rating", lambda s: (s>=4).mean()),
                      count=("Rating","size"),
                      avg_rating=("Rating","mean"))
                 .reset_index())
        order = {"Spring":1,"Summer":2,"Fall":3,"Winter":4,"Unknown":5}
        bys["order"] = bys["Season"].map(order).fillna(99)
        out["by_season"] = (bys.sort_values("order")
                              .drop(columns=["order"])
                              .to_dict(orient="records"))
    else:
        out["by_season"] = []

    # Cluster label distributions (if available)
    if cluster_label_col and cluster_label_col in df.columns:
        vc = (df[cluster_label_col].dropna().astype(str).value_counts().reset_index())
        vc.columns = ["cluster_label","count"]
        out["by_cluster_label"] = vc.to_dict(orient="records")

        byc = (df.dropna(subset=[cluster_label_col])
                .groupby(cluster_label_col)
                .agg(avg_rating=("Rating","mean"),
                     count=("Rating","size"))
                .reset_index()
                .sort_values("count", ascending=False))
        byc.columns = ["cluster_label","avg_rating","count"]
        out["by_cluster_rating"] = byc.to_dict(orient="records")
    else:
        out["by_cluster_label"] = []
        out["by_cluster_rating"] = []

    # Topic counts on text
    texts = df["Review_Text"].astype(str).values if "Review_Text" in df.columns else []
    rows = []
    for patt in topic_regexes:
        rx = re.compile(patt, flags=re.IGNORECASE)
        hits = int(sum(1 for t in texts if rx.search(t)))
        rows.append({"topic_regex": patt, "count": hits, "share": (hits/n) if n else 0.0})
    out["topic_counts"] = rows
    return out

# ---------------- main ----------------

def main():
    args = parse_args()
    df = pd.read_parquet(args.meta)
    need = {"Review_ID","Review_Text","Branch","Reviewer_Location","Year_Month","Rating"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{args.meta} missing columns: {missing} — re-run embedding to retain metadata.")
    df = add_derived_cols(df)

    cluster_id_col, cluster_label_col = detect_active_cluster_cols(df)

    fdf = apply_filters(
        df,
        branch=args.branch,
        countries=args.country,
        months=args.month,
        seasons=args.season,
        years=args.year,
        rating_gte=args.rating_gte,
        sentiment=args.sentiment,
        cluster_id_col=cluster_id_col,
        cluster_label_col=cluster_label_col,
        cluster_id=args.cluster_id,
        cluster_label=args.cluster_label
    )
    result = compute_metrics(fdf, topic_regexes=args.topics, cluster_label_col=cluster_label_col)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
