#!/usr/bin/env python3
"""
build_index.py  —  minimal, robust Chroma indexer for Disney reviews

What it does:
- Loads embeddings (.npy) and a parquet with text + metadata
- Adds Year/Month/Season derived from Year_Month
- Picks a clean set of PRIMITIVE metadata to store in Chroma:
  * base: Review_ID, Branch, Reviewer_Location, Year, Month, Season, Rating
  * dynamic: any columns starting with 'cluster_' and ending with '_label'
  * optional: sentiment, keywords_str (stringified keywords)
- Upserts documents+embeddings into a persistent Chroma collection

Usage (example):
  python src\\build_index.py ^
    --embeddings data/embeddings_out/embeddings.npy ^
    --meta clustering_out/reviews_with_clusters_labeled.parquet ^
    --chroma_path index/chroma_db ^
    --collection reviews --reset
"""

import argparse
from pathlib import Path
import re
import json
import numpy as np
import pandas as pd
import chromadb

REQ_COLS = ["Review_ID", "Review_Text", "Branch", "Reviewer_Location", "Year_Month", "Rating"]

# ---------- helpers ----------

def add_derived_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add Year/Month/Season from Year_Month like '2023-07'."""
    def to_ym(s):
        s = str(s)
        if re.fullmatch(r"\d{4}-\d{1,2}", s):
            y, m = s.split("-")
            return int(y), int(m)
        return None, None

    years, months = zip(*(to_ym(v) for v in df["Year_Month"]))
    df["Year"] = list(years)
    df["Month"] = list(months)

    def season(m):
        if m in (3, 4, 5):   return "Spring"
        if m in (6, 7, 8):   return "Summer"
        if m in (9, 10, 11): return "Fall"
        if m in (12, 1, 2):  return "Winter"
        return "Unknown"

    df["Season"] = [season(m) for m in df["Month"]]
    return df

def to_primitive(x):
    """Convert any value to a Chroma-safe primitive (str/int/float/bool/None)."""
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, np.generic):  # numpy scalar
        return x.item()
    if isinstance(x, (list, tuple, set)):
        return " | ".join(map(str, x))
    if isinstance(x, np.ndarray):
        try:
            return " | ".join(map(str, x.tolist()))
        except Exception:
            return str(x)
    if isinstance(x, dict):
        return json.dumps(x, ensure_ascii=False)
    return str(x)

def choose_metadata_columns(df: pd.DataFrame) -> list[str]:
    """Pick base + dynamic cluster/label columns and small optional fields (as strings)."""
    base = ["Review_ID", "Branch", "Reviewer_Location", "Year", "Month", "Season", "Rating"]
    cluster_cols = [c for c in df.columns if c.startswith("cluster_")]
    label_cols   = [c for c in df.columns if c.endswith("_label")]

    # Optional small text fields
    optional_small = []
    if "sentiment" in df.columns:
        optional_small.append("sentiment")

    # Provide a safe string field for keywords
    cols = base + cluster_cols + label_cols + optional_small
    cols = [c for c in cols if c in df.columns]

    # Unique, stable order
    seen, out = set(), []
    for c in cols:
        if c not in seen:
            out.append(c); seen.add(c)
    return out

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True, help="Path to embeddings .npy")
    ap.add_argument("--meta", required=True, help="Path to parquet with text + metadata")
    ap.add_argument("--chroma_path", default="index/chroma_db")
    ap.add_argument("--collection", default="reviews")
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--reset", action="store_true", help="Drop existing collection first")
    ap.add_argument("--id_col", default="Review_ID")
    ap.add_argument("--text_col", default="Review_Text")
    args = ap.parse_args()

    # Load
    X = np.load(args.embeddings)
    df = pd.read_parquet(args.meta)

    # Sanity
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Parquet missing columns: {missing}")
    if len(df) != X.shape[0]:
        raise ValueError(f"Row mismatch: meta={len(df)} vs embeddings={X.shape[0]}")

    # Derive time, ensure keyword string
    df = add_derived_cols(df)
    if "keywords" in df.columns and "keywords_str" not in df.columns:
        df["keywords_str"] = df["keywords"].apply(to_primitive)
    # Prefer to include keywords_str if present
    meta_cols = choose_metadata_columns(df)
    if "keywords_str" in df.columns and "keywords_str" not in meta_cols:
        meta_cols.append("keywords_str")

    # IDs / text
    if args.id_col not in df.columns:
        raise ValueError(f"ID column '{args.id_col}' not found.")
    if args.text_col not in df.columns:
        raise ValueError(f"Text column '{args.text_col}' not found.")
    df["doc_id"] = df[args.id_col].astype(str)

    # Prepare Chroma
    Path(args.chroma_path).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=args.chroma_path)
    if args.reset:
        try:
            client.delete_collection(args.collection)
        except Exception:
            pass
    col = client.get_or_create_collection(name=args.collection, metadata={"hnsw:space": "cosine"})

    # Upsert in batches, converting metadata to primitives
    n = len(df)
    bs = int(args.batch_size)
    for i in range(0, n, bs):
        sl = df.iloc[i:i+bs].copy()

        # make a primitive-only metadata frame
        safe_meta = sl[meta_cols].applymap(to_primitive)
        mets = safe_meta.to_dict("records")

        col.upsert(
            ids=sl["doc_id"].astype(str).tolist(),
            documents=sl[args.text_col].astype(str).tolist(),
            embeddings=X[i:i+len(sl)].tolist(),
            metadatas=mets,
        )
        print(f"Upserted {i + len(sl)}/{n}")

    print(f"Done. Collection '{args.collection}' at {args.chroma_path}")

if __name__ == "__main__":
    main()
