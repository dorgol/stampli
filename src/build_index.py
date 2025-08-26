#!/usr/bin/env python3
"""
build_index.py
Create a persistent Chroma collection for Disney reviews using precomputed embeddings.

Inputs:
  --embeddings  embeddings_out/embeddings.npy
  --meta        embeddings_out/reviews_with_embeddings.parquet
                (must include: Review_ID, Review_Text, Branch, Reviewer_Location, Year_Month, Rating)
Outputs (folder):
  --chroma_path index/chroma_db           (DuckDB/Parquet files)
Collection:
  --collection  reviews                   (default)
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import chromadb

REQ_COLS = ["Review_ID","Review_Text","Branch","Reviewer_Location","Year_Month","Rating"]

def add_derived_cols(df: pd.DataFrame) -> pd.DataFrame:
    def to_ym(s):
        s = str(s)
        if re.fullmatch(r"\d{4}-\d{1,2}", s):
            y, m = s.split("-")
            return int(y), int(m)
        return None, None

    years, months = [], []
    for v in df["Year_Month"]:
        y, m = to_ym(v)
        years.append(y); months.append(m)
    df["Year"] = years
    df["Month"] = months

    def season(m):
        if m in (3,4,5):   return "Spring"
        if m in (6,7,8):   return "Summer"
        if m in (9,10,11): return "Fall"
        if m in (12,1,2):  return "Winter"
        return "Unknown"
    df["Season"] = [season(m) for m in df["Month"]]
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--chroma_path", default="index/chroma_db")
    ap.add_argument("--collection", default="reviews")
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--reset", action="store_true", help="Drop existing collection first")
    args = ap.parse_args()

    # Load
    X = np.load(args.embeddings)  # shape [n, d], dtype float32
    df = pd.read_parquet(args.meta)
    if len(df) != X.shape[0]:
        raise ValueError(f"Row mismatch: meta={len(df)} vs embeddings={X.shape[0]}")

    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Meta parquet missing columns: {missing}. "
                         f"Re-run your embedding script so it preserves metadata.")

    df = add_derived_cols(df)
    # Ensure unique IDs as strings for Chroma
    df["doc_id"] = df["Review_ID"].astype(str)

    # Start Chroma (persistent)
    Path(args.chroma_path).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=args.chroma_path)

    if args.reset:
        try:
            client.delete_collection(args.collection)
        except Exception:
            pass

    # Cosine space is default; we pass embeddings directly
    col = client.get_or_create_collection(name=args.collection, metadata={"hnsw:space": "cosine"})

    n = len(df)
    bs = args.batch_size
    for i in range(0, n, bs):
        sl = df.iloc[i:i+bs]
        ids = sl["doc_id"].tolist()
        docs = sl["Review_Text"].astype(str).tolist()
        mets = sl[["Review_ID","Branch","Reviewer_Location","Year","Month","Season","Rating"]].to_dict("records")
        embs = X[i:i+bs].tolist()
        col.upsert(ids=ids, documents=docs, embeddings=embs, metadatas=mets)
        print(f"Upserted {i+len(sl)}/{n}")

    print(f"Done. Collection '{args.collection}' in {args.chroma_path}")

if __name__ == "__main__":
    main()
