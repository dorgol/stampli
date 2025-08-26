#!/usr/bin/env python3
"""
sample_cluster_reps.py
Select representative reviews per cluster to feed into an LLM for labeling.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

def pick_cluster_col(df: pd.DataFrame) -> str:
    pref = ["cluster_hdbscan", "cluster_micro", "cluster_kmeans", "cluster_mbk"]
    for c in pref:
        if c in df.columns:
            return c
    cands = [c for c in df.columns if c.startswith("cluster_")]
    if not cands:
        raise KeyError("No cluster_* column found in the parquet file.")
    return cands[-1]

def compute_reps(embeddings: np.ndarray, df: pd.DataFrame, cluster_col: str, top_k: int = 20):
    X = normalize(embeddings)
    df = df.copy()
    df["__row_index"] = np.arange(len(df))
    reps, overview = [], []
    for cluster_id, g in df.groupby(cluster_col, sort=False):
        idx = g["__row_index"].values
        Xg = X[idx]
        n = len(idx)
        if n == 0:
            continue
        if cluster_id == -1:
            overview.append({"cluster": int(cluster_id), "n_reviews": int(n), "n_reps": 0})
            continue
        centroid = Xg.mean(axis=0, keepdims=True)
        sims = Xg @ centroid.T
        order = np.argsort(-sims.squeeze())
        take = min(top_k, n)
        chosen_idx = idx[order[:take]]
        chosen_sims = sims.squeeze()[order[:take]]
        for rank, (rid, sim) in enumerate(zip(chosen_idx, chosen_sims), start=1):
            reps.append({
                "cluster": int(cluster_id),
                "emb_index": int(rid),
                "Review_ID": df.loc[rid, "Review_ID"],
                "Review_Text": df.loc[rid, "Review_Text"],
                "rep_rank": int(rank),
                "sim_to_centroid": float(sim),
            })
        overview.append({"cluster": int(cluster_id), "n_reviews": int(n), "n_reps": int(take)})
    reps_df = pd.DataFrame(reps).sort_values(["cluster", "rep_rank"]).reset_index(drop=True)
    over_df = pd.DataFrame(overview).sort_values("n_reviews", ascending=False).reset_index(drop=True)
    return reps_df, over_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=str, default="embeddings_out/embeddings.npy")
    ap.add_argument("--clusters_parquet", type=str, default="clustering_out/reviews_with_clusters.parquet")
    ap.add_argument("--out_dir", type=str, default="labeling_in")
    ap.add_argument("--cluster_col", type=str, default=None)
    ap.add_argument("--top_k", type=int, default=20)
    args = ap.parse_args()

    outp = Path(args.out_dir); outp.mkdir(parents=True, exist_ok=True)
    emb = np.load(args.embeddings)
    df = pd.read_parquet(args.clusters_parquet)
    cluster_col = args.cluster_col or pick_cluster_col(df)
    print(f"Using cluster column: {cluster_col}")
    reps_df, over_df = compute_reps(emb, df, cluster_col=cluster_col, top_k=args.top_k)
    out_parquet = outp / "cluster_reps.parquet"
    out_over = outp / "cluster_overview.csv"
    reps_df.to_parquet(out_parquet, index=False)
    over_df.to_csv(out_over, index=False)
    print(f"Saved reps → {out_parquet}")
    print(f"Saved overview → {out_over}")

if __name__ == "__main__":
    main()
