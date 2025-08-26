#!/usr/bin/env python3
"""
cluster_and_label.py
First-pass enrichment: cluster -> reps -> LLM-label -> merge -> save labeled parquet.

Inputs:
  --embeddings  embeddings_out/embeddings.npy
  --meta        embeddings_out/reviews_with_embeddings.parquet
Output:
  --labeled_meta  clustering_out/reviews_with_clusters_labeled.parquet

Notes:
- Assumes exactly one clustering method and one label per cluster.
- No index building here; do that later with build_index.py using the labeled parquet.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis_tool import add_derived_cols
from src.clustering_reviews import (
    pca_project, run_kmeans, run_minibatch_kmeans, run_hdbscan, run_microcluster_hdbscan
)
from src.label_clusters_llm import call_openai

REQ_COLS = ["Review_ID","Review_Text","Branch","Reviewer_Location","Year_Month","Rating"]

def choose_cluster_col(method: str) -> str:
    return {
        "kmeans": "cluster_kmeans",
        "minibatch_kmeans": "cluster_mbk",
        "hdbscan": "cluster_hdbscan",
        "micro_hdbscan": "cluster_micro",
    }[method]


def llm_label_clusters_bulk(df: pd.DataFrame, cluster_col: str, model: str = "gpt-4o-mini",
                            per_cluster: int = 5, seed: int = 0) -> pd.DataFrame:
    """
    Randomly sample per cluster, send ALL clusters in ONE LLM call,
    parse robustly, and return a DataFrame with columns:
      [cluster_col, f"{cluster_col}_label", keywords, sentiment, summary, actions, confidence?]
    """
    random.seed(seed)

    # Collect random samples per non-noise cluster
    examples_per_cluster = {}
    for cid in sorted(df[cluster_col].dropna().unique()):
        if int(cid) == -1:
            continue
        texts = df.loc[df[cluster_col] == cid, "Review_Text"].dropna().tolist()
        if not texts:
            continue
        samples = random.sample(texts, min(per_cluster, len(texts)))
        examples_per_cluster[int(cid)] = samples

    if not examples_per_cluster:
        # Nothing to label
        return pd.DataFrame(columns=[cluster_col, f"{cluster_col}_label", "keywords", "sentiment", "summary", "actions"])

    # Build one big prompt
    blocks = []
    for cid, samples in examples_per_cluster.items():
        blocks.append(f"Cluster {cid}:\n" + "\n---\n".join(samples))

    expected_ids = sorted(int(c) for c in df[cluster_col].dropna().unique() if int(c) != -1)

    # IMPORTANT: ask for a JSON OBJECT with a 'clusters' list, since response_format=json_object
    system_prompt = (
        "You are labeling clusters of Disneyland reviews.\n"
        "You MUST return a JSON OBJECT with key 'clusters', whose value is a list with EXACTLY one object PER cluster id provided.\n"
        "Do NOT merge, drop, or add clusters. If a cluster is unclear, still return an object with a generic label.\n"
        "Schema for each object:\n"
        f"- {cluster_col} (int, one of {expected_ids})\n"
        "- label (2-4 words)\n"
        "- keywords (comma-separated)\n"
        "- sentiment (very negative/negative/mixed/positive/very positive)\n"
        "- summary (2-3 sentences)\n"
        "- actions (2-4 short actionable bullets)\n"
        "- confidence (0-100)\n"
    )
    user_prompt = (
            "Sample reviews grouped by cluster follow. For EVERY cluster id listed, produce exactly ONE label object.\n\n"
            + "\n\n".join(blocks)
            + "\n\nRespond with JSON ONLY, exactly this shape:\n"
              "{\n  \"clusters\": [ { ...one per cluster id in the list above... } ]\n}\n"
    )

    content = call_openai((system_prompt, user_prompt), model=model, temperature=0.2)

    # Parse JSON robustly
    try:
        payload = json.loads(content)
        print(payload)
    except Exception:
        payload = json.loads(str(content).strip("` \n"))

    # Normalize payload into a list of rows
    rows = None
    if isinstance(payload, dict):
        # Prefer 'clusters' key
        if "clusters" in payload and isinstance(payload["clusters"], list):
            rows = payload["clusters"]
        else:
            # If there's exactly one list value, take it
            list_vals = [v for v in payload.values() if isinstance(v, list)]
            if len(list_vals) == 1:
                rows = list_vals[0]
            else:
                # dict of dicts keyed by cluster id?
                dict_vals = [v for v in payload.values() if isinstance(v, dict)]
                if dict_vals:
                    tmp = []
                    for k, v in payload.items():
                        if isinstance(v, dict):
                            v = {**v}
                            v.setdefault("cluster", k)
                            tmp.append(v)
                    rows = tmp
    elif isinstance(payload, list):
        rows = payload

    if rows is None:
        # Fallback: empty DF rather than crashing
        return pd.DataFrame(columns=[cluster_col, f"{cluster_col}_label", "keywords", "sentiment", "summary", "actions", "confidence"])

    labels_df = pd.DataFrame(rows)

    # Unify the cluster id column to match cluster_col
    if cluster_col in labels_df.columns:
        pass
    elif "cluster" in labels_df.columns:
        labels_df = labels_df.rename(columns={"cluster": cluster_col})
    elif "cluster_id" in labels_df.columns:
        labels_df = labels_df.rename(columns={"cluster_id": cluster_col})
    elif "cid" in labels_df.columns:
        labels_df = labels_df.rename(columns={"cid": cluster_col})
    else:
        # If still missing, attach by order to the cluster ids we provided
        ordered_ids = sorted(examples_per_cluster.keys())
        labels_df[cluster_col] = ordered_ids[: len(labels_df)]

    # Ensure numeric cluster ids
    labels_df[cluster_col] = pd.to_numeric(labels_df[cluster_col], errors="coerce").astype("Int64")

    # Normalize the label column name
    if "label" in labels_df.columns:
        labels_df = labels_df.rename(columns={"label": f"{cluster_col}_label"})
    elif f"{cluster_col}_label" not in labels_df.columns:
        labels_df[f"{cluster_col}_label"] = None

    # Keep useful columns only (and in stable order)
    keep = [cluster_col, f"{cluster_col}_label", "keywords", "sentiment", "summary", "actions", "confidence"]
    keep = [c for c in keep if c in labels_df.columns]
    labels_df = labels_df[keep].drop_duplicates(subset=[cluster_col]).reset_index(drop=True)

    return labels_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="data/embeddings_out/embeddings.npy")
    ap.add_argument("--meta", default="data/embeddings_out/reviews_with_embeddings.parquet")
    ap.add_argument("--method", choices=["kmeans","minibatch_kmeans","hdbscan","micro_hdbscan"], default="hdbscan")
    ap.add_argument("--pca", type=int, default=50)
    ap.add_argument("--k", type=int, default=30)                 # for kmeans/mbk
    ap.add_argument("--micro_clusters", type=int, default=400)   # for micro_hdbscan
    ap.add_argument("--min_cluster_size", type=int, default=100) # for hdbscan
    ap.add_argument("--min_samples", type=int, default=20)       # for hdbscan
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--per_cluster", type=int, default=12)
    ap.add_argument("--min_chars", type=int, default=40)
    ap.add_argument("--label_model", type=str, default="gpt-4o-mini")

    ap.add_argument("--out_dir", default="data/clustering_out")
    ap.add_argument("--labeled_meta", default="data/clustering_out/reviews_with_clusters_labeled.parquet")
    args = ap.parse_args()

    t0 = time.time()

    # Load base data
    X = np.load(args.embeddings)
    df = pd.read_parquet(args.meta)
    miss = [c for c in REQ_COLS if c not in df.columns]
    if miss:
        raise ValueError(f"Meta parquet missing columns: {miss} — re-run embedding to retain metadata.")
    df = add_derived_cols(df)  # adds Year/Month/Season if missing:contentReference[oaicite:4]{index=4}

    # Clustering
    Xp = pca_project(X, n_components=args.pca, seed=args.seed)         # :contentReference[oaicite:5]{index=5}
    if args.method == "kmeans":
        labels, dur, sil = run_kmeans(Xp, k=args.k, seed=args.seed)    # :contentReference[oaicite:6]{index=6}
        print(f"KMeans: k={args.k} | time={dur:.2f}s | silhouette={sil:.4f}")
    elif args.method == "minibatch_kmeans":
        labels, dur, sil = run_minibatch_kmeans(Xp, k=args.k, seed=args.seed)  # :contentReference[oaicite:7]{index=7}
        print(f"MiniBatchKMeans: k={args.k} | time={dur:.2f}s | silhouette={sil:.4f}")
    elif args.method == "hdbscan":
        labels, dur = run_hdbscan(Xp, min_cluster_size=args.min_cluster_size, min_samples=args.min_samples)  # :contentReference[oaicite:8]{index=8}
        print(f"HDBSCAN: mcs={args.min_cluster_size}, ms={args.min_samples} | time={dur:.2f}s | noise={(labels==-1).sum()}")
    else:
        labels, dur, _ = run_microcluster_hdbscan(
            Xp, m=args.micro_clusters,
            min_cluster_size=args.min_cluster_size, min_samples=args.min_samples,
            seed=args.seed
        )                                                                 # :contentReference[oaicite:9]{index=9}
        print(f"Micro-HDBSCAN: m={args.micro_clusters} | time={dur:.2f}s | noise={(labels==-1).sum()}")

    cluster_col = choose_cluster_col(args.method)
    df[cluster_col] = labels


    labels_df = llm_label_clusters_bulk(df, cluster_col, model=args.label_model, per_cluster=args.per_cluster)


    # Merge labels onto the reviews
    df_out = df.merge(labels_df, on=cluster_col, how="left")

    # Save labeled parquet
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    labeled_path = Path(args.labeled_meta)
    labeled_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(labeled_path, index=False)
    print(f"Saved labeled parquet → {labeled_path}")
    print(f"Done in {(time.time()-t0):.1f}s.")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not set; LLM labeling will fail.")
    main()
