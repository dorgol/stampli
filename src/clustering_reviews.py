#!/usr/bin/env python3
"""
clustering_reviews.py
Cluster Disneyland review embeddings using multiple strategies (choose via --method).

Inputs (from the embedding step):
- embeddings_out/embeddings.npy
- embeddings_out/reviews_with_embeddings.parquet

Outputs:
- clustering_out/reviews_with_clusters.parquet            (Review_ID, Review_Text, emb_index, cluster_* columns)
- clustering_out/cluster_summary_<method>.csv             (cluster id, n, share)
- clustering_out/centroid_assignments.parquet             (for micro-cluster pipeline; optional)
- console prints of timing + silhouette (for KMeans)
"""

import argparse
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

# Optional: HDBSCAN (pip install hdbscan pynndescent)
try:
    import hdbscan  # type: ignore
    _HAS_HDBSCAN = True
except Exception:
    _HAS_HDBSCAN = False


def load_inputs(emb_path: str, parquet_path: str):
    X = np.load(emb_path)
    df = pd.read_parquet(parquet_path)
    return X, df


def pca_project(X: np.ndarray, n_components: int = 50, seed: int = 0) -> np.ndarray:
    """L2-normalize then project to PCA dims to speed up clustering & tame high-dim noise."""
    Xn = normalize(X)
    Xp = PCA(n_components=n_components, random_state=seed).fit_transform(Xn)
    return Xp


def run_kmeans(X: np.ndarray, k: int = 30, seed: int = 0):
    t0 = time.time()
    km = KMeans(n_clusters=k, n_init="auto", random_state=seed)
    labels = km.fit_predict(X)
    dur = time.time() - t0
    sil = silhouette_score(X, labels) if len(set(labels)) > 1 else float("nan")
    return labels, dur, sil


def run_minibatch_kmeans(X: np.ndarray, k: int = 30, seed: int = 0, batch_size: int = 4096):
    t0 = time.time()
    km = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, random_state=seed)
    labels = km.fit_predict(X)
    dur = time.time() - t0
    sil = silhouette_score(X, labels) if len(set(labels)) > 1 else float("nan")
    return labels, dur, sil


def run_hdbscan(X: np.ndarray, min_cluster_size: int = 100, min_samples: int = 20):
    if not _HAS_HDBSCAN:
        raise RuntimeError("hdbscan not installed. Try: pip install hdbscan pynndescent")
    t0 = time.time()
    cl = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        core_dist_n_jobs=-1,
        prediction_data=True,
    )
    labels = cl.fit_predict(X)
    dur = time.time() - t0
    return labels, dur


def run_microcluster_hdbscan(
    X: np.ndarray,
    m: int = 400,
    min_cluster_size: int = 10,
    min_samples: int = 5,
    seed: int = 0,
):
    """
    Two-stage: MiniBatchKMeans to m micro-clusters, then HDBSCAN on centroids, then map back.
    Much faster while preserving density-based grouping behavior.
    """
    if not _HAS_HDBSCAN:
        raise RuntimeError("hdbscan not installed. Try: pip install hdbscan pynndescent")
    t0 = time.time()
    mbk = MiniBatchKMeans(n_clusters=m, batch_size=4096, random_state=seed).fit(X)
    centroids = mbk.cluster_centers_
    cl = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        core_dist_n_jobs=-1,
        prediction_data=True,
    ).fit(centroids)
    centroid_labels = cl.labels_
    point_labels = centroid_labels[mbk.labels_]
    dur = time.time() - t0
    return point_labels, dur, centroid_labels


def summarize(labels: np.ndarray, method_name: str, out_dir: Path):
    counts = Counter(labels)
    n = len(labels)
    rows = [{"cluster": int(lab), "n": int(cnt), "share": cnt / n} for lab, cnt in counts.items()]
    summary = pd.DataFrame(rows).sort_values("n", ascending=False)
    out_csv = out_dir / f"cluster_summary_{method_name}.csv"
    summary.to_csv(out_csv, index=False)
    return summary, out_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=str, default="embeddings_out/embeddings.npy")
    ap.add_argument("--parquet", type=str, default="embeddings_out/reviews_with_embeddings.parquet")
    ap.add_argument("--out_dir", type=str, default="clustering_out")
    ap.add_argument("--method", type=str, default="hdbscan",
                    choices=["kmeans", "minibatch_kmeans", "hdbscan", "micro_hdbscan"])
    ap.add_argument("--k", type=int, default=30, help="for kmeans/minibatch_kmeans")
    ap.add_argument("--pca", type=int, default=50, help="PCA dimensions before clustering")
    ap.add_argument("--min_cluster_size", type=int, default=100, help="for HDBSCAN")
    ap.add_argument("--min_samples", type=int, default=20, help="for HDBSCAN")
    ap.add_argument("--micro_clusters", type=int, default=400, help="for micro_hdbscan")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge_labels", action="store_true",
                    help="If set, merge LLM labels from --labels_csv onto the clustered reviews")
    ap.add_argument("--labels_csv", type=str, default="labeling_out/cluster_labels.csv",
                    help="CSV produced by label_clusters_llm.py (columns: cluster,label,...)")
    args = ap.parse_args()

    outp = Path(args.out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    print("Loading inputs...")
    X, df = load_inputs(args.embeddings, args.parquet)
    print(f"Embeddings shape: {X.shape}, reviews: {len(df)}")

    print(f"PCA → {args.pca} dims (L2-normalize first)...")
    Xp = pca_project(X, n_components=args.pca, seed=args.seed)

    if args.method == "kmeans":
        labels, dur, sil = run_kmeans(Xp, k=args.k, seed=args.seed)
        print(f"KMeans: k={args.k} | time={dur:.2f}s | silhouette={sil:.4f}")
        method_name = f"kmeans_k{args.k}"
        df["cluster_kmeans"] = labels

    elif args.method == "minibatch_kmeans":
        labels, dur, sil = run_minibatch_kmeans(Xp, k=args.k, seed=args.seed)
        print(f"MiniBatchKMeans: k={args.k} | time={dur:.2f}s | silhouette={sil:.4f}")
        method_name = f"minibatch_k{args.k}"
        df["cluster_mbk"] = labels

    elif args.method == "hdbscan":
        labels, dur = run_hdbscan(Xp, min_cluster_size=args.min_cluster_size, min_samples=args.min_samples)
        n_noise = int((labels == -1).sum())
        print(f"HDBSCAN: min_cluster_size={args.min_cluster_size}, min_samples={args.min_samples} | time={dur:.2f}s | noise={n_noise}")
        method_name = f"hdbscan_mcs{args.min_cluster_size}_ms{args.min_samples}"
        df["cluster_hdbscan"] = labels

    else:  # micro_hdbscan
        labels, dur, centroid_labels = run_microcluster_hdbscan(
            Xp, m=args.micro_clusters,
            min_cluster_size=args.min_cluster_size, min_samples=args.min_samples,
            seed=args.seed
        )
        n_noise = int((labels == -1).sum())
        print(f"Micro(HDBSCAN on {args.micro_clusters} centroids): min_cluster_size={args.min_cluster_size}, min_samples={args.min_samples} | time={dur:.2f}s | noise={n_noise}")
        method_name = f"micro_m{args.micro_clusters}_mcs{args.min_cluster_size}_ms{args.min_samples}"
        df["cluster_micro"] = labels
        # Save centroid labels for transparency
        cent_path = outp / "centroid_assignments.parquet"
        pd.DataFrame({"centroid_label": centroid_labels}).to_parquet(cent_path, index=False)
        print(f"Saved centroid labels → {cent_path}")

    # Summaries
    col_map = {
        "kmeans": "cluster_kmeans",
        "minibatch_kmeans": "cluster_mbk",
        "hdbscan": "cluster_hdbscan",
        "micro_hdbscan": "cluster_micro",
    }
    lab_col = col_map[args.method]
    summary, csv_path = summarize(df[lab_col].values, method_name, outp)
    print("Top clusters:\n", summary.head(15))

    # Save clustered reviews
    out_parquet = outp / "reviews_with_clusters.parquet"
    df.to_parquet(out_parquet, index=False)
    print(f"Saved: {out_parquet}")
    print(f"Saved: {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
