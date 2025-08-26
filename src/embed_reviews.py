#!/usr/bin/env python3
"""
embed_reviews.py
Create embeddings for Disneyland reviews (text + ids) using either OpenAI or SentenceTransformers.

Outputs:
- {out_dir}/reviews_with_embeddings.parquet   (Review_ID, Review_Text, emb_index)
- {out_dir}/embeddings.npy                    (float32 array [n, d])
- {out_dir}/embeddings_with_ids.npz           (compressed, contains ids + embeddings)
"""

from pathlib import Path
from typing import List, Optional, Iterable, Tuple
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd

def _clean_text(s: str) -> str:
    s = (s or "").strip()
    return " ".join(s.split())

def _read_csv_permissive(path: str) -> pd.DataFrame:
    # Try a few common encodings
    tried = []
    for enc in [None, "utf-8", "utf-8-sig", "cp1252", "ISO-8859-1"]:
        try:
            return pd.read_csv(path, encoding=enc) if enc else pd.read_csv(path)
        except Exception as e:
            tried.append((enc or "default", str(e)))
            continue
    raise RuntimeError(f"Failed to read {path} with common encodings. Errors tried={tried}")

def read_reviews(input_path: str,
                 text_col: str = "Review_Text",
                 id_col: str = "Review_ID",
                 min_chars: int = 5,
                 max_rows: Optional[int] = None) -> pd.DataFrame:
    df = _read_csv_permissive(input_path)
    if max_rows is not None:
        df = df.head(max_rows).copy()
    # Ensure columns exist
    if text_col not in df.columns:
        raise KeyError(f"Missing text_col='{text_col}' in CSV.")
    # Clean/normalize
    df[text_col] = df[text_col].astype(str).map(_clean_text)
    df = df[df[text_col].str.len() >= min_chars].copy()
    # Ensure ID
    if id_col not in df.columns:
        df[id_col] = np.arange(len(df))
    # Deduplicate IDs
    df = df.drop_duplicates(subset=[id_col]).reset_index(drop=True)
    return df

def iter_batches(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i+batch_size]

def embed_openai(texts: List[str], model: str = "text-embedding-3-small",
                 batch_size: int = 256, max_retries: int = 5, base_sleep: float = 2.0) -> np.ndarray:
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("OpenAI python SDK not installed. Try: pip install openai") from e
    if os.getenv("OPENAI_API_KEY") is None:
        raise EnvironmentError("OPENAI_API_KEY not set in environment.")
    client = OpenAI()
    out: List[List[float]] = []
    for batch in iter_batches(texts, batch_size):
        for attempt in range(max_retries + 1):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                out.extend([d.embedding for d in resp.data])
                break
            except Exception as e:
                if attempt >= max_retries:
                    raise
                time.sleep(base_sleep * (2 ** attempt))
    arr = np.asarray(out, dtype=np.float32)
    return arr

def embed_st(texts: List[str], model_name: str = "all-MiniLM-L6-v2",
             batch_size: int = 256) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        raise RuntimeError("SentenceTransformers not installed. Try: pip install sentence-transformers") from e
    mdl = SentenceTransformer(model_name)
    vecs = mdl.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    return vecs.astype(np.float32)

def save_outputs(df: pd.DataFrame,
                 embeddings: np.ndarray,
                 out_dir: str,
                 id_col: str = "Review_ID",
                 text_col: str = "Review_Text") -> Tuple[Path, Path, Path]:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    npy_path = outp / "embeddings.npy"
    np.save(npy_path, embeddings)
    npz_path = outp / "embeddings_with_ids.npz"
    np.savez_compressed(npz_path, ids=df[id_col].values, embeddings=embeddings)
    parquet_path = outp / "reviews_with_embeddings.parquet"
    df_out = df.copy()
    df_out["emb_index"] = np.arange(len(df_out))
    df_out.to_parquet(parquet_path, index=False)
    return parquet_path, npy_path, npz_path

def main():
    parser = argparse.ArgumentParser(description="Create embeddings for Disneyland reviews")
    parser.add_argument("--input", type=str, required=True, help="Path to CSV (e.g., DisneylandReviews.csv)")
    parser.add_argument("--out_dir", type=str, default="embeddings_out", help="Output directory")
    parser.add_argument("--id_col", type=str, default="Review_ID")
    parser.add_argument("--text_col", type=str, default="Review_Text")
    parser.add_argument("--min_chars", type=int, default=5)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--backend", type=str, default="openai", choices=["openai", "sentence-transformers"])
    parser.add_argument("--st_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--oa_model", type=str, default="text-embedding-3-small")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    df = read_reviews(args.input, text_col=args.text_col, id_col=args.id_col,
                      min_chars=args.min_chars, max_rows=args.max_rows)
    texts = df[args.text_col].tolist()

    if args.backend == "openai":
        print(f"Embedding {len(texts)} reviews with OpenAI model={args.oa_model} ...")
        embs = embed_openai(texts, model=args.oa_model, batch_size=args.batch_size)
    else:
        print(f"Embedding {len(texts)} reviews with SentenceTransformers model={args.st_model} ...")
        embs = embed_st(texts, model_name=args.st_model, batch_size=args.batch_size)

    parquet_path, npy_path, npz_path = save_outputs(df, embs, args.out_dir,
                                                    id_col=args.id_col, text_col=args.text_col)
    print(f"Saved: {parquet_path}")
    print(f"Saved: {npy_path}")
    print(f"Saved: {npz_path}")
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Example usage:\n"
              "  export OPENAI_API_KEY=... \n"
              "  python embed_reviews.py --input DisneylandReviews.csv --backend openai --out_dir embeddings_out\n"
              "or local:\n"
              "  pip install sentence-transformers\n"
              "  python embed_reviews.py --input DisneylandReviews.csv --backend sentence-transformers --st_model all-MiniLM-L6-v2\n")
    else:
        main()
