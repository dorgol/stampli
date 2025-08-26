#!/usr/bin/env python3
"""
query_parser.py
Parse a natural-language question into structured filters using OpenAI JSON Schema (structured outputs),
with dynamic allowed values loaded from the labeled parquet.

Usage (PowerShell/cmd):
  $env:OPENAI_API_KEY="sk-..."
  python src/query_parser.py --question "Is the staff in Paris friendly?"

Options:
  --labeled_parquet  Override the parquet path (defaults to clustering_out/reviews_with_clusters_labeled.parquet)
  --print-where      Also print a Chroma 'where' filter dict (equality / $in only; no $contains)
"""

from __future__ import annotations
import argparse
import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI

# ---------- Config ----------
DEFAULT_PARQUET = "clustering_out/reviews_with_clusters_labeled.parquet"
SEASONS = ["Spring", "Summer", "Fall", "Winter"]

SYSTEM_MSG = (
    "You are a strict JSON generator. Return ONLY a JSON object matching the provided schema. "
    "Prefer specific filters if the question clearly implies them; otherwise leave fields null. "
    "Do not invent values outside the allowed lists. Keep 'focus' as a short theme like 'queues', 'staff', 'food'."
)

# ---------- Catalog loading (from parquet) ----------

def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0

@lru_cache(maxsize=1)
def _load_catalog_cached(path: str, mtime: float, top_n_labels: int = 80):
    """
    Load a lightweight catalog from the labeled parquet (no nrows, pandas/pyarrow compatible).
    """
    # 1) Discover available columns (pyarrow schema if possible)
    try:
        import pyarrow.parquet as pq
        cols_available = set(pq.ParquetFile(path).schema.names)
    except Exception:
        import pandas as pd  # local import to avoid shadowing
        df_all = pd.read_parquet(path)
        cols_available = set(df_all.columns)

    want_all = [
        "Branch", "Reviewer_Location", "Year", "Month", "Season", "sentiment",
        "cluster_mbk", "cluster_mbk_label",
        "cluster_micro", "cluster_micro_label",
        "cluster_kmeans", "cluster_kmeans_label",
        "cluster_hdbscan", "cluster_hdbscan_label",
    ]
    want_cols = [c for c in want_all if c in cols_available]

    import pandas as pd
    df = pd.read_parquet(path, columns=want_cols)

    # 2) Detect active cluster columns
    label_cols = [c for c in df.columns if c.endswith("_label") and df[c].notna().any()]
    label_col = max(label_cols, key=lambda c: df[c].notna().sum()) if label_cols else None
    id_col = label_col[:-6] if (label_col and label_col[:-6] in df.columns) else None

    # 3) Build allowed values
    branches   = sorted(df["Branch"].dropna().unique().tolist()) if "Branch" in df else []
    countries  = sorted(df["Reviewer_Location"].dropna().unique().tolist()) if "Reviewer_Location" in df else []
    sentiments = sorted(df["sentiment"].dropna().unique().tolist()) if "sentiment" in df else []
    years      = sorted(int(y) for y in df["Year"].dropna().unique().tolist()) if "Year" in df else []
    seasons    = sorted(df["Season"].dropna().unique().tolist()) if "Season" in df else ["Spring","Summer","Fall","Winter"]

    # Use value_counts() → head() → index.tolist() (compatible with old pandas)
    cluster_labels = []
    if label_col:
        cluster_labels = (
            df[label_col]
            .dropna()
            .astype(str)
            .value_counts()
            .head(top_n_labels)
            .index
            .tolist()
        )

    return {
        "path": path,
        "mtime": mtime,
        "cluster_id_col": id_col,
        "cluster_label_col": label_col,
        "branches": branches,
        "countries": countries,
        "sentiments": sentiments,
        "years": years,
        "seasons": seasons,
        "cluster_labels": cluster_labels,
    }

def load_catalog(parquet_path: str = DEFAULT_PARQUET, top_n_labels: int = 80) -> Dict[str, Any]:
    """Auto-refreshing catalog built from the labeled parquet."""
    return _load_catalog_cached(parquet_path, _mtime(parquet_path), top_n_labels)

# ---------- Dynamic JSON Schema ----------

def build_dynamic_schema(cat: Dict[str, Any]) -> Dict[str, Any]:
    """JSON Schema used for structured outputs (Draft-07)."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "DisneyReviewQuery",
        "type": "object",
        "properties": {
            "branch": {
                "description": "Target parks (array or null).",
                "type": ["array", "null"],
                "items": {"type": "string", "enum": cat["branches"]},
            },
            "countries": {
                "description": "Reviewer locations (array or null).",
                "type": ["array", "null"],
                "items": {"type": "string", "enum": cat["countries"]},
            },
            "seasons": {
                "type": ["array", "null"],
                "items": {"type": "string", "enum": cat["seasons"]},
            },
            "months": {
                "type": ["array", "null"],
                "items": {"type": "integer", "minimum": 1, "maximum": 12},
            },
            "years": {
                "type": ["array", "null"],
                "items": {"type": "integer", "enum": cat["years"]},
            },
            "rating_gte": {"type": ["number", "null"], "minimum": 1, "maximum": 5},
            "cluster_label": {
                "type": ["string", "null"],
                "enum": cat["cluster_labels"] + [None] if cat["cluster_labels"] else [None],
            },
            "cluster_id": {"type": ["integer", "null"], "minimum": -1, "maximum": 1000000},
            "sentiment": {
                "type": ["string", "null"],
                "enum": cat["sentiments"] + [None] if cat["sentiments"] else [None],
            },
            "focus": {"type": ["string", "null"], "minLength": 1},
        },
        "required": ["branch", "countries", "seasons", "months", "years", "rating_gte",
                     "cluster_label", "cluster_id", "sentiment", "focus"],
        "additionalProperties": False,
    }

# ---------- OpenAI call + coercion ----------

def parse_question(question: str, parquet_path: str = DEFAULT_PARQUET,
                   model: str = "gpt-4o-mini") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns (parsed_filters, catalog_used). Filters are already coerced to allowed values.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set.")

    cat = load_catalog(parquet_path)
    schema = build_dynamic_schema(cat)

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": f"Question: {question}\nReturn JSON only."},
        ],
        temperature=0.0,
        max_tokens=500,
        response_format={"type": "json_schema",
                         "json_schema": {"name": "DisneyReviewQuery", "strict": True, "schema": schema}},
    )

    raw = json.loads(resp.choices[0].message.content)

    # Coerce to allowed lists/values (defensive)
    out: Dict[str, Any] = {
        "branch": _coerce_list(raw.get("branch"), cat["branches"]),
        "countries": _coerce_list(raw.get("countries"), cat["countries"]),
        "seasons": _coerce_list(raw.get("seasons"), cat["seasons"]),
        "months": _coerce_int_list(raw.get("months"), 1, 12),
        "years": _coerce_list(raw.get("years"), cat["years"]),
        "rating_gte": _coerce_num(raw.get("rating_gte"), 1, 5),
        "cluster_label": raw.get("cluster_label") if raw.get("cluster_label") in cat["cluster_labels"] else None,
        "cluster_id": _coerce_int(raw.get("cluster_id")),
        "sentiment": raw.get("sentiment") if raw.get("sentiment") in cat["sentiments"] else None,
        "focus": raw.get("focus"),
    }
    return out, cat

def _coerce_list(vals, allowed: List[Any]) -> Optional[List[Any]]:
    if vals is None:
        return None
    if isinstance(vals, (str, int)):
        vals = [vals]
    try:
        filtered = [v for v in vals if (allowed is None or not allowed) or v in allowed]
        return filtered if filtered else None
    except Exception:
        return None

def _coerce_int_list(vals, minv: int, maxv: int) -> Optional[List[int]]:
    if vals is None:
        return None
    if isinstance(vals, int):
        vals = [vals]
    out = []
    for v in vals:
        try:
            x = int(v)
            if minv <= x <= maxv:
                out.append(x)
        except Exception:
            pass
    return out if out else None

def _coerce_num(v, minv: float, maxv: float) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
        if minv <= x <= maxv:
            return x
    except Exception:
        pass
    return None

def _coerce_int(v) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None

# ---------- Optional: build a Chroma 'where' dict (eq / $in only) ----------

def build_where_from_filters(filters: Dict[str, Any], catalog: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Build a Chroma-compliant 'where' filter using equality / $in only (no $contains).
    Uses the active cluster id/label columns detected from parquet.
    """
    clauses: List[Dict[str, Any]] = []

    def one_or_in(field: str, vals):
        if not vals:
            return None
        if len(vals) == 1:
            return {field: vals[0]}
        return {field: {"$in": vals}}

    v = one_or_in("Branch", filters.get("branch"));                v and clauses.append(v)
    v = one_or_in("Reviewer_Location", filters.get("countries"));  v and clauses.append(v)
    v = one_or_in("Season", filters.get("seasons"));               v and clauses.append(v)
    v = one_or_in("Month", filters.get("months"));                 v and clauses.append(v)
    v = one_or_in("Year", filters.get("years"));                   v and clauses.append(v)

    if filters.get("rating_gte") is not None:
        clauses.append({"Rating": {"$gte": float(filters["rating_gte"])}})

    # Cluster id / label on active columns
    id_col, lbl_col = catalog.get("cluster_id_col"), catalog.get("cluster_label_col")
    if filters.get("cluster_id") is not None and id_col:
        clauses.append({id_col: int(filters["cluster_id"])})
    if filters.get("cluster_label") and lbl_col:
        clauses.append({lbl_col: {"$in": [filters["cluster_label"]]}})

    if filters.get("sentiment"):
        clauses.append({"sentiment": filters["sentiment"]})

    # compact
    clauses = [c for c in clauses if c]
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--labeled_parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--print-where", action="store_true")
    args = ap.parse_args()

    filters, catalog = parse_question(args.question, parquet_path=args.labeled_parquet, model=args.model)
    print(json.dumps({"filters": filters}, indent=2))

    if args.__dict__.get("print_where"):
        where = build_where_from_filters(filters, catalog)
        print("\nwhere =")
        print(json.dumps(where, indent=2))

if __name__ == "__main__":
    main()
