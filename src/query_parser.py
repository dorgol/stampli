#!/usr/bin/env python3
"""
query_parser.py
Parse a natural-language question into structured filters using OpenAI structured outputs,
with dynamic allowed values loaded from the labeled parquet.

Public API:
  - load_catalog(parquet_path=DEFAULT_PARQUET, top_n_labels=80) -> Dict[str, Any]
  - parse_question(question, parquet_path=DEFAULT_PARQUET, model="gpt-4o-mini") -> (filters, catalog)
  - build_where_from_filters(filters, catalog) -> Optional[Dict[str, Any]]

Notes:
- Uses the active cluster label column detected from parquet (the *_label column with most non-nulls).
- Falls back to a lightweight heuristic parser if OPENAI_API_KEY is not set.
- Output filter keys are pluralized to align with apply_filters() and server expectations:
    branch (List[str] or None),
    countries, months, seasons, years (List[...] or None),
    rating_gte (float or None),
    sentiment (str or None),
    cluster_id (int or None),
    cluster_label (str or None)
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

# Local retrieval builder for Chroma 'where'
from src.retrieval import build_where as build_where_base

# ---------------- Config ----------------

DEFAULT_PARQUET = "clustering_out/reviews_with_clusters_labeled.parquet"
SEASONS = ["Spring", "Summer", "Fall", "Winter"]

SYSTEM_MSG = (
    "You are a strict JSON generator for parsing user questions about Disneyland reviews. "
    "Return ONLY a JSON object that matches the provided JSON Schema. "
    "Prefer specific filters if the question clearly implies them; otherwise leave fields null. "
    "Do not invent values outside the allowed lists. "
    "Keep 'focus' as a short theme like 'queues', 'staff', 'food' when implied; otherwise null."
)


# ---------------- Catalog loading (from labeled parquet) ----------------

def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0

@lru_cache(maxsize=1)
def _load_catalog_cached(path: str, mtime: float, top_n_labels: int = 80) -> Dict[str, Any]:
    """
    Load a lightweight catalog from the labeled parquet without reading full text columns.
    Detect the active cluster label column and provide allowed values for schema.
    """
    # Try to read only relevant columns (fast path with pyarrow if available)
    try:
        import pyarrow.parquet as pq  # type: ignore
        pf = pq.ParquetFile(path)
        cols_available = set(pf.schema.names)
    except Exception:
        df_tmp = pd.read_parquet(path)
        cols_available = set(df_tmp.columns)

    want_all = [
        "Branch", "Reviewer_Location", "Year", "Month", "Season", "sentiment",
        # cluster id/label possibilities:
        "cluster_mbk", "cluster_mbk_label",
        "cluster_micro", "cluster_micro_label",
        "cluster_kmeans", "cluster_kmeans_label",
        "cluster_hdbscan", "cluster_hdbscan_label",
    ]
    want_cols = [c for c in want_all if c in cols_available]
    df = pd.read_parquet(path, columns=want_cols)  # small frame with only metadata

    # Detect active cluster label column = *_label with most non-nulls
    label_cols = [c for c in df.columns if c.endswith("_label") and df[c].notna().any()]
    label_col = max(label_cols, key=lambda c: df[c].notna().sum()) if label_cols else None
    id_col = label_col[:-6] if (label_col and label_col[:-6] in df.columns) else None

    # Allowed values
    branches   = sorted(df["Branch"].dropna().unique().tolist()) if "Branch" in df else []
    countries  = sorted(df["Reviewer_Location"].dropna().unique().tolist()) if "Reviewer_Location" in df else []
    sentiments = sorted(df["sentiment"].dropna().unique().tolist()) if "sentiment" in df else []
    years      = sorted(int(y) for y in df["Year"].dropna().unique().tolist()) if "Year" in df else []
    seasons    = sorted(df["Season"].dropna().unique().tolist()) if "Season" in df else SEASONS

    # Most frequent cluster labels (cap list size for schema readability)
    cluster_labels: List[str] = []
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
        "cluster_id_col": id_col,               # e.g., "cluster_micro"
        "cluster_label_col": label_col,         # e.g., "cluster_micro_label"
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


# ---------------- JSON Schema for structured output ----------------

def _build_dynamic_schema(cat: Dict[str, Any]) -> Dict[str, Any]:
    """
    JSON Schema (Draft-07) for structured parser output.
    All fields are optional (nullable) and constrained to catalog sets where relevant.
    """
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "DisneyReviewQuery",
        "type": "object",
        "properties": {
            "branch": {
                "description": "Target park(s); array or null.",
                "type": ["array", "null"],
                "items": {"type": "string", "enum": cat.get("branches", [])}
            },
            "countries": {
                "description": "Reviewer origin country/countries; array or null.",
                "type": ["array", "null"],
                "items": {"type": "string", "enum": cat.get("countries", [])}
            },
            "months": {
                "description": "Target month numbers (1-12); array or null.",
                "type": ["array", "null"],
                "items": {"type": "integer", "minimum": 1, "maximum": 12}
            },
            "seasons": {
                "description": "Target season(s); array or null.",
                "type": ["array", "null"],
                "items": {"type": "string", "enum": cat.get("seasons", SEASONS)}
            },
            "years": {
                "description": "Target years; array or null.",
                "type": ["array", "null"],
                "items": {"type": "integer"}
            },
            "rating_gte": {
                "description": "Minimum rating threshold (e.g., 4 ⇒ positive).",
                "type": ["number", "null"]
            },
            "sentiment": {
                "description": "Coarse sentiment if explicitly mentioned (e.g., 'positive', 'very negative').",
                "type": ["string", "null"],
                "enum": cat.get("sentiments", []) + [None]
            },
            "cluster_id": {
                "description": "Exact cluster id if explicitly referenced.",
                "type": ["integer", "null"]
            },
            "cluster_label": {
                "description": "Exact cluster label if explicitly referenced.",
                "type": ["string", "null"],
                "enum": cat.get("cluster_labels", []) + [None]
            },
            "focus": {
                "description": "Optional short focus/theme (e.g., 'queues', 'staff', 'food').",
                "type": ["string", "null"]
            }
        },
        "additionalProperties": False
    }


# ---------------- Parser ----------------

def _llm_parse(question: str, cat: Dict[str, Any], model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Use OpenAI structured outputs to parse the question into filters.
    Returns a dict that conforms to the dynamic JSON schema.
    """
    from openai import OpenAI  # lazy import

    schema = _build_dynamic_schema(cat)
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {
                "role": "user",
                "content": (
                    "Question:\n" + question + "\n\n"
                    "Return a JSON object with keys:\n"
                    + json.dumps(list(schema["properties"].keys()))
                    + "\n\nSchema:\n" + json.dumps(schema)
                ),
            },
        ],
    )
    content = resp.choices[0].message.content
    try:
        parsed = json.loads(content)
    except Exception:
        parsed = json.loads(str(content).strip("` \n"))
    return parsed


def _heuristic_parse(question: str, cat: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fallback: very light heuristics when OPENAI_API_KEY is unavailable.
    Tries to grab branch names, months/seasons, basic sentiment cues.
    """
    q = question.lower()

    def pick(items: List[str]) -> List[str]:
        out = []
        for it in items:
            if it and it.lower() in q:
                out.append(it)
        return sorted(set(out))

    # months (by name) and simple numbers 1-12
    month_map = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
    }
    months = []
    for name, num in month_map.items():
        if name in q:
            months.append(num)
    # digits
    for m in range(1, 13):
        if re.search(rf"\b{m}\b", q):
            months.append(m)
    months = sorted(set(months)) or None

    seasons = pick(cat.get("seasons", SEASONS)) or None
    branches = pick(cat.get("branches", [])) or None
    countries = pick(cat.get("countries", [])) or None

    rating_gte = None
    if "positive" in q or "good" in q or "great" in q:
        rating_gte = 4.0
    sentiment = None
    if "friendly" in q:  # toy example
        sentiment = "positive" if "positive" in cat.get("sentiments", []) else None

    return {
        "branch": branches,
        "countries": countries,
        "months": months,
        "seasons": seasons,
        "years": None,
        "rating_gte": rating_gte,
        "sentiment": sentiment,
        "cluster_id": None,
        "cluster_label": None,
        "focus": None,
    }


def _normalize_filters(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure the parser output uses the canonical keys & types expected downstream.
    All list-like values become lists; empty lists -> None.
    """
    def tolist(x: Union[None, str, int, float, List[Any]]) -> Optional[List[Any]]:
        if x is None:
            return None
        if isinstance(x, list):
            return x or None
        return [x]

    out = {
        "branch": tolist(raw.get("branch")),
        "countries": tolist(raw.get("countries")),
        "months": tolist(raw.get("months")),
        "seasons": tolist(raw.get("seasons")),
        "years": tolist(raw.get("years")),
        "rating_gte": float(raw["rating_gte"]) if isinstance(raw.get("rating_gte"), (int, float, str)) and str(raw.get("rating_gte")).strip() != "" else None,
        "sentiment": (str(raw.get("sentiment")).strip() or None) if raw.get("sentiment") is not None else None,
        "cluster_id": int(raw["cluster_id"]) if isinstance(raw.get("cluster_id"), (int, float, str)) and str(raw.get("cluster_id")).strip() != "" else None,
        "cluster_label": (str(raw.get("cluster_label")).strip() or None) if raw.get("cluster_label") is not None else None,
        "focus": (str(raw.get("focus")).strip() or None) if raw.get("focus") is not None else None,
    }
    return out


def parse_question(question: str,
                   parquet_path: str = DEFAULT_PARQUET,
                   model: str = "gpt-4o-mini") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Parse question -> (filters, catalog).
    If OPENAI_API_KEY is set, use structured outputs; otherwise, heuristic fallback.
    """
    cat = load_catalog(parquet_path)
    if os.getenv("OPENAI_API_KEY"):
        raw = _llm_parse(question, cat, model=model)
    else:
        raw = _heuristic_parse(question, cat)
    filters = _normalize_filters(raw)

    # If the LLM returned a single scalar for branch, ensure we keep list type
    # and also coerce impossible values to None based on catalog
    def scrub_list(vals: Optional[List[Any]], allowed: List[Any]) -> Optional[List[Any]]:
        if not vals:
            return None
        cleaned = [v for v in vals if v in allowed]
        return cleaned or None

    filters["branch"]    = scrub_list(filters.get("branch"),    cat.get("branches", []))
    filters["countries"] = scrub_list(filters.get("countries"), cat.get("countries", []))
    filters["seasons"]   = scrub_list(filters.get("seasons"),   cat.get("seasons", SEASONS))

    return filters, cat


# ---------------- Where-builder (with cluster + sentiment support) ----------------

def build_where_from_filters(filters: Dict[str, Any],
                             catalog: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert parsed filters → Chroma where dict.
    - Uses retrieval.build_where for core fields (Branch, Reviewer_Location, Year, Month, Season, Rating).
    - Adds sentiment equality if provided.
    - Adds cluster_id equality (against active cluster id column) if provided.
    - Adds cluster_label equality (against active cluster label column) if provided.
    """
    # Core where from base builder
    core = build_where_base(
        branch=filters.get("branch"),
        country=filters.get("countries"),
        month=filters.get("months"),
        season=filters.get("seasons"),
        year=filters.get("years"),
        rating_gte=filters.get("rating_gte"),
    )

    clauses: List[Dict[str, Any]] = []
    if core:
        # If core is a single dict or an $and, normalize to a list of dicts
        if "$and" in core:
            clauses.extend(core["$and"])
        else:
            clauses.append(core)

    # Sentiment
    if filters.get("sentiment"):
        clauses.append({"sentiment": {"$eq": filters["sentiment"]}})

    # Cluster constraints
    cid_col = catalog.get("cluster_id_col")
    clab_col = catalog.get("cluster_label_col")
    if cid_col and filters.get("cluster_id") is not None:
        clauses.append({cid_col: {"$eq": int(filters["cluster_id"])}})
    if clab_col and filters.get("cluster_label"):
        clauses.append({clab_col: {"$eq": str(filters["cluster_label"])}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
