from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.analysis_tool import apply_filters, compute_metrics
from src.query_parser import parse_question, build_where_from_filters, DEFAULT_PARQUET
from src.retrieval import embed_query_openai, retrieve_snippets
from src.synthesis import synthesize_answer

# ---- config ----
CHROMA_PATH      = os.getenv("CHROMA_PATH", "index/chroma_db")
COLLECTION       = os.getenv("CHROMA_COLLECTION", "reviews")
LABELED_PARQUET  = os.getenv("LABELED_PARQUET", DEFAULT_PARQUET)

# ---- tiny parquet cache (reloads when file changes) ----
_DF_CACHE: Dict[str, Any] = {"df": None, "mtime": None}
def get_parquet_df() -> pd.DataFrame:
    mtime = os.path.getmtime(LABELED_PARQUET)
    if _DF_CACHE["df"] is None or _DF_CACHE["mtime"] != mtime:
        _DF_CACHE["df"] = pd.read_parquet(LABELED_PARQUET)
        _DF_CACHE["mtime"] = mtime
    return _DF_CACHE["df"]

# ---- API models ----
class AskReq(BaseModel):
    question: str
    top_k: int = 8

class AskResp(BaseModel):
    answer: str
    filters: Dict[str, Any]
    where: Optional[Dict[str, Any]]
    snippets: List[Dict[str, Any]]
    stats: Dict[str, Any]

# ---- app ----
app = FastAPI(title="Disney Reviews QA")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

@app.post("/ask", response_model=AskResp)
def ask(req: AskReq):
    # 1) PARSE: NL question -> structured filters (dynamic enums from parquet)
    filters, catalog = parse_question(req.question, parquet_path=LABELED_PARQUET)
    where = build_where_from_filters(filters, catalog)  # for Chroma (equality / $in only)

    # 2) RETRIEVE: semantic search -> top-K snippets (grounding)
    emb = embed_query_openai(req.question)
    snippets = retrieve_snippets(CHROMA_PATH, COLLECTION, emb, where=where, top_k=req.top_k)

    # 3) COMPUTE STATS: full filtered dataset (not just top-K)
    df = get_parquet_df()
    fdf = apply_filters(
        df,
        branch=filters.get("branch"),
        countries=filters.get("countries"),
        months=filters.get("months"),
        seasons=filters.get("seasons"),
        years=filters.get("years"),
        rating_gte=filters.get("rating_gte"),
        sentiment=[filters["sentiment"]] if filters.get("sentiment") else None,
        cluster_id_col=catalog.get("cluster_id_col"),
        cluster_label_col=catalog.get("cluster_label_col"),
        cluster_id=filters.get("cluster_id"),
        cluster_label=filters.get("cluster_label"),
    )
    stats = compute_metrics(fdf, cluster_label_col=catalog.get("cluster_label_col"))

    # 4) SYNTHESIZE: use stats (trends) + snippets (evidence) to produce the answer
    answer = synthesize_answer(req.question, snippets, stats=stats)

    # 5) RETURN: everything the UI needs
    return AskResp(
        answer=answer,
        filters=filters,
        where=where,
        snippets=snippets,
        stats=stats
    )
