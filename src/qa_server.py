#!/usr/bin/env python3
"""
qa_server.py
FastAPI server exposing a Q&A endpoint over Disney reviews.

Run:
    uvicorn src.qa_server:app --reload --port 8000
"""

from __future__ import annotations

import os
from typing import Dict, Any

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.analysis_tool import add_derived_cols, apply_filters, compute_metrics
from src.query_parser import parse_question
from src.retrieval import embed_query_openai, build_where, retrieve_snippets
from src.synthesis import run as synthesize

# ---------- Config ----------
META_PATH = os.getenv("META_PATH", "data/embeddings_out/reviews_with_embeddings.parquet")
CHROMA_PATH = os.getenv("CHROMA_PATH", "data/index/chroma_db")
COLLECTION = os.getenv("COLLECTION", "reviews")

# Load metadata once
_df_meta: pd.DataFrame | None = None


def get_meta_df() -> pd.DataFrame:
    global _df_meta
    if _df_meta is None:
        df = pd.read_parquet(META_PATH)
        df = add_derived_cols(df)
        _df_meta = df
    return _df_meta


# ---------- API ----------
app = FastAPI(title="Disney Reviews QA", version="0.1.0")


class QARequest(BaseModel):
    question: str
    top_k: int = 12


class QAResponse(BaseModel):
    answer: str
    filters: Dict[str, Any]
    stats: Dict[str, Any]
    citations: list[int]
    snippets_used: list[Dict[str, Any]]


@app.post("/qa", response_model=QAResponse)
def answer_question(req: QARequest):
    """
    Main endpoint: ask a natural language question and get a grounded answer.
    """
    # 1. Parse NL question into structured filters
    parsed = parse_question(req.question)

    branch   = parsed.get("branch")
    country  = parsed.get("country")
    month    = parsed.get("month")
    season   = parsed.get("season")
    year     = parsed.get("year")
    rating_gte = parsed.get("rating_gte")

    # 2. Analysis: compute stats over filtered reviews
    df = get_meta_df()
    filtered = apply_filters(df, branch, country, month, season, year, rating_gte)
    stats = compute_metrics(filtered)

    # 3. Retrieval: get evidence snippets from Chroma
    q_emb = embed_query_openai(req.question)
    where = build_where(branch, country, month, season, year, rating_gte)
    snippets = retrieve_snippets(
        chroma_path=CHROMA_PATH,
        collection=COLLECTION,
        query_embedding=q_emb,
        where=where,
        top_k=req.top_k,
    )

    # 4. Synthesis: generate grounded answer
    synth = synthesize(req.question, stats, snippets)

    # 5. Build payload
    return {
        "answer": synth["answer"],
        "filters": parsed,
        "stats": stats,
        "citations": synth["citations"],
        "snippets_used": snippets[:8],
    }


@app.get("/health")
def health():
    return {"status": "ok"}
