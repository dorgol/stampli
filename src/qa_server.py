#!/usr/bin/env python3
# src/qa_server.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import os, json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

from src.query_parser import parse_question, build_where_from_filters, DEFAULT_PARQUET
from src.retrieval import embed_query_openai, retrieve_snippets

CHROMA_PATH = os.getenv("CHROMA_PATH", "index/chroma_db")
COLLECTION  = os.getenv("CHROMA_COLLECTION", "reviews")
LABELED_PARQUET = os.getenv("LABELED_PARQUET", DEFAULT_PARQUET)
OPENAI_MODEL_SYNTH = os.getenv("OPENAI_MODEL_SYNTH", "gpt-4o-mini")

app = FastAPI(title="Disney Reviews QA")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

class AskReq(BaseModel):
    question: str
    top_k: int = 8

class AskResp(BaseModel):
    answer: str
    filters: Dict[str, Any]
    where: Optional[Dict[str, Any]]
    snippets: List[Dict[str, Any]]

def synthesize_answer(question: str, snippets: List[Dict[str, Any]], model: str = OPENAI_MODEL_SYNTH) -> str:
    """LLM summary grounded in retrieved snippets (shows cluster labels when present)."""
    if not os.getenv("OPENAI_API_KEY"):
        # Fallback: simple extractive answer
        bullets = []
        for s in snippets[:5]:
            tag = s.get("cluster_label") or s.get("sentiment") or ""
            bullets.append(f"- ({tag}) {s['snippet']}")
        return "Here’s what I found:\n" + "\n".join(bullets)

    # Build compact evidence block
    lines = []
    for i, s in enumerate(snippets[:10], 1):
        meta = []
        if s.get("Branch"):         meta.append(str(s["Branch"]))
        if s.get("cluster_label"):  meta.append(str(s["cluster_label"]))
        if s.get("sentiment"):      meta.append(f"sentiment={s['sentiment']}")
        if s.get("Rating") is not None: meta.append(f"⭐{s['Rating']}")
        header = f"[{i}] " + " | ".join(meta)
        lines.append(header + "\n" + s["snippet"])
    evidence = "\n\n".join(lines)

    sys = (
        "Answer the user's question using ONLY the provided snippets. "
        "Be concise and actionable. If evidence conflicts, note the nuance. "
        "If asked about seasons/months/branches, be explicit. "
        "Do not hallucinate facts not present in the snippets."
    )
    usr = f"Question: {question}\n\nSnippets:\n{evidence}\n\nWrite a short answer (4–7 sentences), then 2–3 action items."

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":sys},{"role":"user","content":usr}],
        temperature=0.2, max_tokens=400
    )
    return resp.choices[0].message.content.strip()

@app.post("/ask", response_model=AskResp)
def ask(req: AskReq):
    filters, catalog = parse_question(req.question, parquet_path=LABELED_PARQUET)
    where = build_where_from_filters(filters, catalog)
    emb = embed_query_openai(req.question)
    snippets = retrieve_snippets(CHROMA_PATH, COLLECTION, emb, where=where, top_k=req.top_k)
    answer = synthesize_answer(req.question, snippets)
    return AskResp(answer=answer, filters=filters, where=where, snippets=snippets)
