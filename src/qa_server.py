# src/qa_server.py
import hashlib
import json
import os
import time
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.query_parser import parse_question
from src.synthesis import run as synth_run

load_dotenv()

META_PATH   = os.getenv("META_PATH",   "data/embeddings_out/reviews_with_embeddings.parquet")
CHROMA_PATH = os.getenv("CHROMA_PATH", "data/index/chroma_db")
COLLECTION  = os.getenv("COLLECTION",  "reviews")

if not os.getenv("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY missing. Set in .env or environment.")

app = FastAPI(title="Disney Reviews Q&A", version="1.0")

# CORS for local Streamlit dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str
    top_k: int = 12
    # optional explicit overrides
    branch: Optional[List[str]] = None
    country: Optional[List[str]] = None
    season: Optional[List[str]] = None
    month: Optional[List[int]] = None
    year: Optional[List[int]] = None
    rating_gte: Optional[float] = None

def _tolist(x):
    if x is None: return None
    return x if isinstance(x, list) else [x]

def _cache_key(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

# simple in-memory cache
_CACHE: Dict[str, Dict[str, Any]] = {}

@app.get("/health")
def health():
    ok = os.path.exists(META_PATH)
    return {"status": "ok" if ok else "meta_not_found", "meta_path": META_PATH}

@app.post("/ask")
def ask(req: AskRequest) -> Dict[str, Any]:
    t0 = time.time()
    try:
        # 1) parse question → structured filters
        parsed = parse_question(req.question)
        # allow client overrides to take precedence
        branch = req.branch if req.branch is not None else _tolist(parsed.get("branch"))
        country = req.country if req.country is not None else _tolist(parsed.get("country"))
        season  = req.season if req.season is not None else (_tolist(parsed.get("season")) if parsed.get("season") else None)
        month   = req.month  if req.month  is not None else (_tolist(parsed.get("month"))  if parsed.get("month")  else None)
        year    = req.year   if req.year   is not None else (_tolist(parsed.get("year"))   if parsed.get("year")   else None)
        rating  = req.rating_gte if req.rating_gte is not None else parsed.get("rating_gte")

        cache_payload = {
            "q": req.question, "branch": branch, "country": country,
            "season": season, "month": month, "year": year,
            "rating_gte": rating, "top_k": req.top_k,
            "meta": META_PATH, "chroma": CHROMA_PATH, "collection": COLLECTION
        }
        key = _cache_key(cache_payload)
        if key in _CACHE:
            out = _CACHE[key]
            out["_meta"] = {"cached": True, "latency_ms": int((time.time()-t0)*1000)}
            return out

        # 2) first attempt: exact filters
        out = synth_run(
            question=req.question,
            meta_path=META_PATH,
            chroma_path=CHROMA_PATH,
            collection=COLLECTION,
            branch=branch, country=country,
            month=month, season=season, year=year,
            rating_gte=rating,
            top_k=req.top_k
        )

        relaxed = None
        # 3) coverage backoff if needed
        if out["stats"]["n"] == 0:
            # drop time filters
            out = synth_run(
                question=req.question,
                meta_path=META_PATH,
                chroma_path=CHROMA_PATH,
                collection=COLLECTION,
                branch=branch, country=country,
                month=None, season=None, year=None,
                rating_gte=rating, top_k=req.top_k
            )
            if out["stats"]["n"] > 0:
                relaxed = "time"

        if out["stats"]["n"] == 0 and branch:
            # drop branch last (broaden)
            out = synth_run(
                question=req.question,
                meta_path=META_PATH,
                chroma_path=CHROMA_PATH,
                collection=COLLECTION,
                branch=None, country=country,
                month=None, season=None, year=None,
                rating_gte=rating, top_k=req.top_k
            )
            if out["stats"]["n"] > 0:
                relaxed = "branch" if relaxed is None else f"{relaxed}+branch"

        if relaxed:
            out["answer"] = f"Note: no results with the exact filters; relaxed {relaxed} constraint(s).\n\n" + out["answer"]

        out["parsed"] = parsed
        out["_meta"] = {"cached": False, "latency_ms": int((time.time()-t0)*1000)}
        _CACHE[key] = out
        return out

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
