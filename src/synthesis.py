#!/usr/bin/env python3
"""
synthesis.py
Given a question, precomputed stats, and retrieved snippets,
ask GPT-4o-mini to synthesize a grounded answer.

This module is library-only (no CLI).
"""

import json
from typing import Any, Dict, List

from openai import OpenAI


def synthesize_answer(
    question: str,
    stats: Dict[str, Any],
    snippets: List[Dict[str, Any]],
    model: str = "gpt-4o-mini"
) -> str:
    """
    Ask the LLM to write a concise, grounded answer using provided stats + snippets.
    """
    client = OpenAI()

    # Keep only compact preview snippets to avoid blowing up the context
    preview = []
    for s in snippets[:8]:
        preview.append({
            "id": s.get("Review_ID") or s.get("id"),
            "branch": s.get("Branch"),
            "country": s.get("Reviewer_Location"),
            "year": s.get("Year"), "month": s.get("Month"),
            "rating": s.get("Rating"),
            "quote": (s.get("snippet") or "")[:300]
        })

    system_msg = (
        "You are a precise analyst. Answer ONLY using the provided statistics and review snippets. "
        "Start with a clear 1–2 sentence verdict. Then add 2–4 bullet points with key numbers "
        "(round percentages; show n where helpful). Cite 2–3 review IDs inline like [ID: 670772142]. "
        "If evidence is weak (n<20), state uncertainty."
    )

    payload = {"question": question, "stats": stats, "snippets": preview}

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": json.dumps(payload)},
        ],
        temperature=0.2,
        max_tokens=450,
    )

    return resp.choices[0].message.content.strip()


def run(
    question: str,
    stats: Dict[str, Any],
    snippets: List[Dict[str, Any]],
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    Public entry: return dict with answer + citations.
    """
    answer_text = synthesize_answer(question, stats, snippets, model=model)
    citations = [s.get("Review_ID") for s in snippets[:3] if s.get("Review_ID") is not None]

    return {
        "answer": answer_text,
        "citations": citations,
    }
