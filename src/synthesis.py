# synthesis.py

from typing import List, Dict, Any, Optional
import os
from openai import OpenAI

OPENAI_MODEL_SYNTH = os.getenv("OPENAI_MODEL_SYNTH", "gpt-4o-mini")

def _summarize_stats(stats: Dict[str, Any]) -> str:
    """Compact 2–4 lines for the prompt from dataset-level stats."""
    if not stats:
        return "No aggregate stats available."
    n = stats.get("n") or stats.get("n_matching_reviews")
    avg = stats.get("avg_rating")
    top_clusters = []
    for row in (stats.get("by_cluster_label") or [])[:3]:
        lbl, cnt = row.get("cluster_label"), row.get("count")
        if lbl and cnt:
            top_clusters.append(f"{lbl} ({cnt})")
    parts = []
    if n is not None: parts.append(f"Total matching reviews: {n}.")
    if avg is not None: parts.append(f"Average rating: {avg:.2f}.")
    if top_clusters: parts.append("Top clusters: " + ", ".join(top_clusters) + ".")
    return " ".join(parts) if parts else "No aggregate stats available."

def synthesize_answer(
    question: str,
    snippets: List[Dict[str, Any]],
    *,
    stats: Optional[Dict[str, Any]] = None,
    model: str = OPENAI_MODEL_SYNTH,
) -> str:
    """
    Produce the final answer using dataset-level stats for trends + snippets for evidence.
    """
    # Build compact evidence block
    lines = []
    for i, s in enumerate(snippets[:10], 1):
        meta = []
        if s.get("Branch"):         meta.append(str(s["Branch"]))
        if s.get("cluster_label"):  meta.append(str(s["cluster_label"]))
        if s.get("sentiment"):      meta.append(f"sentiment={s['sentiment']}")
        if s.get("Rating") is not None: meta.append(f"⭐{s['Rating']}")
        header = f"[{i}] " + " | ".join(meta)
        lines.append(header + "\n" + s.get("snippet",""))
    evidence = "\n\n".join(lines) if lines else "No snippets."

    stats_txt = _summarize_stats(stats) if stats else "No aggregate stats available."

    sys = (
        "You are a careful analyst. Use dataset-wide stats for overall trends and the snippets as grounded evidence. "
        "Be concise and actionable. If evidence conflicts, note the nuance. Do not invent facts."
    )
    usr = (
        f"Question: {question}\n\n"
        f"Dataset stats:\n{stats_txt}\n\n"
        f"Snippets:\n{evidence}\n\n"
        "Write a short answer (4–7 sentences), then 2–3 concrete action items."
    )

    if not os.getenv("OPENAI_API_KEY"):
        # Fallback if no key — basic concatenation so the app still renders something.
        return f"{stats_txt}\n\nExamples:\n" + "\n\n".join(lines[:3])

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
        temperature=0.2,
        max_tokens=450,
    )
    return resp.choices[0].message.content.strip()
