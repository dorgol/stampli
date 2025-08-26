#!/usr/bin/env python3
"""
label_clusters_llm.py
Use an LLM (gpt-4o-mini by default) to label clusters with business-friendly tags.
"""
import argparse, json, os
from pathlib import Path
import pandas as pd

def make_prompt(examples):
    sys = (
      "You are labeling Disneyland park review clusters.\n"
      "Return a single JSON object with fields:\n"
      "label (2-4 words), keywords (5-8 terms, comma-separated), "
      "sentiment (very negative/negative/mixed/positive/very positive), "
      "summary (2-3 sentences), actions (2-4 bullet strings), confidence (0-100)."
    )
    joined = "\n---\n".join(examples)
    usr = (
      "Read the cluster's representative customer review snippets below and produce the JSON.\n"
      "Focus on operationally actionable insights (queues, staff, food, cleanliness, price, weather, rides, shows, navigation, etc.).\n"
      "Snippets:\n" + joined
    )
    return sys, usr

def call_openai(messages, model="gpt-4o-mini", temperature=0.2):
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":messages[0]},
                  {"role":"user","content":messages[1]}],
        temperature=temperature,
        response_format={"type":"json_object"}
    )
    return resp.choices[0].message.content

def truncate(text, max_chars=300):
    t = str(text).replace("\n"," ").strip()
    return (t[:max_chars] + "…") if len(t) > max_chars else t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps_parquet", type=str, default="labeling_in/cluster_reps.parquet")
    ap.add_argument("--out_dir", type=str, default="labeling_out")
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--per_cluster", type=int, default=12)
    ap.add_argument("--min_chars", type=int, default=40)
    args = ap.parse_args()

    outp = Path(args.out_dir); outp.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.reps_parquet)
    rows = []
    jsonl_path = outp / "cluster_labels.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for cid in sorted(df["cluster"].unique()):
            if cid == -1:
                continue
            sub = df[df["cluster"] == cid].sort_values("rep_rank").head(args.per_cluster).copy()
            examples = [truncate(t, 300) for t in sub["Review_Text"].tolist() if len(str(t)) >= args.min_chars]
            if len(examples) < 3:
                payload = {
                    "label": "Sparse / Misc",
                    "keywords": "miscellaneous",
                    "sentiment": "mixed",
                    "summary": "Insufficient text to determine a clear theme.",
                    "actions": ["Collect more data"],
                    "confidence": 20
                }
            else:
                sys, usr = make_prompt(examples)
                content = call_openai((sys, usr), model=args.model)
                try:
                    payload = json.loads(content)
                except Exception:
                    payload = json.loads(content.strip("` \n"))

            rows.append({
                "cluster": int(cid),
                "label": payload.get("label","").strip(),
                "keywords": payload.get("keywords","").strip(),
                "sentiment": payload.get("sentiment","").strip(),
                "summary": payload.get("summary","").strip(),
                "actions": "| ".join(payload.get("actions", [])) if isinstance(payload.get("actions"), list) else str(payload.get("actions","")).strip(),
                "confidence": payload.get("confidence", 0)
            })
            jf.write(json.dumps({"cluster": int(cid), **payload}, ensure_ascii=False) + "\n")

    out_csv = outp / "cluster_labels.csv"
    pd.DataFrame(rows).sort_values("cluster").to_csv(out_csv, index=False)
    print(f"Saved labels → {out_csv}")
    print(f"Saved JSONL → {jsonl_path}")

if __name__ == "__main__":
    main()
