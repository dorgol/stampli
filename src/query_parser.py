#!/usr/bin/env python3
"""
query_parser.py
Parse a natural-language question into structured filters using OpenAI JSON Schema (structured outputs).

Usage:
  $env:OPENAI_API_KEY="sk-..."
  python src/query_parser.py --question "Is the staff in Paris friendly?"
"""

import argparse
import json
import os
from typing import Any, Dict

from openai import OpenAI

# Draft-07 JSON Schema the model must adhere to
QUERY_SCHEMA: Dict[str, Any] = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "DisneyReviewQuery",
  "type": "object",
  "properties": {
    "branch": {
      "description": "Target parks. Always an array or null.",
      "type": ["array","null"],
      "items": {"type":"string", "enum":[
        "Disneyland_California","Disneyland_Paris","Disneyland_HongKong"
      ]}
    },
    "country": {
      "description": "Reviewer locations. Always an array or null.",
      "type": ["array","null"],
      "items": {"type":"string", "minLength": 1}
    },
    "season": {
      "type": ["string","null"],
      "enum": ["Spring","Summer","Fall","Winter", None]
    },
    "month":  {"type": ["integer","null"], "minimum": 1, "maximum": 12},
    "year":   {"type": ["integer","null"], "minimum": 1990, "maximum": 2100},
    "rating_gte": {"type": ["number","null"], "minimum": 1, "maximum": 5},
    "focus": {"type": ["string","null"], "minLength": 1}
  },
  "required": ["branch","country","season","month","year","rating_gte","focus"],
  "additionalProperties": False
}

SYSTEM_MSG = (
  "You are a strict JSON generator. Return ONLY a JSON object that matches the provided schema. "
  "Rules: 'branch' and 'country' MUST be arrays (or null). If the question implies a single park or country, use a single-element array. "
  "Infer 'focus' as a short theme (e.g., 'crowds', 'staff friendliness', 'food', 'price', 'cleanliness'). "
  "Do not include any extra fields or text."
)

def parse_question(question: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set.")
    client = OpenAI()

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": f"Question: {question}\nReturn JSON only."},
        ],
        temperature=0.0,
        max_tokens=400,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "DisneyReviewQuery",
                "strict": True,
                "schema": QUERY_SCHEMA
            }
        }
    )
    data = json.loads(resp.choices[0].message.content)

    # Safety normalization (should already be correct due to schema)
    for k in ("branch","country"):
        v = data.get(k)
        if isinstance(v, str):
            data[k] = [v]
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    parsed = parse_question(args.question, model=args.model)
    print(json.dumps(parsed, indent=2))

if __name__ == "__main__":
    main()
