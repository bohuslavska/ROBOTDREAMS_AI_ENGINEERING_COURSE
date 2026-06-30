"""
Evaluate a model (base or fine-tuned) against eval.jsonl.

Metrics:
- json_valid_rate
- exact_match_rate (all 5 fields match)
- field_accuracy per field
- avg_input_tokens, avg_output_tokens (for cost analysis)

Usage:
    python3 evaluate.py --model gpt-4o-mini --output results/baseline_metrics.json
    python3 evaluate.py --model ft:gpt-4o-mini:... --output results/finetuned_metrics.json
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

from openai import OpenAI

SYSTEM_PROMPT = (
    "You extract structured data from customer support emails. "
    "Return only a single valid JSON object with fields: "
    "customer_name (string or null), product (string), "
    "issue_category (one of: billing, technical, account, feature_request, other), "
    "urgency (one of: low, medium, high, critical), "
    "summary (one short sentence). No extra text."
)

FIELDS = ["customer_name", "product", "issue_category", "urgency", "summary"]


def safe_json_parse(s: str):
    s = s.strip()
    # try to strip markdown fences if model wraps
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        s = s.rsplit("```", 1)[0].strip()
        if s.startswith("json"):
            s = s[4:].strip()
    try:
        return json.loads(s), True
    except json.JSONDecodeError:
        return None, False


def field_match(predicted, expected, field):
    if field == "summary":
        # summary: relaxed string similarity — count match if both non-empty AND share >= 2 meaningful words
        if not isinstance(predicted, str) or not isinstance(expected, str):
            return False
        p_words = {w.lower().strip(".,!?") for w in predicted.split() if len(w) > 3}
        e_words = {w.lower().strip(".,!?") for w in expected.split() if len(w) > 3}
        if not p_words or not e_words:
            return False
        overlap = len(p_words & e_words) / max(len(e_words), 1)
        return overlap >= 0.4
    if field == "customer_name":
        if predicted is None and expected is None:
            return True
        if predicted is None or expected is None:
            return False
        # match if predicted contains the first name from expected (or vice versa)
        return expected.split()[0].lower() in predicted.lower() if predicted else False
    # exact match for product/category/urgency (case-insensitive)
    if predicted is None:
        return False
    return str(predicted).strip().lower() == str(expected).strip().lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--eval-file", default="data/eval.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent.parent
    eval_path = here / args.eval_file
    out_path = here / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    examples = []
    with open(eval_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))

    print(f"Loaded {len(examples)} eval examples")
    print(f"Model: {args.model}")

    client = OpenAI()

    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    json_valid = 0
    exact_match = 0
    field_correct = defaultdict(int)

    for i, ex in enumerate(examples, 1):
        email = ex["email"]
        expected = ex["expected"]

        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": email},
                ],
                temperature=0,
                max_tokens=300,
            )
        except Exception as e:
            print(f"  [{i}/{len(examples)}] ERROR: {e}")
            results.append({"email": email, "expected": expected, "predicted": None, "error": str(e)})
            continue

        raw = resp.choices[0].message.content or ""
        total_input_tokens += resp.usage.prompt_tokens
        total_output_tokens += resp.usage.completion_tokens

        predicted, valid = safe_json_parse(raw)
        if valid:
            json_valid += 1

        # field-level accuracy
        per_field = {}
        all_match = True
        for f in FIELDS:
            ok = valid and field_match(predicted.get(f) if predicted else None, expected[f], f)
            per_field[f] = ok
            if ok:
                field_correct[f] += 1
            if not ok:
                all_match = False
        if valid and all_match:
            exact_match += 1

        results.append({
            "i": i,
            "email": email,
            "expected": expected,
            "raw_output": raw,
            "predicted": predicted,
            "valid_json": valid,
            "exact_match": valid and all_match,
            "field_match": per_field,
        })

        marker = "✓" if (valid and all_match) else "✗"
        print(f"  [{i}/{len(examples)}] {marker} valid_json={valid} exact={valid and all_match}")

    n = len(examples)
    metrics = {
        "model": args.model,
        "n_examples": n,
        "json_valid_rate": round(json_valid / n, 4),
        "exact_match_rate": round(exact_match / n, 4),
        "field_accuracy": {f: round(field_correct[f] / n, 4) for f in FIELDS},
        "avg_input_tokens": round(total_input_tokens / n, 1),
        "avg_output_tokens": round(total_output_tokens / n, 1),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "details": results,
    }

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Console summary
    print()
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Examples: {n}")
    print(f"JSON valid:       {metrics['json_valid_rate']*100:.1f}%")
    print(f"Exact match:      {metrics['exact_match_rate']*100:.1f}%")
    print("Field accuracy:")
    for f in FIELDS:
        print(f"  {f:20s} {metrics['field_accuracy'][f]*100:.1f}%")
    print(f"Avg input tokens:  {metrics['avg_input_tokens']}")
    print(f"Avg output tokens: {metrics['avg_output_tokens']}")
    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
