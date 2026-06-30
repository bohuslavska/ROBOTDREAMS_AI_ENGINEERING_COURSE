"""Run the full eval pipeline and write machine-readable + human-readable reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.assistant import HRHandbookAssistant
from src.evaluators import evaluate_case, summarize
from src.report import write_report


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_eval(dataset_path: Path, handbook_path: Path, reports_dir: Path, report_path: Path) -> dict[str, Any]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    assistant = HRHandbookAssistant(handbook_path)
    cases = load_jsonl(dataset_path)

    results = []
    for case in cases:
        raw_response = assistant.answer(case["user_input"])
        response = {
            "answer": raw_response.answer,
            "citations": raw_response.citations,
            "refused": raw_response.refused,
            "refusal_reason": raw_response.refusal_reason,
        }
        evaluated = evaluate_case(case, response)
        evaluated["user_input"] = case["user_input"]
        evaluated["refusal_reason"] = raw_response.refusal_reason or ""
        results.append(evaluated)

    summary = summarize(results)

    pd.DataFrame(results).to_csv(reports_dir / "eval_results.csv", index=False)
    (reports_dir / "eval_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(summary, results, report_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/golden_dataset.jsonl")
    parser.add_argument("--handbook", default="data/handbook.md")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--report-path", default="REPORT.md")
    args = parser.parse_args()

    summary = run_eval(
        dataset_path=Path(args.dataset),
        handbook_path=Path(args.handbook),
        reports_dir=Path(args.reports_dir),
        report_path=Path(args.report_path),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
