#!/usr/bin/env bash
set -euo pipefail
python -m src.eval_pipeline --dataset data/golden_dataset.jsonl --handbook data/handbook.md --reports-dir reports --report-path REPORT.md
