# Lesson 15 Homework — Eval Pipeline for AI Assistant

This project implements an end-to-end eval pipeline around a small **HR Handbook AI Assistant**.

The homework requirement is to systematically evaluate an AI assistant against four production-readiness problem classes:

1. **PII leakage**
2. **Prompt injection**
3. **Hallucinations / faithfulness**
4. **Refusal patterns**

The project is intentionally offline and deterministic: no OpenAI key, no Docker, no external service is required.

## Project structure

```text
lesson15_eval_pipeline/
├── data/
│   ├── handbook.md              # assistant knowledge base
│   └── golden_dataset.jsonl      # eval cases
├── reports/
│   ├── eval_results.csv          # generated after running eval
│   └── eval_summary.json         # generated after running eval
├── scripts/
│   └── run_eval.sh
├── src/
│   ├── assistant.py              # system under evaluation
│   ├── eval_pipeline.py          # runner
│   ├── evaluators.py             # metric checks
│   └── report.py                 # REPORT.md writer
├── tests/
│   └── test_eval_pipeline.py
├── Makefile
├── REPORT.md                     # generated production-readiness report
└── requirements.txt
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make eval
```

Or without Make:

```bash
python -m src.eval_pipeline \
  --dataset data/golden_dataset.jsonl \
  --handbook data/handbook.md \
  --reports-dir reports \
  --report-path REPORT.md
```

## Run tests

```bash
make test
```

## What is evaluated?

The golden dataset contains 26 cases:

- 8 faithfulness cases
- 6 PII leakage cases
- 6 prompt-injection cases
- 6 refusal-pattern cases

The report includes concrete pass rates, thresholds, sample cases, failures, limitations, and a final `SHIP` / `NOT_SHIP` verdict.

## Why this matches lesson 15

The lesson demo shows that latency/error monitoring is not enough for LLM systems. This homework applies the same idea in an offline eval setting: instead of only checking whether the assistant returns a response, it checks the **content** of the response for privacy, security, groundedness, and refusal quality.
