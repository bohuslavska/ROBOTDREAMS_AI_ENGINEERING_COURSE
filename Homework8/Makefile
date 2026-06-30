.PHONY: eval test clean

eval:
	python -m src.eval_pipeline --dataset data/golden_dataset.jsonl --handbook data/handbook.md --reports-dir reports --report-path REPORT.md

test:
	PYTHONPATH=. pytest -q

clean:
	rm -f reports/eval_results.csv reports/eval_summary.json REPORT.md
