from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

baseline = pd.read_csv(RESULTS_DIR / "baseline_scaling_results.csv")
hnsw = pd.read_csv(RESULTS_DIR / "hnswlib_scaling_results.csv")


def plot_metric(metric: str, ylabel: str, filename: str) -> None:
    plt.figure(figsize=(8, 5))

    plt.plot(
        baseline["size"],
        baseline[metric],
        marker="o",
        label="Baseline: numpy brute-force",
    )

    plt.plot(
        hnsw["size"],
        hnsw[metric],
        marker="o",
        label="Fix: HNSW index")

    plt.xscale("log")
    plt.xlabel("Corpus size, passages")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs corpus size")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = PLOTS_DIR / filename
    plt.savefig(out_path, dpi=160)
    print(f"Saved: {out_path}")


plot_metric("recall@1", "Recall@1", "recall_at_1.png")
plot_metric("recall@10", "Recall@10", "recall_at_10.png")
plot_metric("mrr@10", "MRR@10", "mrr_at_10.png")
plot_metric("latency_p50_ms", "Latency p50, ms", "latency_p50.png")
plot_metric("latency_p95_ms", "Latency p95, ms", "latency_p95.png")
plot_metric("latency_p99_ms", "Latency p99, ms", "latency_p99.png")
plot_metric("process_ram_mb", "Process RAM, MB", "process_ram.png")