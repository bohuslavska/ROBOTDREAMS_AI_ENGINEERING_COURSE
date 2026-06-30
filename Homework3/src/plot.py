"""
Генерує 3 графіки з results.csv:
    1. pareto_frontier.png      — recall@10 vs latency_p95 (з лінією Pareto)
    2. latency_distribution.png — p50/p95/p99 порівняння (log scale)
    3. disk_size_chart.png      — розмір індексу для кожної БД

Запуск:
    python src/plot.py --input results/results.csv --output results/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Одна палітра для всіх графіків — кожна БД завжди має той самий колір.
DB_COLORS = {
    "faiss_flat":  "#1f77b4",   # blue
    "faiss_hnsw":  "#ff7f0e",   # orange
    "chroma":      "#2ca02c",   # green
    "qdrant":      "#d62728",   # red
    "pgvector":    "#9467bd",   # purple
}


def _load_results(path: Path) -> pd.DataFrame:
    """Читає CSV і відфільтровує невдалі прогони."""
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    if df.empty:
        raise RuntimeError(f"У {path} нема валідних рядків (status=ok).")
    return df


def _pareto_frontier(points: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
    """
    Points: [(latency, recall, db_name), ...].
    Frontier — точки, які НЕ домінуються (нема іншої точки з меншою latency І більшою recall).

    Алгоритм:
      1. Сортуємо за latency зростаючою.
      2. Йдемо зліва направо, тримаємо max-побачену recall.
      3. Якщо нова точка має recall > max_seen, вона — на frontier.
    """
    sorted_pts = sorted(points, key=lambda p: p[0])
    frontier: List[Tuple[float, float, str]] = []
    max_recall = -np.inf
    for latency, recall, name in sorted_pts:
        if recall > max_recall:
            frontier.append((latency, recall, name))
            max_recall = recall
    return frontier


def plot_pareto_frontier(df: pd.DataFrame, output: Path) -> None:
    """Scatter: x=latency_p95, y=recall@10. Лінія через Pareto-точки."""
    fig, ax = plt.subplots(figsize=(10, 7))

    points = list(zip(df["latency_p95_ms"], df["recall_at_10"], df["db"]))

    # Малюємо ВСІ точки (включно з домінованими)
    for latency, recall, name in points:
        color = DB_COLORS.get(name, "gray")
        ax.scatter(latency, recall, s=200, c=color, alpha=0.85, edgecolors="black",
                   linewidths=1.5, zorder=3)
        ax.annotate(
            name,
            xy=(latency, recall),
            xytext=(10, 10), textcoords="offset points",
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.85),
        )

    # Лінія Pareto frontier
    frontier = _pareto_frontier(points)
    if len(frontier) >= 2:
        xs = [p[0] for p in frontier]
        ys = [p[1] for p in frontier]
        ax.plot(xs, ys, "--", c="gray", alpha=0.6, linewidth=2, zorder=2,
                label=f"Pareto frontier ({len(frontier)} БД)")

    # "Ідеальний кут" — top-left
    ax.annotate(
        "Ideal (fast + accurate)",
        xy=(ax.get_xlim()[0] + 0.5, 1.0),
        fontsize=9, style="italic", alpha=0.6,
        ha="left",
    )

    ax.set_xlabel("Latency p95 (ms) — менше краще", fontsize=12)
    ax.set_ylabel("Recall@10 — більше краще", fontsize=12)
    ax.set_title("Pareto Frontier: Recall vs Latency\n(BeIR/quora, 523K документів, BGE-small)",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")  # log scale бо різниця 0.5ms vs 50ms
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {output}")


def plot_latency_distribution(df: pd.DataFrame, output: Path) -> None:
    """Grouped bars: p50, p95, p99 для кожної БД. Log scale на Y."""
    fig, ax = plt.subplots(figsize=(11, 6))

    dbs = df["db"].tolist()
    p50 = df["latency_p50_ms"].values
    p95 = df["latency_p95_ms"].values
    p99 = df["latency_p99_ms"].values

    x = np.arange(len(dbs))
    width = 0.27

    ax.bar(x - width, p50, width, label="p50", color="#4c9aff", edgecolor="black")
    ax.bar(x,         p95, width, label="p95", color="#ffa940", edgecolor="black")
    ax.bar(x + width, p99, width, label="p99", color="#ff4d4f", edgecolor="black")

    # Анотації над барами
    for xi, (v50, v95, v99) in enumerate(zip(p50, p95, p99)):
        for offset, val in zip([-width, 0, width], [v50, v95, v99]):
            ax.text(xi + offset, val * 1.05, f"{val:.2f}",
                    ha="center", fontsize=8, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(dbs, rotation=15, ha="right")
    ax.set_ylabel("Latency (ms) — log scale", fontsize=12)
    ax.set_title("Query Latency Distribution: p50 / p95 / p99",
                 fontsize=14, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {output}")


def plot_disk_size(df: pd.DataFrame, output: Path) -> None:
    """Bar chart розміру індексу."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Сортуємо за disk_mb зростаючою — щоб одразу видно "найдешевших"
    sorted_df = df.sort_values("disk_mb").reset_index(drop=True)

    dbs = sorted_df["db"].tolist()
    sizes = sorted_df["disk_mb"].values
    colors = [DB_COLORS.get(name, "gray") for name in dbs]

    bars = ax.bar(dbs, sizes, color=colors, edgecolor="black", linewidth=1.2)

    # Цифри над барами
    for bar, size in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{size:.1f} MB", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_ylabel("Disk size (MB)", fontsize=12)
    ax.set_title("Index Disk Size by Database\n(523K документів × 384 dim, float32)",
                 fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots from benchmark results.")
    parser.add_argument("--input", type=Path, default=Path("results/results.csv"),
                        help="Шлях до results.csv")
    parser.add_argument("--output", type=Path, default=Path("results/"),
                        help="Директорія для збереження PNG")
    args = parser.parse_args()

    df = _load_results(args.input)
    print(f"[load] {args.input}: {len(df)} БД (status=ok)")

    args.output.mkdir(parents=True, exist_ok=True)

    plot_pareto_frontier(df, args.output / "pareto_frontier.png")
    plot_latency_distribution(df, args.output / "latency_distribution.png")
    plot_disk_size(df, args.output / "disk_size_chart.png")

    print(f"\n[done] 3 графіки збережено у {args.output}")


if __name__ == "__main__":
    main()
