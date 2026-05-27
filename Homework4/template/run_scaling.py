"""
Scaling loop for dense retrieval baseline.

This script runs the same retrieval pipeline on multiple corpus sizes:

    1K -> 10K -> 100K -> 300K

For each size it measures:
- Recall@1
- Recall@5
- Recall@10
- MRR@10
- latency p50 / p95 / p99
- RAM used by embeddings
- total process RAM
- embedding throughput

This is the baseline experiment.

Later you can compare this baseline with a fix:
- Hybrid retrieval: BM25 + dense + RRF
- Reranker: dense top-100 -> reranker -> top-10
- ANN index: FAISS / HNSW / Qdrant instead of brute-force numpy
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm

from data_loader import (
    build_corpus_pool,
    build_subset,
    load_cache,
    load_qrels_and_queries,
    pick_eval_queries,
    save_cache,
)
from embedder import BGEEmbedder
from metrics import evaluate
from retriever import DenseNumpyRetriever


# =============================================================================
# Experiment configuration
# =============================================================================

SEED = 42

N_EVAL_QUERIES = 100

# Must be at least as large as your largest subset size.
# The actual pool will also include all relevant docs.
DISTRACTOR_TARGET = 300_000

SUBSET_SIZES = [1_000, 10_000, 100_000, 300_000]

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64

# We evaluate recall@1, recall@5, recall@10 and mrr@10.
# Your metrics.py computes mrr@max(ks), so max must be 10.
KS = (1, 5, 10)
TOP_K = max(KS)

BASE_DIR = Path(__file__).parent
CACHE_PATH = BASE_DIR / "cache" / "corpus.json"
EMBEDDINGS_CACHE_DIR = BASE_DIR / "cache" / "embeddings"
RESULTS_DIR = BASE_DIR / "results"

EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# If True, corpus embeddings will be rebuilt even if .npy files already exist.
# For the final report, you may want to set it to True once to re-measure
# embedding throughput from scratch.
FORCE_REBUILD_EMBEDDINGS = False


# =============================================================================
# Utility functions
# =============================================================================

def get_process_ram_mb() -> float:
    """
    Return current Python process RAM usage in MB.

    This is useful because the embedding matrix is not the only memory consumer:
    Python objects, model weights, tokenizer, cached tensors, and corpus text
    also use RAM.
    """
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def ensure_corpus_cache() -> tuple[list[dict], list[dict]]:
    """
    Load cached corpus pool and eval_set if available.

    If cache does not exist:
    1. Load qrels and queries.
    2. Pick N eval queries with ground-truth relevant docs.
    3. Stream MS MARCO corpus.
    4. Keep all relevant docs + distractors.
    5. Save everything to cache/corpus.json.

    This step can take several minutes on the first run.
    """
    if CACHE_PATH.exists():
        print(f"Loading corpus cache from: {CACHE_PATH}")
        return load_cache(CACHE_PATH)

    print("Corpus cache not found. Building it now...")
    print("Loading qrels + queries...")

    qrels, queries = load_qrels_and_queries()

    print(f"Picking {N_EVAL_QUERIES} eval queries...")
    eval_set, relevant_ids = pick_eval_queries(
        qrels=qrels,
        queries=queries,
        n=N_EVAL_QUERIES,
    )

    print(
        f"Picked {len(eval_set)} eval queries "
        f"with {len(relevant_ids)} unique relevant docs."
    )

    print(f"Streaming corpus and collecting {DISTRACTOR_TARGET:,} distractors...")
    pool = build_corpus_pool(
        relevant_ids=relevant_ids,
        n_distractors=DISTRACTOR_TARGET,
    )

    print(f"Corpus pool size: {len(pool):,} docs")

    print(f"Saving corpus cache to: {CACHE_PATH}")
    save_cache(pool=pool, eval_set=eval_set, path=CACHE_PATH)

    return pool, eval_set


def embeddings_cache_paths(size: int) -> tuple[Path, Path]:
    """
    Return paths for cached embeddings and cached document ids.

    We save document ids next to embeddings to verify that the cached matrix
    corresponds to the current subset order.
    """
    safe_model_name = MODEL_NAME.replace("/", "__")

    embeddings_path = (
        EMBEDDINGS_CACHE_DIR
        / f"{safe_model_name}__size_{size}.npy"
    )

    ids_path = (
        EMBEDDINGS_CACHE_DIR
        / f"{safe_model_name}__size_{size}_ids.json"
    )

    return embeddings_path, ids_path


def encode_corpus_with_cache(
    size: int,
    subset: list[dict],
    embedder: BGEEmbedder,
) -> tuple[np.ndarray, float | None, float | None, bool]:
    """
    Encode corpus subset into embeddings.

    If cached embeddings exist and doc ids match, load them from disk.
    Otherwise, encode texts and save embeddings to disk.

    Returns
    -------
    tuple
        doc_embeddings:
            Matrix of passage embeddings.

        embedding_seconds:
            Time spent on encoding.
            None if embeddings were loaded from cache.

        embedding_throughput:
            passages/sec.
            None if embeddings were loaded from cache.

        loaded_from_cache:
            True if embeddings were loaded from disk.
    """
    embeddings_path, ids_path = embeddings_cache_paths(size)
    subset_ids = [str(doc["id"]) for doc in subset]

    if (
        not FORCE_REBUILD_EMBEDDINGS
        and embeddings_path.exists()
        and ids_path.exists()
    ):
        cached_ids = json.load(open(ids_path, encoding="utf-8"))

        if cached_ids == subset_ids:
            print(f"Loading cached embeddings from: {embeddings_path}")
            doc_embeddings = np.load(embeddings_path)
            return doc_embeddings, None, None, True

        print(
            "Cached embeddings found, but document ids do not match. "
            "Rebuilding embeddings."
        )

    print("Encoding corpus passages...")
    doc_texts = [doc["text"] for doc in subset]

    start = time.perf_counter()

    doc_embeddings = embedder.encode(
        texts=doc_texts,
        kind="passage",
        show_progress=True,
    )

    embedding_seconds = time.perf_counter() - start
    embedding_throughput = len(doc_texts) / embedding_seconds

    print(f"Saving embeddings to: {embeddings_path}")
    np.save(embeddings_path, doc_embeddings)

    print(f"Saving embedding doc ids to: {ids_path}")
    json.dump(subset_ids, open(ids_path, "w", encoding="utf-8"))

    return doc_embeddings, embedding_seconds, embedding_throughput, False


def run_one_size(
    size: int,
    pool: list[dict],
    eval_set: list[dict],
    embedder: BGEEmbedder,
) -> dict:
    """
    Run the full dense retrieval baseline for one corpus size.

    Steps:
    1. Build reproducible subset of `size` documents.
    2. Encode passages.
    3. Build dense numpy retriever.
    4. Encode eval queries.
    5. Search top-k documents for each query.
    6. Evaluate using metrics.py.
    7. Measure latency and RAM.
    """
    print("\n" + "=" * 90)
    print(f"Running dense baseline for corpus size: {size:,}")
    print("=" * 90)

    # -------------------------------------------------------------------------
    # 1. Build reproducible subset
    # -------------------------------------------------------------------------
    subset = build_subset(
        pool=pool,
        eval_set=eval_set,
        size=size,
        seed=SEED,
    )

    print(f"Subset size: {len(subset):,} docs")

    # -------------------------------------------------------------------------
    # 2. Encode corpus passages
    # -------------------------------------------------------------------------
    doc_embeddings, embedding_seconds, embedding_throughput, loaded_from_cache = (
        encode_corpus_with_cache(
            size=size,
            subset=subset,
            embedder=embedder,
        )
    )

    if loaded_from_cache:
        print("Embeddings loaded from cache; embedding throughput not re-measured.")
    else:
        print(f"Embedding time: {embedding_seconds:.2f} sec")
        print(f"Embedding throughput: {embedding_throughput:.2f} passages/sec")

    # -------------------------------------------------------------------------
    # 3. Build retriever
    # -------------------------------------------------------------------------
    print("Building DenseNumpyRetriever...")

    retriever = DenseNumpyRetriever()
    retriever.build(
        corpus=subset,
        doc_embeddings=doc_embeddings,
    )

    # -------------------------------------------------------------------------
    # 4. Encode evaluation queries
    # -------------------------------------------------------------------------
    print("Encoding evaluation queries...")

    query_texts = [item["query"] for item in eval_set]

    query_embeddings = embedder.encode(
        texts=query_texts,
        kind="query",
        show_progress=False,
    )

    # -------------------------------------------------------------------------
    # 5. Search
    # -------------------------------------------------------------------------
    print(f"Searching top-{TOP_K} documents for each query...")

    search_start = time.perf_counter()

    retrieved_per_query, latencies_ms = retriever.search_many(
        query_embeddings=query_embeddings,
        top_k=TOP_K,
    )

    total_search_seconds = time.perf_counter() - search_start

    # -------------------------------------------------------------------------
    # 6. Evaluate with your metrics.py
    # -------------------------------------------------------------------------
    metrics = evaluate(
        eval_set=eval_set,
        retrieved_per_query=retrieved_per_query,
        ks=KS,
    )

    # -------------------------------------------------------------------------
    # 7. Latency and RAM
    # -------------------------------------------------------------------------
    latency_p50 = float(np.percentile(latencies_ms, 50))
    latency_p95 = float(np.percentile(latencies_ms, 95))
    latency_p99 = float(np.percentile(latencies_ms, 99))

    result = {
        "size": size,
        "model": MODEL_NAME,
        "num_eval_queries": len(eval_set),

        # Quality metrics from metrics.py
        "recall@1": metrics["recall@1"],
        "recall@5": metrics["recall@5"],
        "recall@10": metrics["recall@10"],
        "mrr@10": metrics["mrr@10"],

        # Latency metrics
        "total_search_seconds": round(total_search_seconds, 4),
        "latency_p50_ms": round(latency_p50, 4),
        "latency_p95_ms": round(latency_p95, 4),
        "latency_p99_ms": round(latency_p99, 4),

        # Embedding metrics
        "embeddings_loaded_from_cache": loaded_from_cache,
        "embedding_seconds": (
            round(embedding_seconds, 4)
            if embedding_seconds is not None
            else None
        ),
        "embedding_throughput_passages_per_sec": (
            round(embedding_throughput, 4)
            if embedding_throughput is not None
            else None
        ),

        # Memory metrics
        "index_ram_mb": round(retriever.index_ram_mb, 4),
        "process_ram_mb": round(get_process_ram_mb(), 4),
    }

    print("\nResult:")
    print(json.dumps(result, indent=2))

    return result


def find_breakpoints(results: list[dict]) -> list[dict]:
    """
    Find points where metrics degrade by 20%+ compared to the smallest corpus.

    For quality metrics:
        lower is worse.

    For latency:
        higher is worse.

    Example:
        recall@10 at 1K = 0.80
        recall@10 at 100K = 0.60

        drop = (0.80 - 0.60) / 0.80 = 25%
        => breakpoint detected.
    """
    if not results:
        return []

    baseline = results[0]
    breakpoints: list[dict] = []

    quality_metrics = ["recall@1", "recall@5", "recall@10", "mrr@10"]
    latency_metrics = ["latency_p50_ms", "latency_p95_ms", "latency_p99_ms"]

    for row in results[1:]:
        size = row["size"]

        for metric in quality_metrics:
            base_value = baseline[metric]
            current_value = row[metric]

            if base_value == 0:
                continue

            drop = (base_value - current_value) / base_value

            if drop >= 0.20:
                breakpoints.append(
                    {
                        "size": size,
                        "metric": metric,
                        "baseline": base_value,
                        "current": current_value,
                        "change_percent": round(-drop * 100, 2),
                        "reason": "quality degraded by 20%+",
                    }
                )

        for metric in latency_metrics:
            base_value = baseline[metric]
            current_value = row[metric]

            if base_value == 0:
                continue

            growth = (current_value - base_value) / base_value

            if growth >= 0.20:
                breakpoints.append(
                    {
                        "size": size,
                        "metric": metric,
                        "baseline": base_value,
                        "current": current_value,
                        "change_percent": round(growth * 100, 2),
                        "reason": "latency increased by 20%+",
                    }
                )

    return breakpoints


def main() -> None:
    """
    Main scaling loop.

    This is the actual scaling loop:

        for size in [1K, 10K, 100K, 300K]:
            build subset
            embed corpus
            build retriever
            run retrieval
            evaluate
            save results

    The goal is to observe how retrieval quality and latency change
    as the corpus grows.
    """
    pool, eval_set = ensure_corpus_cache()

    print(f"Loaded corpus pool: {len(pool):,} docs")
    print(f"Loaded eval queries: {len(eval_set):,}")

    embedder = BGEEmbedder(
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
    )

    all_results: list[dict] = []

    for size in SUBSET_SIZES:
        result = run_one_size(
            size=size,
            pool=pool,
            eval_set=eval_set,
            embedder=embedder,
        )

        all_results.append(result)

        # Save partial results after every size.
        # If the script crashes at 300K, you still keep 1K/10K/100K results.
        results_df = pd.DataFrame(all_results)

        results_csv_path = RESULTS_DIR / "baseline_scaling_results.csv"
        results_df.to_csv(results_csv_path, index=False)

        print(f"Saved partial results to: {results_csv_path}")

    breakpoints = find_breakpoints(all_results)

    breakpoints_path = RESULTS_DIR / "baseline_breakpoints.json"
    json.dump(
        breakpoints,
        open(breakpoints_path, "w", encoding="utf-8"),
        indent=2,
    )

    print("\n" + "=" * 90)
    print("Final scaling results")
    print("=" * 90)

    final_df = pd.DataFrame(all_results)
    print(final_df)

    print("\nBreakpoints:")
    print(json.dumps(breakpoints, indent=2))

    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
