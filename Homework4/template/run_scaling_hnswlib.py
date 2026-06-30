"""
Scaling loop for HNSWLib fix.

This script compares to the dense numpy brute-force baseline.

It uses:
- the same eval_set
- the same corpus subsets
- the same cached BGE embeddings
- the same metrics.py

Only the retriever changes:

    DenseNumpyRetriever -> HNSWLibRetriever

This makes the experiment fair:
if latency improves, it is because of the index, not because of another model.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from data_loader import (
    build_corpus_pool,
    build_subset,
    load_cache,
    load_qrels_and_queries,
    pick_eval_queries,
    save_cache,
)
from embedder import BGEEmbedder
from hnswlib_retriever import HNSWLibRetriever
from metrics import evaluate


SEED = 42

N_EVAL_QUERIES = 100
DISTRACTOR_TARGET = 300_000

SUBSET_SIZES = [1_000, 10_000, 100_000, 300_000]

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64

KS = (1, 5, 10)
TOP_K = max(KS)

HNSW_M = 32
HNSW_EF_CONSTRUCTION = 80
HNSW_EF_SEARCH = 64

BASE_DIR = Path(__file__).parent
CACHE_PATH = BASE_DIR / "cache" / "corpus.json"
EMBEDDINGS_CACHE_DIR = BASE_DIR / "cache" / "embeddings"
RESULTS_DIR = BASE_DIR / "results"

EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FORCE_REBUILD_EMBEDDINGS = False


def get_process_ram_mb() -> float:
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def ensure_corpus_cache() -> tuple[list[dict], list[dict]]:
    if CACHE_PATH.exists():
        print(f"Loading corpus cache from: {CACHE_PATH}")
        return load_cache(CACHE_PATH)

    print("Corpus cache not found. Building it now...")

    qrels, queries = load_qrels_and_queries()

    eval_set, relevant_ids = pick_eval_queries(
        qrels=qrels,
        queries=queries,
        n=N_EVAL_QUERIES,
    )

    pool = build_corpus_pool(
        relevant_ids=relevant_ids,
        n_distractors=DISTRACTOR_TARGET,
    )

    save_cache(pool=pool, eval_set=eval_set, path=CACHE_PATH)

    return pool, eval_set


def embeddings_cache_paths(size: int) -> tuple[Path, Path]:
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

        print("Cached embeddings doc ids do not match. Rebuilding embeddings.")

    print("Encoding corpus passages because cache was not found...")

    doc_texts = [doc["text"] for doc in subset]

    start = time.perf_counter()

    doc_embeddings = embedder.encode(
        texts=doc_texts,
        kind="passage",
        show_progress=True,
    )

    embedding_seconds = time.perf_counter() - start
    embedding_throughput = len(doc_texts) / embedding_seconds

    np.save(embeddings_path, doc_embeddings)
    json.dump(subset_ids, open(ids_path, "w", encoding="utf-8"))

    return doc_embeddings, embedding_seconds, embedding_throughput, False


def run_one_size_hnswlib(
    size: int,
    pool: list[dict],
    eval_set: list[dict],
    embedder: BGEEmbedder,
) -> dict:
    print("\n" + "=" * 90)
    print(f"Running HNSWLib fix for corpus size: {size:,}")
    print("=" * 90)

    subset = build_subset(
        pool=pool,
        eval_set=eval_set,
        size=size,
        seed=SEED,
    )

    print(f"Subset size: {len(subset):,} docs")

    doc_embeddings, embedding_seconds, embedding_throughput, loaded_from_cache = (
        encode_corpus_with_cache(
            size=size,
            subset=subset,
            embedder=embedder,
        )
    )

    if loaded_from_cache:
        print("Embeddings loaded from cache.")
    else:
        print(f"Embedding time: {embedding_seconds:.2f} sec")
        print(f"Embedding throughput: {embedding_throughput:.2f} passages/sec")

    print(
        "Building HNSWLib index "
        f"(M={HNSW_M}, efConstruction={HNSW_EF_CONSTRUCTION}, efSearch={HNSW_EF_SEARCH})..."
    )

    retriever = HNSWLibRetriever(
        m=HNSW_M,
        ef_construction=HNSW_EF_CONSTRUCTION,
        ef_search=HNSW_EF_SEARCH,
        space="cosine",
    )

    index_build_seconds = retriever.build(
        corpus=subset,
        doc_embeddings=doc_embeddings,
    )

    print(f"HNSWLib index build time: {index_build_seconds:.4f} sec")
    print(f"HNSWLib index vectors: {retriever.ntotal:,}")

    print("Encoding evaluation queries...")

    query_texts = [item["query"] for item in eval_set]

    query_embeddings = embedder.encode(
        texts=query_texts,
        kind="query",
        show_progress=False,
    )

    print(f"Searching top-{TOP_K} documents for each query with HNSWLib...")

    search_start = time.perf_counter()

    retrieved_per_query, latencies_ms = retriever.search_many(
        query_embeddings=query_embeddings,
        top_k=TOP_K,
    )

    total_search_seconds = time.perf_counter() - search_start

    metrics = evaluate(
        eval_set=eval_set,
        retrieved_per_query=retrieved_per_query,
        ks=KS,
    )

    latency_p50 = float(np.percentile(latencies_ms, 50))
    latency_p95 = float(np.percentile(latencies_ms, 95))
    latency_p99 = float(np.percentile(latencies_ms, 99))

    embedding_matrix_ram_mb = doc_embeddings.nbytes / 1024 / 1024

    result = {
        "size": size,
        "fix": "hnswlib",
        "model": MODEL_NAME,
        "num_eval_queries": len(eval_set),

        "hnsw_m": HNSW_M,
        "hnsw_ef_construction": HNSW_EF_CONSTRUCTION,
        "hnsw_ef_search": HNSW_EF_SEARCH,

        "recall@1": metrics["recall@1"],
        "recall@5": metrics["recall@5"],
        "recall@10": metrics["recall@10"],
        "mrr@10": metrics["mrr@10"],

        "index_build_seconds": round(index_build_seconds, 4),
        "total_search_seconds": round(total_search_seconds, 4),
        "latency_p50_ms": round(latency_p50, 4),
        "latency_p95_ms": round(latency_p95, 4),
        "latency_p99_ms": round(latency_p99, 4),

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

        "embedding_matrix_ram_mb": round(embedding_matrix_ram_mb, 4),
        "process_ram_mb": round(get_process_ram_mb(), 4),
    }

    print("\nHNSWLib result:")
    print(json.dumps(result, indent=2))

    return result


def main() -> None:
    pool, eval_set = ensure_corpus_cache()

    print(f"Loaded corpus pool: {len(pool):,} docs")
    print(f"Loaded eval queries: {len(eval_set):,}")

    embedder = BGEEmbedder(
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
    )

    all_results: list[dict] = []

    for size in SUBSET_SIZES:
        result = run_one_size_hnswlib(
            size=size,
            pool=pool,
            eval_set=eval_set,
            embedder=embedder,
        )

        all_results.append(result)

        results_csv_path = RESULTS_DIR / "hnswlib_scaling_results.csv"
        pd.DataFrame(all_results).to_csv(results_csv_path, index=False)

        print(f"Saved partial HNSWLib results to: {results_csv_path}")

    print("\n" + "=" * 90)
    print("Final HNSWLib scaling results")
    print("=" * 90)

    final_df = pd.DataFrame(all_results)
    print(final_df)

    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()