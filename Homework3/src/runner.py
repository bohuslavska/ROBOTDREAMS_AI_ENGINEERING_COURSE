"""
Запуск бенчмарка для всіх 5 vector DB.

Завантажує embeddings + qrels, для кожної БД:
    1. Будує індекс на повному корпусі (~523K векторів за замовчуванням).
    2. Робить warmup queries.
    3. Виконує NUM_REPEATS × num_queries запитів і вимірює latency + recall + MRR.
    4. Зберігає результат у CSV (один рядок на БД).
    5. Очищає БД.

Запуск:
    # Повний бенчмарк (~40-70 хв)
    python src/runner.py --output results/results.csv

    # Швидкий smoke на 10K документів + 200 queries (~2-5 хв)
    python src/runner.py --output results/results_quick.csv --max-docs 10000 --num-queries 200

    # Пропустити повільні БД
    python src/runner.py --output results/results.csv --skip qdrant,pgvector

    # Тільки одна БД
    python src/runner.py --output results/results.csv --only faiss_hnsw
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


WARMUP_QUERIES = 50   # перші N запитів НЕ враховуємо (cold cache, JIT)
NUM_REPEATS = 3       # повторюємо вимір, беремо медіану


# =============================================================================
# Метрики
# =============================================================================
def _recall_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    """Recall@K = |retrieved ∩ relevant| / min(K, |relevant|)."""
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]) & relevant)
    return hits / min(k, len(relevant))


def _mrr_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    """MRR@K = 1 / rank першого правильного результату (0 якщо нема)."""
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


# =============================================================================
# Ядро бенчмарка
# =============================================================================
def benchmark_db(
    db,
    doc_vectors: np.ndarray,
    doc_ids: List[str],
    query_vectors: np.ndarray,
    query_ids: List[str],
    qrels: Dict[str, set],
    top_k: int = 10,
) -> Dict:
    # === INDEX ===
    t0 = time.perf_counter()
    db.index(doc_vectors, ids=doc_ids)
    index_time = time.perf_counter() - t0

    # === WARMUP ===
    for q_vec in query_vectors[:WARMUP_QUERIES]:
        db.search(q_vec, top_k=top_k)

    # === MEASURED QUERIES (NUM_REPEATS repeats, median per query) ===
    all_latencies: List[List[float]] = []
    recalls: List[float] = []
    mrrs: List[float] = []

    for repeat in range(NUM_REPEATS):
        latencies = []
        for q_vec, q_id in zip(query_vectors, query_ids):
            t0 = time.perf_counter()
            results = db.search(q_vec, top_k=top_k)
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

            if repeat == 0:
                retrieved_ids = [doc_id for doc_id, _score in results]
                relevant = qrels.get(q_id, set())
                recalls.append(_recall_at_k(retrieved_ids, relevant, top_k))
                mrrs.append(_mrr_at_k(retrieved_ids, relevant, top_k))
        all_latencies.append(latencies)

    # медіана по repeat'ах окремо для кожного query → percentiles по всіх queries
    latencies_arr = np.median(np.array(all_latencies), axis=0)

    return {
        "index_time_sec": round(index_time, 2),
        "disk_mb": round(db.disk_size_mb(), 1),
        "latency_p50_ms": round(float(np.percentile(latencies_arr, 50)), 3),
        "latency_p95_ms": round(float(np.percentile(latencies_arr, 95)), 3),
        "latency_p99_ms": round(float(np.percentile(latencies_arr, 99)), 3),
        "recall_at_10": round(float(np.mean(recalls)), 4),
        "mrr_at_10": round(float(np.mean(mrrs)), 4),
        "num_queries": len(query_vectors),
    }


# =============================================================================
# Завантаження даних
# =============================================================================
def _load_embeddings(name: str) -> Tuple[np.ndarray, List[str]]:
    """data/<name>.npy + data/<name>.npy.ids.json → (vectors, ids)."""
    npy_path = DATA_DIR / f"{name}.npy"
    ids_path = DATA_DIR / f"{name}.npy.ids.json"
    if not npy_path.exists():
        raise FileNotFoundError(f"{npy_path} не існує. Запусти embed.py спочатку.")

    vectors = np.load(npy_path)
    with ids_path.open() as f:
        ids = json.load(f)
    if vectors.shape[0] != len(ids):
        raise ValueError(f"Mismatch: {vectors.shape[0]} vectors vs {len(ids)} ids")
    return vectors, ids


def _load_qrels(path: Path) -> Dict[str, set]:
    """data/qrels.tsv → {query_id: set(relevant_doc_ids)}."""
    if not path.exists():
        raise FileNotFoundError(f"{path} не існує. Запусти load_data.py спочатку.")

    qrels: Dict[str, set] = {}
    with path.open() as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # пропускаємо header "query-id\tcorpus-id\tscore"
        for row in reader:
            q_id, doc_id, score = row[0], row[1], int(row[2])
            if score > 0:  # 0 у TREC-форматі означає "не релевантно"
                qrels.setdefault(q_id, set()).add(doc_id)
    return qrels


def _filter_queries_by_qrels(
    query_vectors: np.ndarray,
    query_ids: List[str],
    qrels: Dict[str, set],
) -> Tuple[np.ndarray, List[str]]:
    """Залишає тільки queries, для яких є qrels (інакше recall завжди 0)."""
    keep_indices = [i for i, qid in enumerate(query_ids) if qid in qrels]
    return query_vectors[keep_indices], [query_ids[i] for i in keep_indices]


# =============================================================================
# DB factory
# =============================================================================
def _make_db(name: str):
    """За іменем повертає інстанс VectorDB з розумними дефолтами для бенчмарка."""
    index_dir = DATA_DIR / "indexes"
    index_dir.mkdir(exist_ok=True)

    if name == "faiss_flat":
        from src.benchmarks.faiss_flat import FaissFlat
        return FaissFlat(index_path=index_dir / "faiss_flat.index")
    if name == "faiss_hnsw":
        from src.benchmarks.faiss_hnsw import FaissHNSW
        return FaissHNSW(
            index_path=index_dir / "faiss_hnsw.index",
            M=16,
            ef_construction=200,
            ef_search=64,
        )
    if name == "chroma":
        from src.benchmarks.chroma_db import ChromaDB
        return ChromaDB(index_path=index_dir / "chroma_db")
    if name == "qdrant":
        from src.benchmarks.qdrant_db import QdrantDB
        return QdrantDB(m=16, ef_construct=200)
    if name == "pgvector":
        from src.benchmarks.pgvector_db import PgVectorDB
        return PgVectorDB(m=16, ef_construction=64, ef_search=64)
    raise ValueError(f"Unknown db: {name}")


ALL_DBS = ["faiss_flat", "faiss_hnsw", "chroma", "qdrant", "pgvector"]


# =============================================================================
# CSV
# =============================================================================
CSV_FIELDS = [
    "db",
    "num_docs",
    "num_queries",
    "index_time_sec",
    "disk_mb",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "recall_at_10",
    "mrr_at_10",
    "status",
    "error",
]


def _init_csv(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()


def _append_csv(output: Path, row: Dict) -> None:
    with output.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Run vector DB benchmark on all 5 DBs.")
    parser.add_argument("--output", type=Path, default=Path("results/results.csv"))
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Обмежити кількість документів (для швидкого тесту). None = всі 523K.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=2000,
        help="Скільки queries з тих, що мають qrels, використати. Default 2000.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-K для пошуку (для recall@K і MRR@K).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Запустити тільки одну БД (faiss_flat, faiss_hnsw, chroma, qdrant, pgvector).",
    )
    parser.add_argument(
        "--skip",
        type=str,
        default="",
        help="Пропустити БД через кому, напр. --skip qdrant,pgvector",
    )
    args = parser.parse_args()

    # ---- Які БД запускаємо
    if args.only:
        dbs_to_run = [args.only]
    else:
        skip = set(args.skip.split(",")) if args.skip else set()
        dbs_to_run = [db for db in ALL_DBS if db not in skip]

    print(f"\n{'='*60}")
    print(f"BENCHMARK RUNNER")
    print(f"{'='*60}")
    print(f"DBs:          {dbs_to_run}")
    print(f"Output:       {args.output}")
    print(f"Max docs:     {args.max_docs or 'ALL (523K)'}")
    print(f"Queries:      up to {args.num_queries} (from those with qrels)")
    print(f"Top-K:        {args.top_k}")

    # ---- Load data
    print(f"\n[load] Embeddings...")
    corpus_vecs, corpus_ids = _load_embeddings("embeddings_corpus")
    query_vecs, query_ids = _load_embeddings("embeddings_queries")
    print(f"[load] corpus: {corpus_vecs.shape}, queries: {query_vecs.shape}")

    print(f"[load] qrels...")
    qrels = _load_qrels(DATA_DIR / "qrels.tsv")
    print(f"[load] qrels: {len(qrels)} queries with relevance judgments")

    # ---- Filter queries (тільки ті, що мають qrels)
    query_vecs, query_ids = _filter_queries_by_qrels(query_vecs, query_ids, qrels)
    print(f"[filter] {len(query_ids)} queries мають qrels")

    # ---- Subsample
    if args.max_docs is not None and args.max_docs < corpus_vecs.shape[0]:
        corpus_vecs = corpus_vecs[: args.max_docs]
        corpus_ids = corpus_ids[: args.max_docs]
        print(f"[subset] corpus → {corpus_vecs.shape}")

    if args.num_queries < len(query_ids):
        # детермінований subsample (перші N), щоб результати були reproducible
        query_vecs = query_vecs[: args.num_queries]
        query_ids = query_ids[: args.num_queries]
        print(f"[subset] queries → {len(query_ids)}")

    # ---- CSV init
    _init_csv(args.output)
    print(f"\n[csv] {args.output} ready (with header)")

    # ---- Benchmark loop
    results_summary = []
    for db_name in dbs_to_run:
        print(f"\n{'='*60}\n[run] {db_name}\n{'='*60}")
        t_start = time.perf_counter()
        try:
            db = _make_db(db_name)
            result = benchmark_db(
                db=db,
                doc_vectors=corpus_vecs,
                doc_ids=corpus_ids,
                query_vectors=query_vecs,
                query_ids=query_ids,
                qrels=qrels,
                top_k=args.top_k,
            )
            row = {
                "db": db_name,
                "num_docs": corpus_vecs.shape[0],
                **result,
                "status": "ok",
                "error": "",
            }
            results_summary.append(row)
            _append_csv(args.output, row)

            print(f"\n[result] {db_name}:")
            for k, v in result.items():
                print(f"  {k:20s} = {v}")

            db.cleanup()
            print(f"[cleanup] OK")
        except Exception as e:
            print(f"\n[ERROR] {db_name} failed: {e}")
            traceback.print_exc()
            row = {
                "db": db_name,
                "num_docs": corpus_vecs.shape[0],
                "status": "error",
                "error": str(e),
            }
            results_summary.append(row)
            _append_csv(args.output, row)
        finally:
            elapsed = time.perf_counter() - t_start
            print(f"[elapsed] {db_name}: {elapsed:.1f}s")

    # ---- Final summary table
    print(f"\n\n{'='*60}\nSUMMARY\n{'='*60}\n")
    headers = ["db", "index_t", "disk_mb", "p50", "p95", "p99", "recall@10", "mrr@10"]
    print(f"{headers[0]:14s} {headers[1]:>9s} {headers[2]:>9s} "
          f"{headers[3]:>7s} {headers[4]:>7s} {headers[5]:>7s} "
          f"{headers[6]:>10s} {headers[7]:>8s}")
    print("-" * 80)
    for r in results_summary:
        if r.get("status") == "ok":
            print(f"{r['db']:14s} {r['index_time_sec']:>9} {r['disk_mb']:>9} "
                  f"{r['latency_p50_ms']:>7} {r['latency_p95_ms']:>7} {r['latency_p99_ms']:>7} "
                  f"{r['recall_at_10']:>10} {r['mrr_at_10']:>8}")
        else:
            print(f"{r['db']:14s}   ERROR: {r.get('error', '')[:60]}")

    print(f"\n[done] Results saved to {args.output}")


if __name__ == "__main__":
    main()
