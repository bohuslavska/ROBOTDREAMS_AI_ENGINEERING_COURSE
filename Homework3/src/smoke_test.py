"""
Швидкий smoke-test для будь-якої VectorDB-обгортки.

Робить 3 базові перевірки:
  1. Self-query: беремо корпусний вектор, шукаємо його ж — top-1 score має бути ~1.0
     (бо cos(v, v) = 1, якщо нормалізовано). Це класичний sanity-check для similarity search.
  2. Real query: беремо реальний query вектор, шукаємо top-10, друкуємо результати.
  3. Disk size: перевіряємо, що індекс зберігся на диск (для disk_size_mb).

Запуск:
    python src/smoke_test.py --db faiss_flat
    python src/smoke_test.py --db faiss_flat --num-docs 1000   # ще швидше

Якщо все ОК — побачиш у кінці "✓ ALL CHECKS PASSED".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# Додаємо корінь репо у sys.path, щоб `from src.benchmarks.*` працював при
# запуску як `python src/smoke_test.py` (за замовчуванням python додає лише
# папку зі скриптом, тобто src/ — звідки `src` як пакет невидимий).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_npy_and_ids(name: str) -> tuple[np.ndarray, list[str]]:
    """Завантажує data/<name>.npy + data/<name>.npy.ids.json."""
    npy_path = DATA_DIR / f"{name}.npy"
    ids_path = DATA_DIR / f"{name}.npy.ids.json"
    if not npy_path.exists():
        raise FileNotFoundError(f"{npy_path} не існує. Запусти embed.py спочатку.")
    if not ids_path.exists():
        raise FileNotFoundError(f"{ids_path} не існує.")

    vectors = np.load(npy_path)
    with ids_path.open() as f:
        ids = json.load(f)

    if vectors.shape[0] != len(ids):
        raise ValueError(f"Mismatch: {vectors.shape[0]} vectors vs {len(ids)} ids")
    return vectors, ids


def _make_db(name: str, index_path: Path):
    """Фабрика: за іменем повертає інстанс відповідної VectorDB."""
    if name == "faiss_flat":
        from src.benchmarks.faiss_flat import FaissFlat
        return FaissFlat(index_path=index_path)
    if name == "faiss_hnsw":
        from src.benchmarks.faiss_hnsw import FaissHNSW
        return FaissHNSW(index_path=index_path)
    if name == "chroma":
        from src.benchmarks.chroma_db import ChromaDB
        # Для Chroma index_path — це директорія, не файл.
        return ChromaDB(index_path=index_path.parent / f"{index_path.stem}_dir")
    if name == "qdrant":
        from src.benchmarks.qdrant_db import QdrantDB
        return QdrantDB()
    if name == "pgvector":
        from src.benchmarks.pgvector_db import PgVectorDB
        return PgVectorDB()
    raise ValueError(f"Unknown db: {name}")


def smoke_test(db_name: str, num_docs: int, num_queries: int, top_k: int) -> None:
    print(f"\n{'='*60}\nSMOKE TEST: {db_name}\n{'='*60}\n")

    # --- Load data ---
    print(f"[load] Завантажую {num_docs:,} corpus vectors і {num_queries} queries...")
    corpus_vecs, corpus_ids = _load_npy_and_ids("embeddings_corpus")
    query_vecs, query_ids = _load_npy_and_ids("embeddings_queries")

    corpus_vecs = corpus_vecs[:num_docs]
    corpus_ids = corpus_ids[:num_docs]
    query_vecs = query_vecs[:num_queries]
    query_ids = query_ids[:num_queries]
    print(f"[load] corpus: {corpus_vecs.shape}, queries: {query_vecs.shape}")

    # --- Build index ---
    index_path = DATA_DIR / f"_smoke_{db_name}.index"
    db = _make_db(db_name, index_path)

    print(f"\n[index] Будую індекс...")
    t0 = time.perf_counter()
    db.index(corpus_vecs, ids=corpus_ids)
    t_index = time.perf_counter() - t0
    print(f"[index] OK за {t_index:.2f}s")

    # --- CHECK 1: Self-query ---
    print(f"\n[check 1] Self-query (top-1 score має бути ~1.0)...")
    sample_indices = [0, num_docs // 2, num_docs - 1]
    for i in sample_indices:
        results = db.search(corpus_vecs[i], top_k=1)
        if not results:
            raise AssertionError(f"Self-query на vec[{i}] повернув порожній список")
        top_id, top_score = results[0]
        expected_id = corpus_ids[i]
        assert top_id == expected_id, f"Top-1 ID '{top_id}' != expected '{expected_id}'"
        assert top_score > 0.99, f"Top-1 score {top_score:.4f} < 0.99 — норми не 1?"
        print(f"  vec[{i}]: top1=({top_id}, {top_score:.4f}) ✓")

    # --- CHECK 2: Real queries ---
    print(f"\n[check 2] Real queries (top-{top_k})...")
    latencies = []
    for i, q_vec in enumerate(query_vecs[:3]):  # друкуємо тільки перші 3
        t0 = time.perf_counter()
        results = db.search(q_vec, top_k=top_k)
        latencies.append((time.perf_counter() - t0) * 1000)

        assert len(results) == top_k, f"Очікувано {top_k} результатів, отримано {len(results)}"
        assert all(isinstance(r[0], str) for r in results), "IDs мають бути рядками"
        assert all(isinstance(r[1], float) for r in results), "Scores мають бути float"

        print(f"  query_id={query_ids[i]}: top-3 = "
              f"{[(rid, round(s, 3)) for rid, s in results[:3]]}")

    # Решта queries без друку, тільки виміри
    for q_vec in query_vecs[3:]:
        t0 = time.perf_counter()
        db.search(q_vec, top_k=top_k)
        latencies.append((time.perf_counter() - t0) * 1000)

    print(f"\n[latency] {num_queries} queries:")
    print(f"  p50 = {np.percentile(latencies, 50):.3f} ms")
    print(f"  p95 = {np.percentile(latencies, 95):.3f} ms")
    print(f"  max = {max(latencies):.3f} ms")

    # --- CHECK 3: Disk size ---
    print(f"\n[check 3] Disk size...")
    size_mb = db.disk_size_mb()
    print(f"  disk_size_mb() = {size_mb:.2f} MB")
    # FAISS/Chroma пишуть локально, тому assert > 0.
    # Qdrant/pgvector тримають дані в Docker — disk size береться через docker exec
    # або pg_total_relation_size, може бути 0 на дрібних об'ємах. Не assert'имо.
    if db_name in ("faiss_flat", "faiss_hnsw", "chroma"):
        assert size_mb > 0, "Очікували > 0, бо передали index_path"

    # --- Cleanup ---
    db.cleanup()
    print(f"  cleanup OK ✓")

    print(f"\n{'='*60}\n✓ ALL CHECKS PASSED for {db_name}\n{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db",
        default="faiss_flat",
        choices=["faiss_flat", "faiss_hnsw", "chroma", "qdrant", "pgvector"],
    )
    parser.add_argument("--num-docs", type=int, default=5000)
    parser.add_argument("--num-queries", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    smoke_test(args.db, args.num_docs, args.num_queries, args.top_k)


if __name__ == "__main__":
    main()
