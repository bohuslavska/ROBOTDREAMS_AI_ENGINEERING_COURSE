"""
pgvector — Postgres extension через Docker.

Принцип:
- Postgres з extension `vector`, який додає тип `vector(N)` і операції.
- HNSW index створюється через стандартний SQL: CREATE INDEX ... USING hnsw.
- Запит — звичайний SELECT з ORDER BY embedding <=> query_vector.

Оператори:
    <->   euclidean distance
    <#>   negative inner product
    <=>   cosine distance  (= 1 - cosine_similarity)

Ми використовуємо <=> бо у нас нормалізовані вектори → cosine.
Конвертуємо: similarity = 1 - distance.

Нюанси:
- Insert повільний за рядками. Використовуємо execute_values з батчами.
  COPY ще швидше, але execute_values + jsonb-friendly літерали достатньо для нашого масштабу.
- HNSW параметри: m=16, ef_construction=64 — pgvector defaults. Можна підвищити.
- ef_search налаштовується як SET hnsw.ef_search = N перед запитом.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.benchmarks.base import VectorDB


TABLE_NAME = "embeddings"
DEFAULT_DSN = "host=localhost port=5432 dbname=bench user=bench password=bench"


def _format_vector(vec: np.ndarray) -> str:
    """numpy → pgvector літерал виду '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{x:.7f}" for x in vec.tolist()) + "]"


class PgVectorDB(VectorDB):
    """pgvector + HNSW + cosine."""

    def __init__(
        self,
        dsn: str = DEFAULT_DSN,
        m: int = 16,
        ef_construction: int = 64,
        ef_search: int = 64,
        index_path: Optional[Path] = None,  # ігноруємо, для compat з інтерфейсом
    ) -> None:
        self.dsn = dsn
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.index_path = index_path

        self._conn = None

    def _connect(self):
        import psycopg

        self._conn = psycopg.connect(self.dsn, autocommit=True)

    def index(self, vectors: np.ndarray, ids: List[str]) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if vectors.shape[0] != len(ids):
            raise ValueError(f"vectors.shape[0]={vectors.shape[0]} != len(ids)={len(ids)}")

        self._connect()
        dim = vectors.shape[1]

        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"DROP TABLE IF EXISTS {TABLE_NAME};")
            cur.execute(
                f"CREATE TABLE {TABLE_NAME} ("
                f"id TEXT PRIMARY KEY, "
                f"embedding vector({dim})"
                f");"
            )

            # Bulk insert через COPY — найшвидший шлях у psycopg3.
            # mogrify() з psycopg2 був видалений; copy() це канонічна заміна
            # і ~5-10x швидше за execute_many на нашому масштабі.
            n = vectors.shape[0]
            with cur.copy(f"COPY {TABLE_NAME} (id, embedding) FROM STDIN") as copy:
                for i in range(n):
                    copy.write_row((ids[i], _format_vector(vectors[i])))

            # HNSW index. vector_cosine_ops — cosine distance.
            cur.execute(
                f"CREATE INDEX ON {TABLE_NAME} "
                f"USING hnsw (embedding vector_cosine_ops) "
                f"WITH (m = {self.m}, ef_construction = {self.ef_construction});"
            )

            # ef_search на рівні сесії (вплине на майбутні запити)
            cur.execute(f"SET hnsw.ef_search = {self.ef_search};")

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        if self._conn is None:
            raise RuntimeError("Index не побудований. Виклич .index(...) спочатку.")

        if query_vec.ndim == 2:
            query_vec = query_vec[0]
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype(np.float32)

        q_literal = _format_vector(query_vec)
        with self._conn.cursor() as cur:
            # <=> — cosine distance. similarity = 1 - distance.
            cur.execute(
                f"SELECT id, 1 - (embedding <=> %s::vector) AS score "
                f"FROM {TABLE_NAME} "
                f"ORDER BY embedding <=> %s::vector "
                f"LIMIT %s;",
                (q_literal, q_literal, top_k),
            )
            rows = cur.fetchall()
        return [(row[0], float(row[1])) for row in rows]

    def disk_size_mb(self) -> float:
        if self._conn is None:
            return 0.0
        with self._conn.cursor() as cur:
            # total = table + index + toast
            cur.execute(f"SELECT pg_total_relation_size('{TABLE_NAME}');")
            size_bytes = cur.fetchone()[0]
        return size_bytes / 1024 / 1024

    def cleanup(self) -> None:
        if self._conn is not None:
            try:
                with self._conn.cursor() as cur:
                    cur.execute(f"DROP TABLE IF EXISTS {TABLE_NAME};")
                self._conn.close()
            except Exception:
                pass
            self._conn = None
