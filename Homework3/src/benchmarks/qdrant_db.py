"""
Qdrant — серверна vector DB через Docker (порти 6333 REST / 6334 gRPC).

Принцип:
- Векторна БД як окремий сервіс. Python-клієнт ходить по HTTP/gRPC.
- HNSW + cosine з коробки.
- Production-grade: replication, snapshots, filters, payload з коробки.
- ID мають бути або int64, або UUID-строки. Звичайні рядки (як "doc_xyz") НЕ працюють.
  Тому ми мапимо рядкові ID → int (позиції 0..N-1) і назад.

Нюанси:
- При query повертає Score, який є cosine_similarity (для Distance.COSINE) — формат
  узгоджений з FAISS, конвертація не потрібна.
- Disk size: Qdrant зберігає дані у контейнері. Можна спитати через API розмір сегментів,
  або зробити `docker exec` для `du`. Тут робимо API-варіант (приблизний, але достатній).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.benchmarks.base import VectorDB


COLLECTION_NAME = "benchmark"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 6333  # REST


class QdrantDB(VectorDB):
    """Qdrant через Python-клієнт (REST)."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        m: int = 16,
        ef_construct: int = 200,
        container_name: str = "hw_qdrant",
        index_path: Optional[Path] = None,  # для сумісності з інтерфейсом (ігноруємо)
    ) -> None:
        self.host = host
        self.port = port
        self.m = m
        self.ef_construct = ef_construct
        self.container_name = container_name
        self.index_path = index_path  # не використовуємо — для compat

        self._client = None
        self._ids: List[str] = []  # позиція i → рядковий ID

    def index(self, vectors: np.ndarray, ids: List[str]) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if vectors.shape[0] != len(ids):
            raise ValueError(f"vectors.shape[0]={vectors.shape[0]} != len(ids)={len(ids)}")

        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels

        self._client = QdrantClient(host=self.host, port=self.port, timeout=120)
        self._ids = list(ids)

        # Видаляємо стару колекцію, якщо є
        try:
            self._client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        dim = vectors.shape[1]
        self._client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(
                size=dim,
                distance=qmodels.Distance.COSINE,
            ),
            hnsw_config=qmodels.HnswConfigDiff(
                m=self.m,
                ef_construct=self.ef_construct,
            ),
        )

        # upload_collection — асинхронна batched-вставка. Qdrant сам розрулює батчі.
        self._client.upload_collection(
            collection_name=COLLECTION_NAME,
            vectors=vectors,
            ids=list(range(len(ids))),  # int ID = позиції; рядкові живуть у self._ids
            batch_size=512,
            parallel=2,
            wait=True,  # чекаємо, поки всі вставляться, перед поверненням
        )

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        if self._client is None:
            raise RuntimeError("Index не побудований. Виклич .index(...) спочатку.")

        if query_vec.ndim == 2:
            query_vec = query_vec[0]
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype(np.float32)

        # У qdrant-client >=1.14 метод .search() видалений на користь .query_points().
        # Це уніфікований API для всіх типів запитів (vector search, filter, hybrid).
        response = self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec.tolist(),
            limit=top_k,
        )

        # response.points — список ScoredPoint з .id (наш int) і .score (cosine similarity)
        return [(self._ids[point.id], float(point.score)) for point in response.points]

    def disk_size_mb(self) -> float:
        """
        Розмір storage Qdrant у контейнері через `docker exec du`.
        Якщо контейнер недоступний — повертаємо 0.
        """
        try:
            result = subprocess.run(
                ["docker", "exec", self.container_name, "du", "-sm", "/qdrant/storage"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                # вивід виду "123\t/qdrant/storage\n"
                size_mb = int(result.stdout.split()[0])
                return float(size_mb)
        except (subprocess.SubprocessError, ValueError, IndexError):
            pass
        return 0.0

    def cleanup(self) -> None:
        if self._client is not None:
            try:
                self._client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            self._client = None
            self._ids = []
