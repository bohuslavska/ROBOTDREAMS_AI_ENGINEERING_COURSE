"""
FAISS HNSW (Hierarchical Navigable Small World) — approximate search.

Принцип:
- Будує багаторівневий граф сусідства. Верхні рівні — рідкі "магістралі",
  нижні — щільні локальні зв'язки.
- Пошук: спускаємось з верхнього рівня, на кожному рівні greedy-пошук
  найближчої точки, потім стрибок униз. Це O(log N) проти O(N) у Flat.
- Recall <100%, бо greedy-пошук може пропустити справжнього сусіда.

Параметри (можна налаштовувати в __init__):
    M               — кількість зв'язків на вершину (default 16)
    efConstruction  — глибина пошуку при побудові (default 200)
    efSearch        — глибина пошуку при query (default 64)

Більше M/ef → краще recall, але повільніше і більше памʼяті.
Для Pareto frontier варто пробувати ef_search ∈ {16, 32, 64, 128, 256}.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np

from src.benchmarks.base import VectorDB


class FaissHNSW(VectorDB):
    """Approximate search через FAISS IndexHNSWFlat (IP metric)."""

    def __init__(
        self,
        index_path: Optional[Path] = None,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
    ) -> None:
        """
        Args:
            index_path: куди зберегти індекс на диск (для disk_size_mb).
            M: кількість зв'язків на вершину (більше → краще recall, більше RAM).
            ef_construction: глибина пошуку при побудові (більше → краще recall, повільніше).
            ef_search: глибина пошуку при query (більше → краще recall, повільніший пошук).
        """
        self.index_path = Path(index_path) if index_path else None
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        self._index: Optional[faiss.IndexHNSWFlat] = None
        self._ids: List[str] = []

    def index(self, vectors: np.ndarray, ids: List[str]) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if vectors.shape[0] != len(ids):
            raise ValueError(f"vectors.shape[0]={vectors.shape[0]} != len(ids)={len(ids)}")

        dim = vectors.shape[1]

        # IndexHNSWFlat: HNSW + raw vectors (без квантизації).
        # METRIC_INNER_PRODUCT — бо наші вектори нормалізовані, IP == cosine.
        self._index = faiss.IndexHNSWFlat(dim, self.M, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efConstruction = self.ef_construction
        self._index.hnsw.efSearch = self.ef_search

        self._index.add(vectors)
        self._ids = list(ids)

        if self.index_path is not None:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(self.index_path))

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        if self._index is None:
            raise RuntimeError("Index не побудований. Виклич .index(...) спочатку.")

        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype(np.float32)

        distances, indices = self._index.search(query_vec, top_k)

        return [
            (self._ids[idx], float(score))
            for idx, score in zip(indices[0], distances[0])
            if idx != -1
        ]

    def disk_size_mb(self) -> float:
        if self.index_path is None or not self.index_path.exists():
            return 0.0
        return self.index_path.stat().st_size / 1024 / 1024

    def cleanup(self) -> None:
        if self.index_path is not None and self.index_path.exists():
            self.index_path.unlink()
