"""
FAISS Flat (Inner Product) — exact brute-force search.

Принцип:
- Зберігає всі вектори у RAM як суцільний масив.
- На кожен query: повний прохід по всіх N векторах + top-K.
- Recall завжди 100% (це exact-метод), але latency росте лінійно з N.
- Це наш baseline — всі інші БД будемо порівнювати з ним.

ВАЖЛИВО: ми використовуємо `IndexFlatIP` (Inner Product), бо наші
вектори вже L2-нормалізовані в embed.py. Inner product нормалізованих
векторів == cosine similarity, але швидше (нема ділення на норму).

Якщо би вектори були ненормалізовані, треба було б `IndexFlatL2`
(euclidean) або руками ділити на норми.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np

from src.benchmarks.base import VectorDB


class FaissFlat(VectorDB):
    """Точний пошук через FAISS IndexFlatIP."""

    def __init__(self, index_path: Optional[Path] = None) -> None:
        """
        Args:
            index_path: куди зберегти індекс на диск (для disk_size_mb).
                        Якщо None — індекс in-memory, disk_size_mb() == 0.
        """
        self.index_path = Path(index_path) if index_path else None
        self._index: Optional[faiss.IndexFlatIP] = None
        self._ids: List[str] = []

    def index(self, vectors: np.ndarray, ids: List[str]) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if vectors.shape[0] != len(ids):
            raise ValueError(f"vectors.shape[0]={vectors.shape[0]} != len(ids)={len(ids)}")

        dim = vectors.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vectors)
        self._ids = list(ids)

        if self.index_path is not None:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(self.index_path))

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        if self._index is None:
            raise RuntimeError("Index не побудований. Виклич .index(...) спочатку.")

        # FAISS чекає shape (n_queries, dim). Wrapper робить reshape сам.
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype(np.float32)

        # distances: (1, k), indices: (1, k) — позиції 0..N-1
        distances, indices = self._index.search(query_vec, top_k)

        return [
            (self._ids[idx], float(score))
            for idx, score in zip(indices[0], distances[0])
            if idx != -1  # FAISS повертає -1 якщо знайшло менше за k
        ]

    def disk_size_mb(self) -> float:
        if self.index_path is None or not self.index_path.exists():
            return 0.0
        return self.index_path.stat().st_size / 1024 / 1024

    def cleanup(self) -> None:
        if self.index_path is not None and self.index_path.exists():
            self.index_path.unlink()
