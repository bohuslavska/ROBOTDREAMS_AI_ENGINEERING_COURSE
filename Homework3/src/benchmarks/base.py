from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import numpy as np


class VectorDB(ABC):
    """Спільний інтерфейс для FAISS / Qdrant / Chroma / pgvector."""

    @abstractmethod
    def index(self, vectors: np.ndarray, ids: List[str]) -> None:
        """
        Побудувати індекс з векторів.
        vectors: shape (N, dim), float32, L2-нормалізовані для cosine
        ids: рядкові ID, паралельні до vectors
        """

    @abstractmethod
    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Знайти top-K найближчих векторів.
        query_vec: shape (dim,) — 1D! Wrapper має сам зробити reshape якщо треба.
        Повертає: [(doc_id, score), ...] довжиною top_k
        """

    @abstractmethod
    def disk_size_mb(self) -> float:
        """Розмір індексу на диску в MB (0 якщо in-memory)."""

    def cleanup(self) -> None:
        """Закрити з'єднання, видалити тимчасові файли. За замовчуванням no-op."""
        pass
