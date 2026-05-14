"""
Chroma DB — embedded persistent vector store.

Принцип:
- Працює як SQLite: жодного сервера, файли на диску.
- Вбудований HNSW з cosine metric.
- ID можуть бути рядковими (на відміну від FAISS).
- Збереження на диск через PersistentClient(path=...).

Нюанси:
- Chroma повертає DISTANCE (1 - cos_sim), не similarity. Конвертуємо назад.
- При вставці є ліміт max_batch_size (~5461). Якщо більше — батчимо.
- storage_path — це ДИРЕКТОРІЯ (не файл, як у FAISS).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.benchmarks.base import VectorDB


COLLECTION_NAME = "benchmark"
# Chroma в нових версіях має внутрішній ліміт ~5461 на add(). Беремо запас.
BATCH_SIZE = 5000


class ChromaDB(VectorDB):
    """Embedded Chroma з HNSW + cosine."""

    def __init__(self, index_path: Optional[Path] = None) -> None:
        """
        Args:
            index_path: ДИРЕКТОРІЯ для persistent storage. Якщо None — in-memory.
        """
        self.index_path = Path(index_path) if index_path else None
        self._client = None
        self._collection = None

    def index(self, vectors: np.ndarray, ids: List[str]) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if vectors.shape[0] != len(ids):
            raise ValueError(f"vectors.shape[0]={vectors.shape[0]} != len(ids)={len(ids)}")

        import chromadb

        # Видаляємо стару директорію, якщо є — щоб не змішувати з минулим запуском
        if self.index_path is not None and self.index_path.exists():
            shutil.rmtree(self.index_path)

        if self.index_path is not None:
            self.index_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.index_path))
        else:
            self._client = chromadb.Client()

        # cosine space → distance = 1 - cosine_similarity
        # (за замовчуванням Chroma використовує L2; нам потрібен cosine)
        self._collection = self._client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # Batched insert (Chroma має внутрішній ліміт)
        n = vectors.shape[0]
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            self._collection.add(
                embeddings=vectors[start:end].tolist(),
                ids=ids[start:end],
            )

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        if self._collection is None:
            raise RuntimeError("Index не побудований. Виклич .index(...) спочатку.")

        if query_vec.ndim == 1:
            query_vec_list = query_vec.astype(np.float32).tolist()
        else:
            query_vec_list = query_vec[0].astype(np.float32).tolist()

        results = self._collection.query(
            query_embeddings=[query_vec_list],
            n_results=top_k,
        )

        # Chroma повертає distance = 1 - cosine_similarity.
        # Конвертуємо назад у similarity, щоб формат був узгоджений з FAISS.
        ids = results["ids"][0]
        distances = results["distances"][0]
        return [(doc_id, 1.0 - float(dist)) for doc_id, dist in zip(ids, distances)]

    def disk_size_mb(self) -> float:
        if self.index_path is None or not self.index_path.exists():
            return 0.0
        # Сумуємо розмір усіх файлів у директорії (sqlite + hnsw binaries)
        total = sum(f.stat().st_size for f in self.index_path.rglob("*") if f.is_file())
        return total / 1024 / 1024

    def cleanup(self) -> None:
        if self._client is not None:
            try:
                self._client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            self._client = None
            self._collection = None
        if self.index_path is not None and self.index_path.exists():
            shutil.rmtree(self.index_path)
