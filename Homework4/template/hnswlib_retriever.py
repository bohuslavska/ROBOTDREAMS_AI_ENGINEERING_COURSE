"""
HNSWLib retriever.

This is the scaling fix for the dense numpy brute-force baseline.

Baseline:
    DenseNumpyRetriever compares each query vector with every document vector.
    This is exact, but search latency grows with corpus size: O(N).

HNSW fix:
    hnswlib builds an approximate nearest-neighbor graph over embeddings.
    At search time, it navigates this graph instead of scanning all vectors.

Main trade-off:
    HNSW is much faster on large corpora, but it is approximate.
    Recall can slightly decrease depending on ef_search.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import hnswlib
import numpy as np


@dataclass
class SearchResult:
    """
    One retrieved document.

    doc_id:
        Original external document id from MS MARCO / BeIR.

    score:
        Similarity score converted from hnswlib distance.

    internal_index:
        Integer index inside hnswlib.
    """

    doc_id: str
    score: float
    internal_index: int


class HNSWLibRetriever:
    """
    Dense retriever based on hnswlib.

    This class has the same role as DenseNumpyRetriever:
        query embedding -> top-k document ids

    Difference:
        DenseNumpyRetriever scans all vectors.
        HNSWLibRetriever uses an approximate nearest-neighbor graph.

    Parameters:
        m:
            Number of graph connections per vector.
            Higher M usually improves recall but uses more RAM.

        ef_construction:
            How carefully the graph is built.
            Higher value means better graph quality but slower build.

        ef_search:
            How deeply the graph is searched at query time.
            Higher value improves recall but increases latency.
    """

    def __init__(
        self,
        m: int = 32,
        ef_construction: int = 80,
        ef_search: int = 64,
        space: str = "cosine",
    ) -> None:
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.space = space

        self.doc_ids: list[str] = []
        self.index: hnswlib.Index | None = None
        self.dimension: int | None = None

    def build(
        self,
        corpus: list[dict],
        doc_embeddings: np.ndarray,
    ) -> float:
        """
        Build HNSW index from document embeddings.

        corpus[i] must correspond to doc_embeddings[i].
        This order is critical because hnswlib returns internal integer labels,
        and we map those labels back to corpus document ids.
        """
        if len(corpus) != len(doc_embeddings):
            raise ValueError(
                f"Corpus size {len(corpus)} does not match "
                f"embeddings size {len(doc_embeddings)}."
            )

        if doc_embeddings.ndim != 2:
            raise ValueError(
                f"doc_embeddings must be 2D, got shape {doc_embeddings.shape}"
            )

        self.doc_ids = [str(doc["id"]) for doc in corpus]

        embeddings = np.ascontiguousarray(
            doc_embeddings.astype("float32")
        )

        num_docs, dim = embeddings.shape
        self.dimension = dim

        start = time.perf_counter()

        self.index = hnswlib.Index(
            space=self.space,
            dim=dim,
        )

        self.index.init_index(
            max_elements=num_docs,
            ef_construction=self.ef_construction,
            M=self.m,
        )

        # hnswlib uses integer labels.
        # We set label i for embeddings[i], then map i -> doc_ids[i].
        labels = np.arange(num_docs)

        self.index.add_items(
            data=embeddings,
            ids=labels,
        )

        self.index.set_ef(self.ef_search)

        build_seconds = time.perf_counter() - start

        return build_seconds

    def search_one(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> tuple[list[SearchResult], float]:
        """
        Search top-k documents for one query embedding.

        hnswlib with space='cosine' returns distances, not similarities.
        For cosine space:
            lower distance = better match

        We convert it to a score for readability:
            score = 1 - distance
        """
        if self.index is None:
            raise RuntimeError("Index is not built. Call build() first.")

        if query_embedding.ndim != 1:
            raise ValueError(
                f"query_embedding must be 1D, got shape {query_embedding.shape}"
            )

        query = np.ascontiguousarray(
            query_embedding.reshape(1, -1).astype("float32")
        )

        start = time.perf_counter()

        labels, distances = self.index.knn_query(
            query,
            k=top_k,
        )

        latency_ms = (time.perf_counter() - start) * 1000

        results: list[SearchResult] = []

        for internal_index, distance in zip(labels[0], distances[0]):
            internal_index = int(internal_index)

            results.append(
                SearchResult(
                    doc_id=self.doc_ids[internal_index],
                    score=float(1.0 - distance),
                    internal_index=internal_index,
                )
            )

        return results, latency_ms

    def search_many(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10,
    ) -> tuple[list[list[str]], list[float]]:
        """
        Search top-k documents for many queries.

        Returns the exact format expected by metrics.py:

            retrieved_per_query = [
                ["doc1", "doc2", ...],  # results for eval_set[0]
                ["doc7", "doc3", ...],  # results for eval_set[1]
            ]
        """
        retrieved_per_query: list[list[str]] = []
        latencies_ms: list[float] = []

        for query_embedding in query_embeddings:
            results, latency_ms = self.search_one(
                query_embedding=query_embedding,
                top_k=top_k,
            )

            retrieved_per_query.append([result.doc_id for result in results])
            latencies_ms.append(latency_ms)

        return retrieved_per_query, latencies_ms

    @property
    def ntotal(self) -> int:
        """
        Number of vectors stored in the HNSW index.
        """
        if self.index is None:
            return 0

        return len(self.doc_ids)