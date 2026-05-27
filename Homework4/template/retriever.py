"""
Dense retriever module.

This file implements a simple numpy brute-force retriever.

"Brute-force" means:
For every query, compare the query embedding with every document embedding.

This is intentionally simple and not optimized.
It is useful as a baseline because the homework asks you to observe how
latency grows when corpus size grows:

    1K -> 10K -> 100K -> 300K

Complexity:
    search time per query = O(N)

where N is the number of documents/passages in the corpus.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class SearchResult:
    """
    One retrieved document.

    doc_id:
        ID of the retrieved document/passage.

    score:
        Similarity score between query and document.
        In our case this is cosine similarity because embeddings are normalized.
    """

    doc_id: str
    score: float


class DenseNumpyRetriever:
    """
    Dense vector retriever based on numpy.

    Responsibilities:
    1. Store document ids and document embeddings.
    2. For each query embedding, calculate similarity to all documents.
    3. Return top-k document ids ordered by similarity.

    Important:
    This retriever does NOT create a real ANN index like FAISS/Qdrant/HNSW.
    It does exact brute-force search.

    That is good for baseline experiments because it clearly shows the cost of
    searching through all vectors.
    """

    def __init__(self) -> None:
        self.doc_ids: list[str] = []
        self.doc_embeddings: np.ndarray | None = None

    def build(
        self,
        corpus: list[dict],
        doc_embeddings: np.ndarray,
    ) -> None:
        """
        Store corpus ids and embeddings.

        Parameters
        ----------
        corpus:
            List of documents from data_loader.build_subset().

            Expected format:
                [
                    {"id": "doc1", "text": "..."},
                    {"id": "doc2", "text": "..."},
                    ...
                ]

        doc_embeddings:
            Matrix of embeddings for the same documents.

            Shape:
                (number_of_documents, embedding_dimension)

        Why order matters:
            corpus[i] must correspond to doc_embeddings[i].
            If this order is broken, retrieved doc_ids will be wrong.
        """
        if len(corpus) != len(doc_embeddings):
            raise ValueError(
                f"Corpus size {len(corpus)} does not match "
                f"embeddings size {len(doc_embeddings)}."
            )

        self.doc_ids = [str(doc["id"]) for doc in corpus]
        self.doc_embeddings = doc_embeddings.astype("float32")

    def search_one(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> tuple[list[SearchResult], float]:
        """
        Search top-k documents for one query.

        Parameters
        ----------
        query_embedding:
            One query vector.

            Shape:
                (embedding_dimension,)

        top_k:
            Number of results to return.

        Returns
        -------
        tuple[list[SearchResult], float]
            - ordered search results
            - latency in milliseconds
        """
        if self.doc_embeddings is None:
            raise RuntimeError("Retriever is not built. Call build() first.")

        if query_embedding.ndim != 1:
            raise ValueError(
                f"query_embedding must be 1D, got shape {query_embedding.shape}"
            )

        start = time.perf_counter()

        # Since all embeddings are normalized:
        # dot product == cosine similarity.
        #
        # scores shape:
        #     (number_of_documents,)
        scores = self.doc_embeddings @ query_embedding

        k = min(top_k, len(scores))

        # np.argpartition is faster than sorting all documents.
        # It gives us the indices of top-k scores, but not in sorted order.
        candidate_indices = np.argpartition(-scores, k - 1)[:k]

        # Now sort only these top-k candidates by score descending.
        sorted_indices = candidate_indices[np.argsort(-scores[candidate_indices])]

        results = [
            SearchResult(
                doc_id=self.doc_ids[i],
                score=float(scores[i]),
            )
            for i in sorted_indices
        ]

        latency_ms = (time.perf_counter() - start) * 1000

        return results, latency_ms

    def search_many(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10,
    ) -> tuple[list[list[str]], list[float]]:
        """
        Search top-k documents for many queries.

        This method returns data in the exact format expected by your metrics.py:

            retrieved_per_query = [
                ["doc1", "doc2", ...],  # results for eval_set[0]
                ["doc7", "doc3", ...],  # results for eval_set[1]
                ...
            ]

        Parameters
        ----------
        query_embeddings:
            Matrix of query embeddings.

            Shape:
                (number_of_queries, embedding_dimension)

        top_k:
            Number of documents to return per query.

        Returns
        -------
        tuple[list[list[str]], list[float]]
            - retrieved_per_query
            - latencies_ms
        """
        retrieved_per_query: list[list[str]] = []
        latencies_ms: list[float] = []

        for query_embedding in query_embeddings:
            results, latency_ms = self.search_one(
                query_embedding=query_embedding,
                top_k=top_k,
            )

            retrieved_doc_ids = [result.doc_id for result in results]

            retrieved_per_query.append(retrieved_doc_ids)
            latencies_ms.append(latency_ms)

        return retrieved_per_query, latencies_ms

    @property
    def index_ram_mb(self) -> float:
        """
        Approximate RAM used by the embedding matrix.

        This does not include all Python overhead.
        But it is enough for the homework report because it shows how RAM grows
        with corpus size.
        """
        if self.doc_embeddings is None:
            return 0.0

        return self.doc_embeddings.nbytes / 1024 / 1024
