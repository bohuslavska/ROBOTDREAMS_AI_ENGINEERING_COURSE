"""
Embedder module.

This file is responsible for converting text into dense vectors.

In this homework:
- passages/documents are embedded once per corpus subset;
- queries are embedded once per evaluation run;
- embeddings are normalized, so dot product can be used as cosine similarity.

Why normalization matters:
If vectors are normalized to length 1, then:

    cosine_similarity(query, doc) == query_vector @ doc_vector

This makes retrieval faster and simpler.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sentence_transformers import SentenceTransformer


TextKind = Literal["query", "passage"]


class BGEEmbedder:
    """
    Small wrapper around SentenceTransformer for BGE embeddings.

    Responsibilities:
    1. Load the embedding model.
    2. Convert texts into vectors.
    3. Apply BGE query prefix for query embeddings.
    4. Normalize embeddings for cosine-similarity search.

    We keep this class separate from the retriever because:
    - embedder = "turn text into vectors"
    - retriever = "search among vectors"

    This separation makes the code easier to read and easier to replace later.
    For example, you can later replace BGE with OpenAI embeddings without
    rewriting the retriever.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 64,
        device: str | None = None,
        normalize_embeddings: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        model_name:
            HuggingFace / SentenceTransformers model name.

        batch_size:
            How many texts are encoded at once.
            Larger batch = faster, but uses more RAM/VRAM.

        device:
            Optional device, e.g. "cpu", "cuda", "mps".
            If None, SentenceTransformer decides automatically.

        normalize_embeddings:
            If True, output vectors are normalized to length 1.
            This lets us use dot product as cosine similarity.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        self.model = SentenceTransformer(model_name, device=device)

        # BGE models usually work better when queries are prefixed like this.
        # Passages are encoded as-is.
        self.query_prefix = "Represent this sentence for searching relevant passages: "

    def encode(
        self,
        texts: list[str],
        kind: TextKind,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Convert a list of texts into a 2D numpy array of embeddings.

        Parameters
        ----------
        texts:
            Input texts.

        kind:
            "query" or "passage".

            For query:
                we add the BGE query prefix.

            For passage:
                we keep the original text.

        show_progress:
            Whether to show progress bar during encoding.

        Returns
        -------
        np.ndarray
            Shape:
                (number_of_texts, embedding_dimension)

            Example:
                1000 passages, 384-dimensional model
                -> shape = (1000, 384)
        """
        if not texts:
            raise ValueError("Cannot encode empty text list.")

        if kind == "query":
            prepared_texts = [
                self.query_prefix + text
                for text in texts
            ]
        elif kind == "passage":
            prepared_texts = texts
        else:
            raise ValueError(f"Unknown text kind: {kind}")

        embeddings = self.model.encode(
            prepared_texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )

        # float32 is enough for retrieval and saves RAM compared to float64.
        return embeddings.astype("float32")
