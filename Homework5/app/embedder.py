from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load embedding model once and reuse it.

    all-MiniLM-L6-v2 produces 384-dimensional vectors.
    """
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert multiple texts/chunks into embeddings.

    Used during indexing:
        chunks -> embeddings -> Qdrant
    """
    if not texts:
        return []

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """
    Convert user query into one embedding.

    Used during retrieval:
        user query -> query embedding -> vector search
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")

    model = get_embedding_model()

    embedding = model.encode(
        query,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embedding.tolist()


def validate_embedding_dimension(embedding: list[float]) -> None:
    """
    Make sure embedding size matches Qdrant collection config.
    """
    actual_dimension = len(embedding)

    if actual_dimension != settings.embedding_dim:
        raise ValueError(
            f"Invalid embedding dimension: expected {settings.embedding_dim}, "
            f"got {actual_dimension}"
        )