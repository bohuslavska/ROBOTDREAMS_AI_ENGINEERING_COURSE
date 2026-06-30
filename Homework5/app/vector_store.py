from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings


def get_qdrant_client() -> QdrantClient:
    """
    Create Qdrant client.

    For local Qdrant:
        QDRANT_URL=http://localhost:6333
        QDRANT_API_KEY=

    For Qdrant Cloud:
        QDRANT_URL=https://...
        QDRANT_API_KEY=...
    """
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )


def recreate_collection() -> None:
    """
    Delete existing collection if it exists and create a fresh one.

    This is good for homework/dev mode because every indexing run starts clean.
    """
    client = get_qdrant_client()

    if client.collection_exists(collection_name=settings.qdrant_collection):
        client.delete_collection(collection_name=settings.qdrant_collection)

    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(
            size=settings.embedding_dim,
            distance=Distance.COSINE,
        ),
    )


def upsert_chunks(
    chunk_ids: list[str],
    chunks: list[str],
    embeddings: list[list[float]],
    document_id: str = "source",
) -> None:
    """
    Store chunks and their vectors in Qdrant.

    Each Qdrant point contains:
    - id: integer point id
    - vector: embedding
    - payload: chunk_id, document_id, text
    """
    if not (len(chunk_ids) == len(chunks) == len(embeddings)):
        raise ValueError(
            "chunk_ids, chunks, and embeddings must have the same length. "
            f"Got chunk_ids={len(chunk_ids)}, chunks={len(chunks)}, embeddings={len(embeddings)}"
        )

    client = get_qdrant_client()

    points = []

    for index, (chunk_id, chunk_text, embedding) in enumerate(
        zip(chunk_ids, chunks, embeddings)
    ):
        points.append(
            PointStruct(
                id=index,
                vector=embedding,
                payload={
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "text": chunk_text,
                },
            )
        )

    client.upsert(
        collection_name=settings.qdrant_collection,
        points=points,
    )


def search_chunks(
    query_embedding: list[float],
    top_k: int = 3,
) -> list[dict]:
    """
    Search Qdrant by query embedding and return top-k chunks.

    Returns list of dicts:
    [
        {
            "chunk_id": "...",
            "document_id": "...",
            "text": "...",
            "score": 0.82,
        }
    ]
    """
    client = get_qdrant_client()

    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )

    found_chunks: list[dict] = []

    for point in results.points:
        payload = point.payload or {}

        found_chunks.append(
            {
                "chunk_id": payload.get("chunk_id"),
                "document_id": payload.get("document_id"),
                "text": payload.get("text"),
                "score": point.score,
            }
        )

    return found_chunks