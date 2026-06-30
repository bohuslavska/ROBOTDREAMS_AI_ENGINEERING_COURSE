import time
import uuid
from dataclasses import dataclass

from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)

from app.config import settings
from app.vector_store import get_qdrant_client


@dataclass
class SemanticCacheHit:
    query: str
    response: str
    model: str
    score: float
    sources: list[str]
    created_at: int
    expire_at: int
    usage: dict


def ensure_cache_collection() -> None:
    """
    Create semantic cache collection if it does not exist, then ensure the
    payload index on ``expire_at`` exists.  The index is required by Qdrant
    for Range filters on numeric fields.  Creating it is idempotent, so it is
    safe to call on every startup even when the collection already exists.
    """
    client = get_qdrant_client()

    if not client.collection_exists(settings.qdrant_cache_collection):
        client.create_collection(
            collection_name=settings.qdrant_cache_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE,
            ),
        )

    client.create_payload_index(
        collection_name=settings.qdrant_cache_collection,
        field_name="expire_at",
        field_schema=PayloadSchemaType.INTEGER,
    )


def find_semantic_cache_hit(
    query_embedding: list[float],
) -> SemanticCacheHit | None:
    """
    Search semantic cache by query embedding.

    HIT condition:
        top result exists
        similarity score > threshold
        expire_at is still in the future
    """
    ensure_cache_collection()

    client = get_qdrant_client()
    now = int(time.time())

    results = client.query_points(
        collection_name=settings.qdrant_cache_collection,
        query=query_embedding,
        limit=1,
        with_payload=True,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="expire_at",
                    range=Range(gte=now),
                )
            ]
        ),
    )

    if not results.points:
        return None

    point = results.points[0]

    if point.score <= settings.semantic_cache_threshold:
        return None

    payload = point.payload or {}

    return SemanticCacheHit(
        query=str(payload.get("query", "")),
        response=str(payload.get("response", "")),
        model=str(payload.get("model", "")),
        score=float(point.score),
        sources=list(payload.get("sources", [])),
        created_at=int(payload.get("created_at", 0)),
        expire_at=int(payload.get("expire_at", 0)),
        usage=dict(payload.get("usage", {})),
    )


def store_semantic_cache_entry(
    query_embedding: list[float],
    query: str,
    response: str,
    model: str,
    sources: list[str],
    usage: dict,
) -> None:
    """
    Store successful LLM answer in semantic cache.

    TTL is implemented through expire_at payload field.
    Qdrant will not delete it automatically, but lookup ignores expired entries.
    """
    ensure_cache_collection()

    client = get_qdrant_client()
    now = int(time.time())

    point = PointStruct(
        id=uuid.uuid4().hex,
        vector=query_embedding,
        payload={
            "query": query,
            "response": response,
            "model": model,
            "sources": sources,
            "usage": usage,
            "created_at": now,
            "expire_at": now + settings.semantic_cache_ttl_seconds,
        },
    )

    client.upsert(
        collection_name=settings.qdrant_cache_collection,
        points=[point],
    )