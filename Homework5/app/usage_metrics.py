import time

from app.rate_limiter import get_redis_client


METRIC_TTL_SECONDS = 3600 + 120


def _cache_metric_key(event_type: str, minute: int) -> str:
    return f"metrics:cache:{event_type}:{minute}"


async def record_cache_event(cache_hit: bool) -> None:
    """
    Record cache hit/miss counter in Redis.

    Stored per minute and expires after ~1 hour.
    """
    client = get_redis_client()

    event_type = "hit" if cache_hit else "miss"
    minute = int(time.time() // 60)

    key = _cache_metric_key(event_type, minute)

    await client.incrby(key, 1)
    await client.expire(key, METRIC_TTL_SECONDS)


async def get_cache_breakdown_last_hour() -> dict:
    """
    Return cache hit rate for the last hour.
    """
    client = get_redis_client()

    current_minute = int(time.time() // 60)
    minutes = list(range(current_minute - 59, current_minute + 1))

    hit_keys = [_cache_metric_key("hit", minute) for minute in minutes]
    miss_keys = [_cache_metric_key("miss", minute) for minute in minutes]

    hit_values = await client.mget(hit_keys)
    miss_values = await client.mget(miss_keys)

    hits = sum(int(value or 0) for value in hit_values)
    misses = sum(int(value or 0) for value in miss_values)

    total = hits + misses
    hit_rate = hits / total if total else 0.0

    return {
        "window_seconds": 3600,
        "cache_hits": hits,
        "cache_misses": misses,
        "cache_total": total,
        "cache_hit_rate": hit_rate,
    }