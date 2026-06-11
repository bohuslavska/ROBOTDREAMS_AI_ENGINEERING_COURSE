import math
import time
from dataclasses import dataclass

import redis.asyncio as redis

from app.config import settings


WINDOW_SECONDS = 60


@dataclass
class RateLimitResult:
    allowed: bool
    retry_after_seconds: int = 0
    used_tokens: int = 0
    limit_tokens: int = 0


_redis_client: redis.Redis | None = None


def get_redis_client() -> redis.Redis:
    """
    Create and reuse Redis client.
    """
    global _redis_client

    if _redis_client is None:
        _redis_client = redis.from_url(
            settings.redis_url,
            decode_responses=True,
        )

    return _redis_client


def _usage_key(api_key: str, second: int) -> str:
    """
    Redis key for one second of usage.

    Example:
        rate_limit:demo-free-key:tokens:1780000000
    """
    return f"rate_limit:{api_key}:tokens:{second}"


async def get_tokens_used_last_60_seconds(api_key: str) -> tuple[int, dict[int, int]]:
    """
    Read token usage from the last 60 seconds.

    We keep usage in per-second Redis buckets:
        second_1 -> 120 tokens
        second_2 -> 80 tokens
        ...

    This uses standard Redis GET/MGET-compatible logic.
    """
    client = get_redis_client()
    now = int(time.time())

    seconds = list(range(now - WINDOW_SECONDS + 1, now + 1))
    keys = [_usage_key(api_key, second) for second in seconds]

    values = await client.mget(keys)

    usage_by_second: dict[int, int] = {}
    total = 0

    for second, value in zip(seconds, values):
        tokens = int(value or 0)

        if tokens > 0:
            usage_by_second[second] = tokens
            total += tokens

    return total, usage_by_second


def calculate_retry_after_seconds(
    usage_by_second: dict[int, int],
    limit_tokens: int,
) -> int:
    """
    Estimate when enough old token usage expires.

    Since every per-second bucket expires after 60 seconds,
    we check when the rolling 60-second usage will go below the limit.
    """
    now = int(time.time())
    current_usage = sum(usage_by_second.values())

    if current_usage < limit_tokens:
        return 0

    running_usage = current_usage

    for second in sorted(usage_by_second):
        tokens_at_second = usage_by_second[second]
        expires_in = max(1, WINDOW_SECONDS - (now - second))

        running_usage -= tokens_at_second

        if running_usage < limit_tokens:
            return expires_in

    return WINDOW_SECONDS


async def check_rate_limit(
    api_key: str,
    limit_tokens_per_minute: int,
) -> RateLimitResult:
    """
    Check whether this API key is currently allowed to make a request.

    Important:
    This does not consume tokens yet.
    Tokens are consumed only after successful LLM completion.
    """
    used_tokens, usage_by_second = await get_tokens_used_last_60_seconds(api_key)

    if used_tokens >= limit_tokens_per_minute:
        retry_after = calculate_retry_after_seconds(
            usage_by_second=usage_by_second,
            limit_tokens=limit_tokens_per_minute,
        )

        return RateLimitResult(
            allowed=False,
            retry_after_seconds=retry_after,
            used_tokens=used_tokens,
            limit_tokens=limit_tokens_per_minute,
        )

    return RateLimitResult(
        allowed=True,
        retry_after_seconds=0,
        used_tokens=used_tokens,
        limit_tokens=limit_tokens_per_minute,
    )


async def record_token_usage(
    api_key: str,
    tokens_used: int,
) -> None:
    """
    Record actual token usage after successful LLM response.

    Uses Redis INCRBY + EXPIRE pattern.
    No Lua scripts.
    """
    if tokens_used <= 0:
        return

    client = get_redis_client()
    now = int(time.time())
    key = _usage_key(api_key, now)

    await client.incrby(key, tokens_used)
    await client.expire(key, WINDOW_SECONDS + 5)