from typing import Annotated, TypedDict

from fastapi import Header, HTTPException, status


class TierMetadata(TypedDict):
    api_key: str
    tier: str
    tokens_per_minute: int
    models: list[str]


API_KEYS: dict[str, TierMetadata] = {
    "demo-free-key": {
        "tier": "demo-free",
        "tokens_per_minute": 5_000,
        "models": [
    "openrouter/free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "qwen/qwen3-coder:free",
],
    },
    "demo-pro-key": {
        "tier": "demo-pro",
        "tokens_per_minute": 20_000,
        "models": [
            "openai/gpt-4o-mini",
            "google/gemini-2.5-flash",
            "mistralai/mistral-nemo",
            "meta-llama/llama-3.1-8b-instruct",
        ],
    },
    "demo-enterprise-key": {
        "tier": "demo-enterprise",
        "tokens_per_minute": 100_000,
        "models": [
            "anthropic/claude-sonnet-4.6",
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
        ],
    },
}


async def require_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> TierMetadata:
    """
    Validate X-API-Key header and return tier metadata.

    If header is missing or invalid, raise 401 Unauthorized.
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )

    tier_metadata = API_KEYS.get(x_api_key)

    if tier_metadata is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return {
        "api_key": x_api_key,
        **tier_metadata,
    }