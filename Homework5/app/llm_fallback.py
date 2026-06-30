import asyncio
import time
from dataclasses import dataclass

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from app.rate_limiter import get_redis_client


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 422}

MODEL_TIMEOUT_SECONDS = 15
CIRCUIT_BREAKER_WINDOW_SECONDS = 60
CIRCUIT_BREAKER_OPEN_SECONDS = 60
CIRCUIT_BREAKER_ERROR_THRESHOLD = 5


@dataclass
class LLMStreamSelection:
    stream: object
    model: str
    fallback_used: bool
    attempted_models: list[str]


class NoLLMModelAvailableError(Exception):
    pass


class NonRetryableLLMError(Exception):
    pass


def _circuit_error_key(model: str) -> str:
    return f"circuit_breaker:llm:{model}:errors"


def _circuit_open_until_key(model: str) -> str:
    return f"circuit_breaker:llm:{model}:open_until"


async def is_circuit_open(model: str) -> bool:
    client = get_redis_client()
    value = await client.get(_circuit_open_until_key(model))

    if value is None:
        return False

    return int(value) > int(time.time())


async def record_model_failure(model: str) -> None:
    """
    Record model failure for circuit breaker.

    If primary fails 5+ times in 60 seconds,
    circuit is opened for 60 seconds.
    """
    client = get_redis_client()

    error_key = _circuit_error_key(model)
    error_count = await client.incrby(error_key, 1)
    await client.expire(error_key, CIRCUIT_BREAKER_WINDOW_SECONDS)

    if int(error_count) >= CIRCUIT_BREAKER_ERROR_THRESHOLD:
        open_until = int(time.time()) + CIRCUIT_BREAKER_OPEN_SECONDS
        open_key = _circuit_open_until_key(model)

        await client.set(open_key, open_until)
        await client.expire(open_key, CIRCUIT_BREAKER_OPEN_SECONDS + 5)


async def record_model_success(model: str) -> None:
    """
    Reset circuit error counter after successful call.
    """
    client = get_redis_client()
    await client.delete(_circuit_error_key(model))


def _looks_like_invalid_model_error(exc: Exception) -> bool:
    """
    OpenRouter may return different status codes/messages for invalid models.

    Acceptance asks us to test fallback by setting primary to:
        openai/this-does-not-exist

    So we treat model-not-found style errors as fallback-worthy.
    """
    message = str(exc).lower()

    return (
        "model" in message
        and (
            "not found" in message
            or "does not exist" in message
            or "not a valid" in message
            or "invalid model" in message
        )
    )


def is_retryable_llm_error(exc: Exception) -> bool:
    """
    Decide whether an error should trigger fallback.
    """
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True

    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True

    if isinstance(exc, APIStatusError):
        status_code = exc.status_code

        if status_code in RETRYABLE_STATUS_CODES:
            return True

        # For acceptance test with invalid primary model.
        if status_code == 404:
            return True

        # Normally 400 is non-retryable, but invalid model is a fallback test case.
        if status_code == 400 and _looks_like_invalid_model_error(exc):
            return True

        return False

    return False


async def create_stream_with_fallback(
    client: AsyncOpenAI,
    model_chain: list[str],
    messages: list[dict],
) -> LLMStreamSelection:
    """
    Try models one by one until one streaming call starts successfully.

    Timeout per model = 15 seconds.
    Retryable errors trigger fallback.
    Non-retryable errors are returned to the client.
    """
    if not model_chain:
        raise ValueError("model_chain cannot be empty")

    primary_model = model_chain[0]
    attempted_models: list[str] = []
    last_error: Exception | None = None

    for index, model in enumerate(model_chain):
        # Circuit breaker applies to primary.
        if index == 0 and await is_circuit_open(primary_model):
            continue

        attempted_models.append(model)

        try:
            stream = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                ),
                timeout=MODEL_TIMEOUT_SECONDS,
            )

            await record_model_success(model)

            return LLMStreamSelection(
                stream=stream,
                model=model,
                fallback_used=(model != primary_model),
                attempted_models=attempted_models,
            )

        except Exception as exc:
            last_error = exc

            if index == 0:
                await record_model_failure(primary_model)

            if is_retryable_llm_error(exc):
                continue

            raise NonRetryableLLMError(str(exc)) from exc

    raise NoLLMModelAvailableError(
        f"All models failed. Attempted models: {attempted_models}. "
        f"Last error: {last_error}"
    )