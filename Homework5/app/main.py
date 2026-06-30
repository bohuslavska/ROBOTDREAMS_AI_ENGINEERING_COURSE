import asyncio
import json
import re
import time
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.auth import TierMetadata, require_api_key
from app.embedder import embed_query
from app.llm_fallback import (
    NoLLMModelAvailableError,
    NonRetryableLLMError,
    create_stream_with_fallback,
)
from app.observability import flush_langfuse, langfuse_generation, langfuse_span
from app.pricing import calculate_cost_usd
from app.rag import (
    build_messages,
    count_prompt_tokens,
    count_text_tokens,
    get_llm_client,
)
from app.rate_limiter import check_rate_limit, record_token_usage
from app.security import (
    check_output_for_system_fragments,
    validate_user_input,
)
from app.semantic_cache import find_semantic_cache_hit, store_semantic_cache_entry
from app.usage_db import (
    create_request_id,
    get_usage_breakdown_last_hour,
    get_usage_today,
    init_usage_db,
    log_usage,
)
from app.vector_store import search_chunks


app = FastAPI(title="robot-dreams-hw")

LLM_MAX_CONCURRENT_STREAMS = 20
llm_semaphore = asyncio.Semaphore(LLM_MAX_CONCURRENT_STREAMS)

metrics = {
    "active_streams": 0,
    "aborted_streams": 0,
}


@app.on_event("startup")
def startup() -> None:
    init_usage_db()


@app.on_event("shutdown")
def shutdown() -> None:
    flush_langfuse()


class ChatRequest(BaseModel):
    message: str


def sse_data(payload: dict) -> str:
    """
    Format payload as Server-Sent Event.

    SSE format:
        data: {...}

    Each event must end with double newline.
    """
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def split_cached_response_for_streaming(text: str) -> list[str]:
    """
    Split cached full response into small pieces.

    This keeps UX similar to LLM token streaming.
    These are not real LLM tokens, but small text chunks.
    """
    return re.findall(r"\S+\s*", text)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "active_streams": metrics["active_streams"],
        "aborted_streams": metrics["aborted_streams"],
        "llm_max_concurrent_streams": LLM_MAX_CONCURRENT_STREAMS,
    }


@app.post("/chat/stream")
async def chat_stream(
    chat_request: ChatRequest,
    request: Request,
    tier_metadata: TierMetadata = Depends(require_api_key),
) -> StreamingResponse:
    """
    Q&A endpoint without chat history.

    Input:
        {"message": "..."}

    Output:
        text/event-stream with token events and one final done event.
    """
    request_id = create_request_id()
    request_started_at = time.perf_counter()

    trace_metadata = {
        "request_id": request_id,
        "api_key": tier_metadata["api_key"],
        "tier": tier_metadata["tier"],
        "cache_hit": None,
        "fallback_used": None,
        "model": None,
    }

    # Security validation must happen before rate limit and before streaming starts.
    with langfuse_span(
        name="security_input_validation",
        input_data={"message": chat_request.message},
        metadata=trace_metadata,
    ) as span:
        try:
            validate_user_input(
                message=chat_request.message,
                api_key=tier_metadata["api_key"],
            )

            if span:
                span.update(output={"allowed": True})

        except ValueError as exc:
            if span:
                span.update(
                    output={
                        "allowed": False,
                        "reason": str(exc),
                    }
                )

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

    # Auth already happened in require_api_key dependency.
    # We still create an observation so the trace shows the full pipeline.
    with langfuse_span(
        name="auth",
        input_data={"api_key": tier_metadata["api_key"]},
        metadata={
            **trace_metadata,
            "tier": tier_metadata["tier"],
        },
    ) as span:
        if span:
            span.update(
                output={
                    "authenticated": True,
                    "tier": tier_metadata["tier"],
                    "tokens_per_minute": tier_metadata["tokens_per_minute"],
                    "models": tier_metadata["models"],
                }
            )

    # Rate limit must also happen before StreamingResponse,
    # because once the stream starts we cannot return HTTP 429.
    with langfuse_span(
        name="rate_limit",
        input_data={
            "api_key": tier_metadata["api_key"],
            "limit_tokens_per_minute": tier_metadata["tokens_per_minute"],
        },
        metadata=trace_metadata,
    ) as span:
        rate_limit_result = await check_rate_limit(
            api_key=tier_metadata["api_key"],
            limit_tokens_per_minute=tier_metadata["tokens_per_minute"],
        )

        if span:
            span.update(
                output={
                    "allowed": rate_limit_result.allowed,
                    "used_tokens": rate_limit_result.used_tokens,
                    "limit_tokens": rate_limit_result.limit_tokens,
                    "retry_after_seconds": rate_limit_result.retry_after_seconds,
                }
            )

    if not rate_limit_result.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "message": "Token rate limit exceeded",
                "used_tokens": rate_limit_result.used_tokens,
                "limit_tokens": rate_limit_result.limit_tokens,
            },
            headers={
                "Retry-After": str(rate_limit_result.retry_after_seconds),
            },
        )

    async def event_generator() -> AsyncGenerator[str, None]:
        stream = None
        output_text_parts: list[str] = []
        aborted_reported = False
        llm_slot_acquired = False

        def mark_aborted() -> None:
            nonlocal aborted_reported

            if not aborted_reported:
                metrics["aborted_streams"] += 1
                aborted_reported = True

        try:
            with langfuse_span(
                name="chat_stream_pipeline",
                input_data={
                    "message": chat_request.message,
                },
                metadata=trace_metadata,
            ):
                # One embedding call per request.
                with langfuse_span(
                    name="embed_query",
                    input_data={"message": chat_request.message},
                    metadata=trace_metadata,
                ) as span:
                    query_embedding = embed_query(chat_request.message)

                    if span:
                        span.update(
                            output={
                                "embedding_dim": len(query_embedding),
                            }
                        )

                # 1. Semantic cache lookup.
                with langfuse_span(
                    name="semantic_cache_check",
                    input_data={"message": chat_request.message},
                    metadata=trace_metadata,
                ) as span:
                    cache_hit_result = find_semantic_cache_hit(query_embedding)

                    if span:
                        span.update(
                            output={
                                "cache_hit": cache_hit_result is not None,
                                "similarity": (
                                    cache_hit_result.score
                                    if cache_hit_result is not None
                                    else None
                                ),
                            }
                        )

                if cache_hit_result is not None:
                    selected_model = cache_hit_result.model or tier_metadata["models"][0]

                    usage = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    }

                    cost_usd = 0.0
                    fallback_used = False

                    first_token_sent = False
                    ttft_ms = 0

                    for token in split_cached_response_for_streaming(
                        cache_hit_result.response
                    ):
                        if await request.is_disconnected():
                            mark_aborted()
                            return

                        if not first_token_sent:
                            ttft_ms = int(
                                (time.perf_counter() - request_started_at) * 1000
                            )
                            first_token_sent = True

                        yield sse_data(
                            {
                                "type": "token",
                                "content": token,
                            }
                        )

                        await asyncio.sleep(0)

                    latency_ms = int(
                        (time.perf_counter() - request_started_at) * 1000
                    )

                    output_filtered = check_output_for_system_fragments(
                        response_text=cache_hit_result.response,
                        request_id=request_id,
                        api_key=tier_metadata["api_key"],
                        model=selected_model,
                    )

                    with langfuse_span(
                        name="stream_cached_response",
                        input_data={
                            "query": chat_request.message,
                            "cached_query": cache_hit_result.query,
                        },
                        metadata={
                            **trace_metadata,
                            "cache_hit": True,
                            "fallback_used": False,
                            "model": selected_model,
                        },
                    ) as span:
                        if span:
                            span.update(
                                output={
                                    "response": cache_hit_result.response,
                                    "similarity": cache_hit_result.score,
                                    "sources": cache_hit_result.sources,
                                    "latency_ms": latency_ms,
                                    "ttft_ms": ttft_ms,
                                    "output_filtered": output_filtered,
                                }
                            )

                    log_usage(
                        request_id=request_id,
                        api_key=tier_metadata["api_key"],
                        model=selected_model,
                        input_tokens=0,
                        output_tokens=0,
                        cost_usd=cost_usd,
                        latency_ms=latency_ms,
                        ttft_ms=ttft_ms,
                        cache_hit=True,
                        fallback_used=False,
                        output_filtered=output_filtered,
                    )

                    yield sse_data(
                        {
                            "type": "done",
                            "request_id": request_id,
                            "usage": usage,
                            "cost_usd": cost_usd,
                            "cache_hit": True,
                            "cache_similarity": cache_hit_result.score,
                            "fallback_used": fallback_used,
                            "sources": cache_hit_result.sources,
                            "tier": tier_metadata["tier"],
                            "model": selected_model,
                            "output_filtered": output_filtered,
                        }
                    )

                    return

                # 2. Cache MISS: use the same embedding for RAG retrieval.
                with langfuse_span(
                    name="vector_search",
                    input_data={
                        "query": chat_request.message,
                        "top_k": 3,
                    },
                    metadata={
                        **trace_metadata,
                        "cache_hit": False,
                    },
                ) as span:
                    chunks = search_chunks(
                        query_embedding=query_embedding,
                        top_k=3,
                    )

                    sources = [
                        chunk["chunk_id"]
                        for chunk in chunks
                        if chunk.get("chunk_id")
                    ]

                    if span:
                        span.update(
                            output={
                                "sources": sources,
                                "scores": [
                                    {
                                        "chunk_id": chunk.get("chunk_id"),
                                        "score": chunk.get("score"),
                                    }
                                    for chunk in chunks
                                ],
                            }
                        )

                # 3. Build prompt.
                with langfuse_span(
                    name="build_prompt",
                    input_data={
                        "user_query": chat_request.message,
                        "sources": sources,
                    },
                    metadata={
                        **trace_metadata,
                        "cache_hit": False,
                    },
                ) as span:
                    messages = build_messages(
                        message=chat_request.message,
                        chunks=chunks,
                    )

                    input_tokens = count_prompt_tokens(messages)

                    if span:
                        span.update(
                            output={
                                "messages": messages,
                                "input_tokens": input_tokens,
                            }
                        )

                client = get_llm_client()
                generation_observation = None

                # 4. LLM call with fallback chain + streaming.
                with langfuse_generation(
                    name="llm_generation",
                    model=tier_metadata["models"][0],
                    input_data=messages,
                    metadata={
                        **trace_metadata,
                        "cache_hit": False,
                        "fallback_used": None,
                        "model_chain": tier_metadata["models"],
                    },
                ) as generation:
                    generation_observation = generation

                    await llm_semaphore.acquire()
                    llm_slot_acquired = True
                    metrics["active_streams"] += 1

                    selection = await create_stream_with_fallback(
                        client=client,
                        model_chain=tier_metadata["models"],
                        messages=messages,
                    )

                    stream = selection.stream
                    selected_model = selection.model
                    fallback_used = selection.fallback_used

                    if generation_observation:
                        generation_observation.update(
                            model=selected_model,
                            metadata={
                                **trace_metadata,
                                "tier": tier_metadata["tier"],
                                "cache_hit": False,
                                "fallback_used": fallback_used,
                                "selected_model": selected_model,
                                "model_chain": tier_metadata["models"],
                            },
                        )

                    first_token_sent = False
                    ttft_ms = 0

                    async for event in stream:
                        if await request.is_disconnected():
                            mark_aborted()

                            if stream is not None:
                                await stream.close()

                            raise asyncio.CancelledError()

                        delta = event.choices[0].delta
                        token = delta.content if delta and delta.content else ""

                        if not token:
                            continue

                        if not first_token_sent:
                            ttft_ms = int(
                                (time.perf_counter() - request_started_at) * 1000
                            )
                            first_token_sent = True

                        output_text_parts.append(token)

                        yield sse_data(
                            {
                                "type": "token",
                                "content": token,
                            }
                        )

                    full_output_text = "".join(output_text_parts)

                    output_filtered = check_output_for_system_fragments(
                        response_text=full_output_text,
                        request_id=request_id,
                        api_key=tier_metadata["api_key"],
                        model=selected_model,
                    )

                    output_tokens = count_text_tokens(full_output_text)
                    total_tokens = input_tokens + output_tokens

                    usage = {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                    }

                    cost_usd = calculate_cost_usd(
                        model=selected_model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )

                    latency_ms = int(
                        (time.perf_counter() - request_started_at) * 1000
                    )

                    if generation_observation:
                        generation_observation.update(
                            output=full_output_text,
                            usage_details={
                                "input": input_tokens,
                                "output": output_tokens,
                                "total": total_tokens,
                            },
                            cost_details={
                                "total": cost_usd,
                            },
                            metadata={
                                **trace_metadata,
                                "tier": tier_metadata["tier"],
                                "cache_hit": False,
                                "fallback_used": fallback_used,
                                "selected_model": selected_model,
                                "sources": sources,
                                "latency_ms": latency_ms,
                                "ttft_ms": ttft_ms,
                                "output_filtered": output_filtered,
                            },
                        )

                await record_token_usage(
                    api_key=tier_metadata["api_key"],
                    tokens_used=total_tokens,
                )

                store_semantic_cache_entry(
                    query_embedding=query_embedding,
                    query=chat_request.message,
                    response=full_output_text,
                    model=selected_model,
                    sources=sources,
                    usage=usage,
                )

                log_usage(
                    request_id=request_id,
                    api_key=tier_metadata["api_key"],
                    model=selected_model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    latency_ms=latency_ms,
                    ttft_ms=ttft_ms,
                    cache_hit=False,
                    fallback_used=fallback_used,
                    output_filtered=output_filtered,
                )

                yield sse_data(
                    {
                        "type": "done",
                        "request_id": request_id,
                        "usage": usage,
                        "cost_usd": cost_usd,
                        "cache_hit": False,
                        "cache_similarity": None,
                        "fallback_used": fallback_used,
                        "sources": sources,
                        "tier": tier_metadata["tier"],
                        "model": selected_model,
                        "output_filtered": output_filtered,
                    }
                )

        except asyncio.CancelledError:
            mark_aborted()

            if stream is not None:
                try:
                    await stream.close()
                except Exception:
                    pass

            raise

        except NonRetryableLLMError as exc:
            yield sse_data(
                {
                    "type": "error",
                    "message": str(exc),
                    "retryable": False,
                }
            )

        except NoLLMModelAvailableError as exc:
            yield sse_data(
                {
                    "type": "error",
                    "message": str(exc),
                    "retryable": True,
                }
            )

        except Exception as exc:
            yield sse_data(
                {
                    "type": "error",
                    "message": str(exc),
                }
            )

        finally:
            if stream is not None:
                try:
                    await stream.close()
                except Exception:
                    pass

            if llm_slot_acquired:
                metrics["active_streams"] = max(
                    0,
                    metrics["active_streams"] - 1,
                )
                llm_semaphore.release()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@app.get("/usage/today")
async def usage_today(
    tier_metadata: TierMetadata = Depends(require_api_key),
) -> dict:
    return get_usage_today(api_key=tier_metadata["api_key"])


@app.get("/usage/breakdown")
async def usage_breakdown(
    tier_metadata: TierMetadata = Depends(require_api_key),
) -> dict:
    breakdown = get_usage_breakdown_last_hour(api_key=tier_metadata["api_key"])

    return {
        **breakdown,
        "aborted_streams": metrics["aborted_streams"],
    }