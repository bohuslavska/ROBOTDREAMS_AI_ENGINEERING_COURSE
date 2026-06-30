# # Document RAG Q&A Bot

Public deployment:

https://robot-dreams-hw.fly.dev

This project implements a production-style RAG service over a selected document. The service supports document retrieval, LLM streaming, API key authentication, token-based rate limiting, semantic cache, fallback models, usage/cost tracking, and Langfuse observability.

## Document choice

For the knowledge base, I used an LLM security document related to OWASP Top 10 for LLM Applications. This document was selected because it contains practical and structured information about prompt injection, vector and embedding weaknesses, supply chain risks, unbounded consumption, and other LLM application security issues.

The PDF document was converted into plain text/Markdown, then split into overlapping chunks. These chunks were embedded and stored in Qdrant as a vector database.

## Implementation path

The implementation was done in several stages:

1. **Document preparation**
   The source PDF was converted into `data/source.md`.

2. **Chunking**
   The document was split into chunks of approximately 500 tokens with 50-token overlap.

3. **Embeddings**
   Each chunk was embedded using `sentence-transformers/all-MiniLM-L6-v2`.

4. **Vector database**
   Embeddings and chunk payloads were stored in Qdrant. In production, Qdrant Cloud is used.

5. **RAG pipeline**
   For each user question, the app creates a query embedding, retrieves the most relevant chunks from Qdrant, builds a prompt with document context, and sends it to the selected LLM.

6. **Streaming API**
   The main endpoint returns Server-Sent Events, so the answer is streamed token by token.

7. **Authentication and rate limiting**
   API keys are divided into tiers. Redis is used to track token usage per API key and enforce token-per-minute limits.

8. **Semantic cache**
   Similar questions can reuse previous answers. Cached responses are stored in a separate Qdrant collection with expiration metadata.

9. **Fallback models**
   If the first model fails, the app tries the next model in the tier’s model chain.

10. **Usage and cost tracking**
    Requests, token usage, latency, fallback usage, cache hits, and estimated costs are stored in SQLite.

11. **Observability**
    Langfuse tracing was added for the full pipeline: security validation, auth, rate limit, embedding, semantic cache, vector search, prompt building, and LLM generation.

12. **Deployment**
    The FastAPI app was containerized with Docker and deployed to Fly.io. External services are used for production dependencies:

* Qdrant Cloud for vector search
* Upstash Redis for rate limiting
* OpenRouter for LLM calls
* Langfuse Cloud for observability
* Fly.io volume for SQLite usage storage

## Architecture

```text
User
  ↓
FastAPI /chat/stream
  ↓
API key auth
  ↓
Redis rate limit
  ↓
Query embedding
  ↓
Semantic cache check
  ↓
Qdrant vector retrieval
  ↓
Prompt construction
  ↓
OpenRouter LLM call with fallback
  ↓
SSE token streaming
  ↓
Usage logging + Langfuse tracing
```

## Public API

### Health check

```bash
curl "https://robot-dreams-hw.fly.dev/health"
```

### Streaming RAG request

```bash
curl -N -X POST "https://robot-dreams-hw.fly.dev/chat/stream" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-pro-key" \
  -d '{"message":"What is Retrieval Augmented Generation according to the document?"}'
```

Expected output includes streamed token events and a final `done` event:

```json
{
  "type": "done",
  "usage": {
    "input_tokens": 1187,
    "output_tokens": 56,
    "total_tokens": 1243
  },
  "cost_usd": 0.00021165,
  "cache_hit": false,
  "fallback_used": false,
  "sources": ["source::chunk_0059", "source::chunk_0023", "source::chunk_0019"],
  "tier": "demo-pro",
  "model": "openai/gpt-4o-mini"
}
```

### Usage endpoint

```bash
curl -H "X-API-Key: demo-pro-key" \
  "https://robot-dreams-hw.fly.dev/usage/today"
```

### Usage breakdown

```bash
curl -H "X-API-Key: demo-pro-key" \
  "https://robot-dreams-hw.fly.dev/usage/breakdown"
```

## Environment variables

All required environment variables are listed in `.env.example`.

Real secrets are not stored in GitHub. For local development, they should be placed in `.env`. For production deployment, they are configured as Fly.io Secrets.

Required services:

```text
OPENROUTER_API_KEY
QDRANT_URL
QDRANT_API_KEY
REDIS_URL
LANGFUSE_SECRET_KEY
LANGFUSE_PUBLIC_KEY
LANGFUSE_BASE_URL
```

For Redis, the app uses the Redis protocol URL, not the Upstash REST URL:

```text
REDIS_URL=rediss://default:YOUR_PASSWORD@YOUR_HOST.upstash.io:6379
```

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Health check locally:

```bash
curl "http://localhost:8000/health"
```

## Deployment

The app is deployed on Fly.io from the `Homework5` project folder.

Production URL:

```text
https://robot-dreams-hw.fly.dev
```

The deployment uses:

```text
Dockerfile
fly.toml
Fly.io Secrets
Fly.io Volume for SQLite
Qdrant Cloud
Upstash Redis
Langfuse Cloud
OpenRouter
```

## Demonstrated features

The submitted screenshots demonstrate:

* Streaming response in terminal with `curl -N`
* RAG response with `sources` in the final `done` event
* Semantic cache hit on a similar request
* Rate limit returning `429 Too Many Requests` with `Retry-After`
* Fallback model execution with `fallback_used=true`
* `/usage/today` with token usage and estimated cost
* Langfuse traces for the RAG pipeline
