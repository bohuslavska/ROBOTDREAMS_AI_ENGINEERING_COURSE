# Document RAG Q&A Bot

FastAPI RAG service with:

- Qdrant vector search
- OpenRouter LLM streaming
- SSE endpoint
- API key auth
- Redis token-based rate limiting
- Semantic cache
- Cost tracking
- Langfuse observability

## Deploy (Fly.io)

```bash
# One-time setup
fly volumes create rag_data --region fra --size 1

fly secrets set \
  OPENROUTER_API_KEY="..." \
  QDRANT_URL="..." \
  QDRANT_API_KEY="..." \
  REDIS_URL="rediss://..." \
  LANGFUSE_SECRET_KEY="..." \
  LANGFUSE_PUBLIC_KEY="..."

fly deploy
```

## Public API

After deployment:

```bash
curl -N -X POST "https://robot_dreams_hw.fly.dev/chat/stream" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-free-key" \
  -d '{"message":"What is prompt injection?"}'

