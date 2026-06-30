# Project Report: Ukrainian Literary Whining Generator

*AI Engineering Course — Final Project*

---

## 1. Problem Statement

Ukrainian users often want to express frustration, sadness, or complaint in a creative and humorous way. This project builds a system that transforms a modern, colloquial Ukrainian complaint into a sentence or paragraph written in the style of Ukrainian classical literature — think Shevchenko, Franko, Lesia Ukrainka.

The system targets a **style transfer** task: the emotional content and meaning are preserved, but the vocabulary, syntax, and tone shift from casual 2026 Ukrainian to 19th-century literary Ukrainian.

---

## 2. Why This Project Matters

- **Language preservation**: Classical Ukrainian is an endangered register. Making it accessible and fun promotes cultural literacy.
- **Technical depth**: The project demonstrates the full ML engineering lifecycle — data collection, fine-tuning, API design, containerized deployment, and monitoring.
- **Product design**: The constraint of working with low-resource, culturally specific data forces creative data engineering decisions.
- **Practical challenge**: Ukrainian NLP is underserved compared to English. This project contributes a reusable synthetic dataset for style transfer.

---

## 3. Dataset Research

### Primary Corpus

**PLuG (Pluperfect GRAC)** — a large Ukrainian text corpus hosted at Dandelliony/pluperfect_grac. Contains ~98,000 .txt files including classical Ukrainian literature.

### Data Pipeline

1. **Extraction**: Sentence splitting with regex + sadness lexicon scoring (~40 Ukrainian emotional words). Filter by score >= 2 and length >= 40 characters.
2. **Modernization**: OpenAI Batch API (GPT-4o-mini) generates 5 modern Ukrainian variants per classical sentence.
3. **Quality Filtering**: Remove empty, too-similar, non-Ukrainian, or classically-contaminated modern inputs. Deduplicate.
4. **SFT Format**: Convert to instruction-following chat format for fine-tuning.

### Dataset Stats (Pilot Run)

| Stage | Count |
|-------|-------|
| Files sampled | 100 |
| Sad sentences extracted | 20 |
| Training pairs generated (pilot) | 25 |
| After quality filter | ~20 |

Full run (all files, no sample limit) is expected to yield 2,000-5,000 quality pairs.

---

## 4. Model Choice

### Evaluated Candidates

| Model | Ukrainian Quality | Fine-Tuning Feasibility | Notes |
|-------|------------------|------------------------|-------|
| Qwen3-8B-Instruct | Good | Excellent (MLX, LoRA) | Best overall fit |
| Llama 3.1 8B | Moderate | Good | Less Ukrainian-specific |
| Mistral 7B | Moderate | Good | Works on limited hardware |

**Decision: Qwen3-8B-Instruct** — best Ukrainian fluency, MLX-compatible for Apple Silicon fine-tuning, reasonable inference cost.

---

## 5. Fine-Tuning Approach

**Method**: LoRA (Low-Rank Adaptation) via MLX-LM on Apple Silicon.

**Key hyperparameters**:

- base_model: mlx-community/Qwen3-8B-4bit
- lora_rank: 16, lora_alpha: 32, lora_dropout: 0.05
- lora_layers: 8
- learning_rate: 2e-5, batch_size: 2, epochs: 3
- max_seq_length: 512, mask_prompt: true

**Prompt format** (SFT chat):

```
System: (short style instruction)
User: Перепиши сучасний текст у стилі класичної літератури. Текст: {modern}
Assistant: {classical}
```

**Experiment tracking**: MLflow logs all hyperparameters, train/eval loss, and adapter artifacts.

---

## 6. System Architecture

### Components

| Component | Technology | Responsibility |
|-----------|-----------|----------------|
| API | FastAPI + Uvicorn | Validation, rate limiting, DB logging, inference dispatch |
| UI | Streamlit | User-facing form and result display |
| Database | PostgreSQL + SQLAlchemy | Log every generation |
| Cache | Redis | Rate limiting + response cache (TTL 24h) |
| Inference | Mock / OpenAI / Local | Text generation |
| Monitoring | Prometheus + Grafana | Request metrics, latency, error rates |
| Experiments | MLflow | Fine-tuning parameter tracking |

### Request Flow

1. User submits text in Streamlit UI
2. Anonymous session_id generated in browser
3. UI calls `POST /generate` on FastAPI backend
4. Backend checks Redis rate limit
5. Backend runs safety check (self-harm detection)
6. Backend checks Redis cache for identical request
7. If cache miss, calls inference client (mock/OpenAI/local)
8. Saves generation to PostgreSQL if `STORE_USER_INPUTS=true`
9. Returns result; updates Prometheus metrics
10. UI renders classical-style output

---

## 7. Deployment Architecture

### MVP (Docker Compose)

All services run with `docker compose up`:

- api, ui, postgres, redis, mlflow, prometheus, grafana

### Production-Like (Kubernetes)

Kubernetes manifests in `deploy/k8s/`:

- `api-deployment.yaml` — 2 replicas
- `ui-deployment.yaml` — 1 replica
- `hpa.yaml` — scale to 5 replicas at 70% CPU
- `ingress.yaml` — nginx ingress
- `configmap.yaml` + `secret.example.yaml`

Capacity estimate for 500 users/day, 1-5 requests each (500-2500 requests/day = ~2 requests/minute average, peak ~10 req/min): 2 replicas + HPA should be sufficient.

---

## 8. Evaluation Results

Automatic metrics on mock inference (baseline):

| Metric | Value |
|--------|-------|
| classical_marker_score | 4.2 avg |
| modern_marker_leakage | 0.0 |
| russian_marker_count | 0.0 |
| ukrainian_char_ratio | 0.085 |
| no_must_not | 100% |

Note: Mock inference always returns the same predefined responses, so these metrics reflect the predefined responses quality, not a trained model. Real evaluation requires running `make eval` with `INFERENCE_MODE=openai` or `local`.

Human evaluation template is at `reports/human_eval_template.csv`.

---

## 9. Trade-offs

| Decision | Alternative | Why chosen |
|----------|-------------|-----------|
| Streamlit UI | React + Vite | Faster to build, sufficient for MVP |
| Redis for rate limit | In-memory / DB | Persistent, works across replicas |
| OpenAI Batch API for modernization | Manual annotation | Scales to thousands of pairs at low cost |
| MLX-LM for fine-tuning | Google Colab + PEFT | Native Apple Silicon, no cloud dependency |
| Mock inference default | Require model | Allows dev/test without GPU |
| SQLite for tests | PostgreSQL | No Docker dependency in CI |

---

## 10. What Would Be Improved in Production

1. **Model quality**: Run full corpus extraction (>100 files), use GPT-5.5 for higher-quality modernization, add human review loop.
2. **Evaluation**: LLM-as-judge scoring for style fidelity and meaning preservation. A/B testing between model versions.
3. **Auth and abuse**: Add basic abuse detection, optional login for persistent history.
4. **Ukrainian NLP**: Use proper Ukrainian tokenizer (Stanza) for sentence splitting instead of regex.
5. **Fine-tuning**: Experiment with higher rank (r=32) and more epochs; try DPO to align on human preference data.
6. **Deployment**: Add proper secrets management (Vault or k8s Secrets), TLS, CDN for Streamlit.
7. **Observability**: Distributed tracing (Jaeger), structured log ingestion (Loki), alert rules in Grafana.
8. **Dataset versioning**: DVC or Hugging Face Datasets for reproducible data lineage.
