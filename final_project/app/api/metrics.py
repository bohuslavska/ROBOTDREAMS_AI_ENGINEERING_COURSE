"""Prometheus metrics for the API."""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

REQUESTS_TOTAL = Counter(
    "whiner_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

GENERATIONS_TOTAL = Counter(
    "whiner_generations_total",
    "Total /generate calls",
    ["model_version", "fallback"],
)

GENERATION_LATENCY = Histogram(
    "whiner_generation_latency_seconds",
    "Latency of /generate in seconds",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60),
)

MODEL_AVAILABLE = Gauge(
    "whiner_model_available",
    "1 if inference backend is reachable, else 0",
)


def metrics_response():
    return generate_latest(), CONTENT_TYPE_LATEST
