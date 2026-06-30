import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4
from datetime import datetime, timezone, timedelta

from app.config import settings


def get_connection() -> sqlite3.Connection:
    db_path = Path(settings.usage_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_usage_db() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_usage (
                request_id TEXT PRIMARY KEY,
                api_key TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cost_usd REAL NOT NULL,
                latency_ms INTEGER NOT NULL,
                ttft_ms INTEGER NOT NULL,
                cache_hit INTEGER NOT NULL,
                fallback_used INTEGER NOT NULL,
                output_filtered INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )

        # Migration for existing SQLite DBs created before output_filtered existed.
        columns = connection.execute("PRAGMA table_info(llm_usage)").fetchall()
        column_names = {column["name"] for column in columns}

        if "output_filtered" not in column_names:
            connection.execute(
                """
                ALTER TABLE llm_usage
                ADD COLUMN output_filtered INTEGER NOT NULL DEFAULT 0
                """
            )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_llm_usage_api_key_created_at
            ON llm_usage (api_key, created_at)
            """
        )


def create_request_id() -> str:
    return str(uuid4())


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_usage(
    request_id: str,
    api_key: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    output_filtered: bool,
    cost_usd: float,
    latency_ms: int,
    ttft_ms: int,
    cache_hit: bool,
    fallback_used: bool,
) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO llm_usage (
                request_id,
                api_key,
                model,
                input_tokens,
                output_tokens,
                output_filtered,
                cost_usd,
                latency_ms,
                ttft_ms,
                cache_hit,
                fallback_used,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                api_key,
                model,
                input_tokens,
                output_tokens,
                int(output_filtered),
                cost_usd,
                latency_ms,
                ttft_ms,
                int(cache_hit),
                int(fallback_used),
                utc_now_iso(),
            ),
        )


def get_usage_today(api_key: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    start = datetime(
        year=now.year,
        month=now.month,
        day=now.day,
        tzinfo=timezone.utc,
    )
    end = start + timedelta(days=1)

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS requests,
                COALESCE(SUM(input_tokens + output_tokens), 0) AS tokens,
                COALESCE(SUM(cost_usd), 0.0) AS cost_usd
            FROM llm_usage
            WHERE api_key = ?
              AND created_at >= ?
              AND created_at < ?
            """,
            (api_key, start.isoformat(), end.isoformat()),
        ).fetchone()

    return {
        "requests": int(row["requests"]),
        "tokens": int(row["tokens"]),
        "cost_usd": round(float(row["cost_usd"]), 8),
    }


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0

    values = sorted(values)
    index = int(round((percentile / 100) * (len(values) - 1)))
    return float(values[index])


def get_usage_breakdown_last_hour(api_key: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=1)
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM llm_usage
            WHERE api_key = ?
              AND created_at >= ?
              AND created_at <= ?
            """,
            (api_key, start.isoformat(), now.isoformat()),
        ).fetchall()

    requests = len(rows)
    output_filtered_count = sum(1 for row in rows if int(row["output_filtered"]) == 1)


    total_tokens = sum(int(row["input_tokens"]) + int(row["output_tokens"]) for row in rows)
    total_cost = sum(float(row["cost_usd"]) for row in rows)

    cache_hits = sum(1 for row in rows if int(row["cache_hit"]) == 1)
    fallback_count = sum(1 for row in rows if int(row["fallback_used"]) == 1)
    output_filtered_count = sum(1 for row in rows if int(row["output_filtered"]) == 1)

    latencies = [int(row["latency_ms"]) for row in rows]

    by_model: dict[str, dict[str, Any]] = {}

    for row in rows:
        model = str(row["model"])

        if model not in by_model:
            by_model[model] = {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "tokens": 0,
                "cost_usd": 0.0,
            }

        by_model[model]["requests"] += 1
        by_model[model]["input_tokens"] += int(row["input_tokens"])
        by_model[model]["output_tokens"] += int(row["output_tokens"])
        by_model[model]["tokens"] += int(row["input_tokens"]) + int(row["output_tokens"])
        by_model[model]["cost_usd"] += float(row["cost_usd"])

    for model_stats in by_model.values():
        model_stats["cost_usd"] = round(model_stats["cost_usd"], 8)

    return {
        "window_seconds": 3600,
        "requests": requests,
        "tokens": total_tokens,
        "cost_usd": round(total_cost, 8),
        "output_filtered_rate": output_filtered_count / requests if requests else 0.0,
        "by_model": by_model,
        "cache_hit_rate": cache_hits / requests if requests else 0.0,
        "fallback_rate": fallback_count / requests if requests else 0.0,
        "avg_latency_ms": round(sum(latencies) / requests, 2) if requests else 0.0,
        "p95_latency_ms": _percentile(latencies, 95),
    }