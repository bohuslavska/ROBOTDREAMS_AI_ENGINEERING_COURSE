"""FastAPI application — Ukrainian Literary Whining Generator."""

import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from app.api.db import create_tables, SessionLocal, check_db
from app.api.inference import check_backend, generate
from app.api.metrics import (
    GENERATION_LATENCY,
    GENERATIONS_TOTAL,
    MODEL_AVAILABLE,
    REQUESTS_TOTAL,
    metrics_response,
)
from app.api.models import Generation
from app.api.schemas import GenerateRequest, GenerateResponse, HealthResponse, ReadyResponse

INFERENCE_MODE = os.getenv("INFERENCE_MODE", "mock")
MODEL_VERSION = os.getenv("MODEL_VERSION", "qwen3-8b-lora-v2")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        create_tables()
    except Exception as e:
        print(f"[startup] DB init warning: {e}")
    MODEL_AVAILABLE.set(1 if check_backend() else 0)
    yield


app = FastAPI(
    title="Ukrainian Literary Whining Generator",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def track_requests(request: Request, call_next):
    response = await call_next(request)
    if request.url.path != "/metrics":
        REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
        ).inc()
    return response


@app.get("/health", response_model=HealthResponse)
def health():
    available = check_backend()
    MODEL_AVAILABLE.set(1 if available else 0)
    return HealthResponse(
        status="ok",
        inference_mode=INFERENCE_MODE,
        model_available=available,
    )


@app.get("/ready", response_model=ReadyResponse)
def ready():
    db_ok = check_db()
    model_ok = check_backend()
    MODEL_AVAILABLE.set(1 if model_ok else 0)
    status = "ok" if db_ok else "degraded"
    return ReadyResponse(status=status, model_available=model_ok)


@app.get("/metrics")
def metrics():
    body, content_type = metrics_response()
    return Response(content=body, media_type=content_type)


@app.get("/examples")
def examples():
    return [
        {
            "input": "У мене був жахливий день на роботі, я втомилась.",
            "output": "Тяжко мені стало серед людей і праці марної, мов серце моє знемогло під вагою днів безрадісних.",
        },
        {
            "input": "Він мене кинув і навіть не пояснив чому.",
            "output": "Пішов він геть, мов вітер перекотиполе, і не залишив по собі нічого, крім пустки в грудях.",
        },
        {
            "input": "Я одна, і ніхто мене не розуміє.",
            "output": "Самотньо мені на цьому світі, мов зірка на небі осінньому — видно всім, та нікому не потрібна.",
        },
    ]


@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(body: GenerateRequest):
    if any(m in body.text.lower() for m in ["хочу померти", "покінчити з собою", "не хочу жити"]):
        return GenerateResponse(
            input_text=body.text,
            output_text=(
                "Я чую, що тобі зараз важко. Зверніться до близьких або зателефонуйте "
                "на лінію психологічної допомоги: 0-800-100-102 (безкоштовно, цілодобово)."
            ),
            model_version="safety-filter",
            latency_ms=0,
        )

    t0 = time.perf_counter()
    result = generate(body.text)
    latency = int((time.perf_counter() - t0) * 1000)

    GENERATIONS_TOTAL.labels(
        model_version=result.model_version,
        fallback=str(result.is_fallback).lower(),
    ).inc()
    GENERATION_LATENCY.observe(latency / 1000)

    _log_generation(
        input_text=body.text,
        output_text=result.text,
        model_version=result.model_version,
        latency_ms=latency,
        status="fallback" if result.is_fallback else "success",
    )

    return GenerateResponse(
        input_text=body.text,
        output_text=result.text,
        model_version=result.model_version,
        latency_ms=latency,
        is_fallback=result.is_fallback,
    )


def _log_generation(
    input_text: str,
    output_text: str,
    model_version: str,
    latency_ms: int,
    status: str = "success",
):
    if os.getenv("STORE_USER_INPUTS", "true").lower() != "true":
        return
    try:
        db: Session = SessionLocal()
        db.add(Generation(
            id=str(uuid.uuid4()),
            input_text=input_text,
            output_text=output_text,
            model_version=model_version,
            latency_ms=latency_ms,
            status=status,
        ))
        db.commit()
    except Exception as e:
        print(f"[db] logging failed: {e}")
    finally:
        db.close()
