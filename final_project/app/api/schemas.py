"""Pydantic schemas."""

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=1000)


class GenerateResponse(BaseModel):
    input_text: str
    output_text: str
    model_version: str
    latency_ms: int
    is_fallback: bool = False


class HealthResponse(BaseModel):
    status: str
    inference_mode: str
    model_available: bool


class ReadyResponse(BaseModel):
    status: str
    model_available: bool
