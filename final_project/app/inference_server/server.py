"""GPU/CPU inference microservice for AWS deployment."""

import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

FALLBACK_MESSAGE = os.getenv("FALLBACK_MESSAGE", "Ой лихо, моделі розгубила")
MODEL_VERSION = os.getenv("MODEL_VERSION", "qwen3-8b-lora-v2")
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-8B")
ADAPTER_PATH = os.getenv("LORA_ADAPTER_PATH", "models/adapters/qwen3-8b-lora-v2")
USE_MLX = os.getenv("USE_MLX", "false").lower() == "true"

_SYSTEM = (
    "Ти перетворюєш сучасні українські скарги на текст у стилі класичної "
    "української літератури. Повертай лише перетворений текст. Не пояснюй. "
    "Не пиши по-російськи."
)
_INSTRUCTION = "Перепиши сучасний текст у стилі класичної української літератури."

_model = None
_tokenizer = None
_backend = None  # "mlx" | "transformers" | None


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=1000)


class GenerateResponse(BaseModel):
    output_text: str
    model_version: str
    latency_ms: int
    is_fallback: bool = False


def _load_model():
    global _model, _tokenizer, _backend

    adapter = Path(ADAPTER_PATH)
    if USE_MLX:
        from mlx_lm import load

        base = os.getenv("MLX_BASE_MODEL", "mlx-community/Qwen3-8B-4bit")
        _model, _tokenizer = load(base, adapter_path=str(adapter))
        _backend = "mlx"
        print(f"[inference-server] MLX loaded: {base} + {adapter}")
        return

    # HuggingFace + PEFT (AWS GPU path)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    if (adapter / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base_model, str(adapter))
        print(f"[inference-server] HF+PEFT loaded: {BASE_MODEL} + {adapter}")
    else:
        model = base_model
        print(f"[inference-server] HF base only (no adapter): {BASE_MODEL}")

    if device == "cpu":
        model = model.to(device)

    _model = model
    _tokenizer = tokenizer
    _backend = "transformers"


def _generate(text: str) -> str:
    prompt_user = f"{_INSTRUCTION}\n\nТекст: {text}"

    if _backend == "mlx":
        from mlx_lm import generate as mlx_generate

        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt_user},
        ]
        formatted = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        raw = mlx_generate(_model, _tokenizer, prompt=formatted, max_tokens=512, verbose=False)
        return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    import torch

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": prompt_user},
    ]
    formatted = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(formatted, return_tensors="pt").to(_model.device)
    with torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.8)
    decoded = _tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded.strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_model()
    except Exception as exc:
        print(f"[inference-server] model load failed: {exc}")
    yield


app = FastAPI(title="Whiner Inference Service", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok" if _backend else "degraded",
        "backend": _backend,
        "model_version": MODEL_VERSION,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(body: GenerateRequest):
    t0 = time.perf_counter()

    if _backend is None:
        return GenerateResponse(
            output_text=FALLBACK_MESSAGE,
            model_version="fallback",
            latency_ms=int((time.perf_counter() - t0) * 1000),
            is_fallback=True,
        )

    try:
        output = _generate(body.text)
        if not output:
            raise ValueError("empty generation")
        return GenerateResponse(
            output_text=output,
            model_version=MODEL_VERSION,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
    except Exception as exc:
        print(f"[inference-server] generation failed: {exc}")
        return GenerateResponse(
            output_text=FALLBACK_MESSAGE,
            model_version="fallback",
            latency_ms=int((time.perf_counter() - t0) * 1000),
            is_fallback=True,
        )
