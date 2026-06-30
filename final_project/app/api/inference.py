"""Inference: mock / openai / local (MLX) / remote (AWS GPU service)."""

import os
import re
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

INFERENCE_MODE = os.getenv("INFERENCE_MODE", "mock")
FALLBACK_MESSAGE = os.getenv(
    "FALLBACK_MESSAGE",
    "Ой лихо, моделі розгубила",
)

_MOCK = "Тяжко мені, тяжко на світі жити — мов хмара темна лягла на душу мою."

_SYSTEM = (
    "Ти перетворюєш сучасні українські скарги на текст у стилі класичної "
    "української літератури. Повертай лише перетворений текст. Не пояснюй. "
    "Не пиши по-російськи."
)

_PROMPT = (
    "Перепиши сучасний текст у стилі класичної української літератури.\n\n"
    "Текст: {text}"
)

_model = None
_tokenizer = None
_backend_checked = False
_backend_available = False


class InferenceResult:
    __slots__ = ("text", "model_version", "is_fallback")

    def __init__(self, text: str, model_version: str, is_fallback: bool = False):
        self.text = text
        self.model_version = model_version
        self.is_fallback = is_fallback


def generate(text: str) -> InferenceResult:
    mode = os.getenv("INFERENCE_MODE", INFERENCE_MODE)
    model_version = os.getenv("MODEL_VERSION", "qwen3-8b-lora-v2")

    try:
        if mode == "mock":
            time.sleep(0.05)
            return InferenceResult(_MOCK, "mock")

        if mode == "openai":
            return InferenceResult(_openai(text), model_version)

        if mode == "local":
            return InferenceResult(_local(text), model_version)

        if mode == "remote":
            return _remote(text, model_version)

        return InferenceResult(_MOCK, "mock")

    except Exception as exc:
        print(f"[inference] error ({mode}): {exc}")
        return InferenceResult(FALLBACK_MESSAGE, "fallback", is_fallback=True)


def check_backend() -> bool:
    """Return True if the configured inference backend looks healthy."""
    global _backend_checked, _backend_available
    mode = os.getenv("INFERENCE_MODE", INFERENCE_MODE)

    if mode == "mock":
        return True

    if mode == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))

    if mode == "local":
        adapter = Path(os.getenv("LORA_ADAPTER_PATH", "models/adapters/qwen3-8b-lora-v2"))
        return (adapter / "adapters.safetensors").exists()

    if mode == "remote":
        url = os.getenv("INFERENCE_SERVICE_URL", "http://inference:8080")
        try:
            r = httpx.get(f"{url.rstrip('/')}/health", timeout=5.0)
            _backend_available = r.status_code == 200
        except Exception:
            _backend_available = False
        _backend_checked = True
        return _backend_available

    return False


def _openai(text: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _PROMPT.format(text=text)},
        ],
        temperature=0.8,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()


def _local(text: str) -> str:
    global _model, _tokenizer
    from mlx_lm import load, generate as mlx_generate

    if _model is None:
        adapter_path = os.getenv("LORA_ADAPTER_PATH", "models/adapters/qwen3-8b-lora-v2")
        base_model = os.getenv("BASE_MODEL", "mlx-community/Qwen3-8B-4bit")
        if not Path(adapter_path, "adapters.safetensors").exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")
        print(f"[inference] Loading {base_model} + {adapter_path}")
        _model, _tokenizer = load(base_model, adapter_path=adapter_path)

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _PROMPT.format(text=text)},
    ]
    formatted = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    result = mlx_generate(_model, _tokenizer, prompt=formatted, max_tokens=512, verbose=False)
    return re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()


def _remote(text: str, model_version: str) -> InferenceResult:
    url = os.getenv("INFERENCE_SERVICE_URL", "http://inference:8080").rstrip("/")
    timeout = float(os.getenv("INFERENCE_TIMEOUT", "120"))

    try:
        resp = httpx.post(
            f"{url}/generate",
            json={"text": text},
            timeout=timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"inference service returned {resp.status_code}")

        data = resp.json()
        output = data.get("output_text", "").strip()
        if not output:
            raise RuntimeError("empty response from inference service")

        return InferenceResult(
            output,
            data.get("model_version", model_version),
            is_fallback=data.get("is_fallback", False),
        )
    except Exception as exc:
        print(f"[inference] remote failed: {exc}")
        return InferenceResult(FALLBACK_MESSAGE, "fallback", is_fallback=True)
