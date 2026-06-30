"""
Lesson 06 — LLM Engineering Homework
Extraction Agent: витягує структуровані дані з транскриптів зустрічей.
Підтримує два провайдери: Ollama (self-hosted) та OpenAI (cloud).
"""

import json
import sys
import time
import re
import os
from pathlib import Path
from dotenv import load_dotenv

import requests
import openai

# ── Завантаження змінних середовища ──
# Шукаємо .env спочатку у homework/, потім у local-llama-demo/
HOMEWORK_DIR = Path(__file__).parent
for env_candidate in [
    HOMEWORK_DIR / ".env",
    HOMEWORK_DIR.parent / "local-llama-demo" / ".env",
]:
    if env_candidate.exists():
        load_dotenv(env_candidate)
        break

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama2"    # llama2:latest
OPENAI_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """Ти — парсер транскриптів ділових зустрічей.
Твоє завдання: витягнути з тексту структуровану інформацію та повернути ТІЛЬКИ валідний JSON.
Ніяких пояснень, ніякого markdown, ніяких зайвих слів — ТІЛЬКИ JSON."""

EXTRACTION_PROMPT_TEMPLATE = """Прочитай текст нижче і витягни:
1. summary — одне речення-підсумок зустрічі (українською)
2. tasks — список всіх завдань з полями: owner (хто відповідає), task (що зробити), deadline (дата у форматі YYYY-MM-DD, або null якщо не вказано)
3. decisions — список прийнятих рішень (рядки)

Текст зустрічі:
---
{text}
---

Поверни ТІЛЬКИ JSON у такому форматі (без жодного додаткового тексту):
{{
  "summary": "...",
  "tasks": [
    {{"owner": "...", "task": "...", "deadline": "..."}},
    ...
  ],
  "decisions": [
    "...",
    ...
  ]
}}"""


# ── OLLAMA (self-hosted) ──────────────────────────────────────────────────────

def call_ollama(prompt: str) -> tuple[str, float]:
    """
    Викликає локальну модель через Ollama API.
    Повертає (текст відповіді, latency_сек).
    """
    start = time.time()
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
            "stream": False,
            "options": {
                "temperature": 0.1,   # низька температура для стабільного JSON
                "num_predict": 1024,
            }
        },
        timeout=600
    )
    latency = time.time() - start
    response.raise_for_status()
    return response.json()["response"], latency


# ── OPENAI (cloud) ────────────────────────────────────────────────────────────

def call_openai(prompt: str) -> tuple[str, float, dict]:
    """
    Викликає GPT-4o-mini через OpenAI API.
    Повертає (текст відповіді, latency_сек, usage_dict).
    """
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    start = time.time()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    latency = time.time() - start
    usage = {
        "prompt_tokens":     response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens":      response.usage.total_tokens,
    }
    return response.choices[0].message.content, latency, usage


# ── JSON вилучення з відповіді ────────────────────────────────────────────────

def extract_json_from_text(text: str) -> dict | None:
    """
    Намагається знайти та спарсити JSON у тексті відповіді.
    Ollama іноді додає markdown або пояснення — це відфільтровується.
    """
    # Спочатку пробуємо спарсити відповідь напряму
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Шукаємо JSON-блок у відповіді (між ```json ... ``` або просто { ... })
    patterns = [
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
        r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    return None


# ── Оцінка токенів (для Ollama, де немає точного лічильника) ─────────────────

def estimate_tokens(text: str) -> int:
    """Наближена оцінка: слово ≈ 1.3 токена."""
    return int(len(text.split()) * 1.3)


# ── Основна функція екстракції ────────────────────────────────────────────────

def extract_meeting_data(
    text: str,
    provider: str = "ollama",
    dataset_name: str = "unknown"
) -> dict:
    """
    Витягує структурований JSON з тексту зустрічі.

    Args:
        text:         Транскрипт / протокол зустрічі
        provider:     "ollama" або "openai"
        dataset_name: Назва датасету (для логів)

    Returns:
        dict з полями: result, json_valid, latency, tokens_in, tokens_out,
                       tokens_total, cost_usd, raw_response
    """
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(text=text)

    output = {
        "dataset":      dataset_name,
        "provider":     provider,
        "json_valid":   False,
        "result":       None,
        "latency":      0.0,
        "tokens_in":    0,
        "tokens_out":   0,
        "tokens_total": 0,
        "cost_usd":     0.0,
        "raw_response": "",
        "error":        None,
    }

    try:
        if provider == "ollama":
            raw, latency = call_ollama(prompt)
            output["latency"]    = round(latency, 2)
            output["tokens_in"]  = estimate_tokens(prompt)
            output["tokens_out"] = estimate_tokens(raw)
            output["cost_usd"]   = 0.0

        elif provider == "openai":
            raw, latency, usage = call_openai(prompt)
            output["latency"]    = round(latency, 2)
            output["tokens_in"]  = usage["prompt_tokens"]
            output["tokens_out"] = usage["completion_tokens"]
            output["cost_usd"]   = round(usage["total_tokens"] / 1_000_000 * 0.30, 6)

        else:
            raise ValueError(f"Unknown provider: {provider}")

        output["tokens_total"] = output["tokens_in"] + output["tokens_out"]
        output["raw_response"] = raw

        parsed = extract_json_from_text(raw)
        if parsed:
            output["json_valid"] = True
            output["result"]     = parsed
        else:
            output["error"] = f"JSON parse failed. Raw: {raw[:300]}"

    except Exception as e:
        output["error"] = str(e)

    return output


# ── Виведення результату ──────────────────────────────────────────────────────

def print_result(output: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  Provider : {output['provider'].upper()}  |  Dataset: {output['dataset']}")
    print(f"{'='*60}")
    print(f"  JSON Valid  : {'YES' if output['json_valid'] else 'NO'}")
    print(f"  Latency     : {output['latency']}s")
    print(f"  Tokens in   : {output['tokens_in']}")
    print(f"  Tokens out  : {output['tokens_out']}")
    print(f"  Tokens total: {output['tokens_total']}")
    print(f"  Cost        : ${output['cost_usd']:.6f}")
    if output["error"]:
        print(f"  Error: {output['error']}")
    if output["result"]:
        print(f"\nExtracted JSON:\n")
        print(json.dumps(output["result"], indent=2, ensure_ascii=False))
    print()


# ── Запуск на всіх датасетах та збереження результатів ───────────────────────

def run_all(samples_dir: Path, results_dir: Path) -> list[dict]:
    datasets = {
        "simple":    samples_dir / "simple_meeting.txt",
        "chaotic":   samples_dir / "chaotic_standup.txt",
        "technical": samples_dir / "technical_sync.txt",
    }
    providers = ["ollama", "openai"]
    all_metrics = []

    for dataset_name, filepath in datasets.items():
        if not filepath.exists():
            print(f"File not found: {filepath}")
            continue

        text = filepath.read_text(encoding="utf-8")
        print(f"\n{'#'*60}")
        print(f"#  Processing: {dataset_name.upper()}")
        print(f"{'#'*60}")

        for provider in providers:
            print(f"\n>> Running {provider}...")
            output = extract_meeting_data(text, provider=provider, dataset_name=dataset_name)
            print_result(output)

            # Зберігаємо JSON-результат
            result_file = results_dir / f"{dataset_name}_{provider}.json"
            result_file.write_text(
                json.dumps(output, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )

            all_metrics.append(output)

    return all_metrics


# ── Генерація eval_results.csv ────────────────────────────────────────────────

def save_eval_csv(metrics: list[dict], output_path: Path) -> None:
    """Зберігає зведену таблицю метрик у CSV."""
    lines = [
        "Dataset,Provider,JSON Valid,Latency (s),Tokens In,Tokens Out,Tokens Total,Cost (USD)"
    ]
    for m in metrics:
        valid = "TRUE" if m["json_valid"] else "FALSE"
        lines.append(
            f"{m['dataset']},{m['provider']},{valid},"
            f"{m['latency']},{m['tokens_in']},{m['tokens_out']},"
            f"{m['tokens_total']},{m['cost_usd']:.6f}"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nEval results saved to {output_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base_dir    = Path(__file__).parent
    samples_dir = base_dir / "samples"
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)

    if len(sys.argv) == 3:
        # Поодинокий запуск: python extraction_agent.py <file> <provider>
        input_file = Path(sys.argv[1])
        provider   = sys.argv[2]
        text       = input_file.read_text(encoding="utf-8")
        output     = extract_meeting_data(text, provider=provider, dataset_name=input_file.stem)
        print_result(output)
    else:
        # Повний запуск по всіх датасетах
        metrics = run_all(samples_dir, results_dir)
        save_eval_csv(metrics, base_dir / "eval_results.csv")
        print("\nAll done!")
