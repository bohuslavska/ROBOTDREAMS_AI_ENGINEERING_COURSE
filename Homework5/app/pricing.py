PRICING_USD_PER_1M_TOKENS: dict[str, dict[str, float]] = {
    # Free / dev models
    "openrouter/free": {"input": 0.0, "output": 0.0},
    "meta-llama/llama-3.2-3b-instruct:free": {"input": 0.0, "output": 0.0},
    "qwen/qwen3-coder:free": {"input": 0.0, "output": 0.0},
    "google/gemma-4-31b-it:free": {"input": 0.0, "output": 0.0},

    # Pro / paid examples
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "google/gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "google/gemini-2.5-flash": {"input": 0.30, "output": 2.50},

    # Enterprise examples
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "anthropic/claude-sonnet-4.6": {"input": 3.00, "output": 15.00},
}


def calculate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """
    Calculate request cost using pricing.py as the only source of truth.

    Prices are stored as USD per 1M tokens.
    """
    if model.endswith(":free"):
        return 0.0

    if model not in PRICING_USD_PER_1M_TOKENS:
        raise ValueError(f"Missing pricing for model: {model}")

    pricing = PRICING_USD_PER_1M_TOKENS[model]

    input_cost = input_tokens * pricing["input"] / 1_000_000
    output_cost = output_tokens * pricing["output"] / 1_000_000

    return round(input_cost + output_cost, 8)