import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Pattern


MAX_USER_INPUT_CHARS = 4_000

LOG_DIR = Path("logs")
SUSPICIOUS_REQUESTS_LOG = LOG_DIR / "suspicious_requests.log"
SUSPICIOUS_RESPONSES_LOG = LOG_DIR / "suspicious_responses.log"


PROMPT_INJECTION_PATTERNS: list[Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"reveal\s+(your\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"\bsystem\s*:", re.IGNORECASE),
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
    re.compile(r"<\|im_end\|>", re.IGNORECASE),
    re.compile(r"</s>", re.IGNORECASE),
    re.compile(r"developer\s+message", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+in\s+developer\s+mode", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
]


SYSTEM_PROMPT_LEAK_PATTERNS: list[Pattern[str]] = [
    re.compile(r"you\s+are\s+a\s+careful\s+q&a\s+assistant", re.IGNORECASE),
    re.compile(r"answer\s+only\s+using\s+the\s+provided\s+document\s+context", re.IGNORECASE),
    re.compile(r"do\s+not\s+invent\s+facts", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"developer\s+message", re.IGNORECASE),
    re.compile(r"<system_instructions>", re.IGNORECASE),
    re.compile(r"</system_instructions>", re.IGNORECASE),
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_jsonl_log(path: Path, payload: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def find_matching_pattern(text: str, patterns: list[Pattern[str]]) -> str | None:
    for pattern in patterns:
        if pattern.search(text):
            return pattern.pattern

    return None


def validate_user_input(message: str, api_key: str | None = None) -> None:
    """
    Validate user input before any RAG/LLM/cache work.

    Raises ValueError if input is invalid or suspicious.
    """
    if len(message) > MAX_USER_INPUT_CHARS:
        _write_jsonl_log(
            SUSPICIOUS_REQUESTS_LOG,
            {
                "created_at": _utc_now_iso(),
                "api_key": api_key,
                "reason": "input_too_long",
                "length": len(message),
                "max_length": MAX_USER_INPUT_CHARS,
                "message_preview": message[:300],
            },
        )

        raise ValueError(
            f"User input is too long. Max length is {MAX_USER_INPUT_CHARS} characters."
        )

    matched_pattern = find_matching_pattern(message, PROMPT_INJECTION_PATTERNS)

    if matched_pattern:
        _write_jsonl_log(
            SUSPICIOUS_REQUESTS_LOG,
            {
                "created_at": _utc_now_iso(),
                "api_key": api_key,
                "reason": "suspicious_input",
                "matched_pattern": matched_pattern,
                "message_preview": message[:500],
            },
        )

        raise ValueError("Suspicious input detected. Request blocked.")


def check_output_for_system_fragments(
    response_text: str,
    request_id: str,
    api_key: str,
    model: str,
) -> bool:
    """
    Check completed response for possible system prompt fragments.

    Does not block live streaming.
    Returns True if suspicious output was detected.
    """
    matched_pattern = find_matching_pattern(response_text, SYSTEM_PROMPT_LEAK_PATTERNS)

    if not matched_pattern:
        return False

    _write_jsonl_log(
        SUSPICIOUS_RESPONSES_LOG,
        {
            "created_at": _utc_now_iso(),
            "request_id": request_id,
            "api_key": api_key,
            "model": model,
            "reason": "possible_system_prompt_leak",
            "matched_pattern": matched_pattern,
            "response_preview": response_text[:1_000],
        },
    )

    return True