from contextlib import contextmanager
from typing import Iterator

from dotenv import load_dotenv
from langfuse import get_client

from app.config import settings


# Important:
# pydantic-settings reads .env for our app settings,
# but Langfuse SDK reads LANGFUSE_* from environment variables.
# load_dotenv() makes .env available to the SDK.
load_dotenv(".env")

def get_langfuse_client():
    return get_client()


def is_langfuse_enabled() -> bool:
    return bool(settings.langfuse_enabled)


@contextmanager
def langfuse_span(
    name: str,
    input_data=None,
    metadata: dict | None = None,
) -> Iterator:
    if not is_langfuse_enabled():
        yield None
        return

    langfuse = get_langfuse_client()

    with langfuse.start_as_current_observation(
        as_type="span",
        name=name,
        input=input_data,
        metadata=metadata or {},
    ) as span:
        yield span


@contextmanager
def langfuse_generation(
    name: str,
    model: str,
    input_data=None,
    metadata: dict | None = None,
) -> Iterator:
    if not is_langfuse_enabled():
        yield None
        return

    langfuse = get_langfuse_client()

    with langfuse.start_as_current_observation(
        as_type="generation",
        name=name,
        model=model,
        input=input_data,
        metadata=metadata or {},
    ) as generation:
        yield generation


def flush_langfuse() -> None:
    if is_langfuse_enabled():
        get_langfuse_client().flush()