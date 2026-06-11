import re

import tiktoken


DEFAULT_ENCODING = "cl100k_base"


def count_tokens(text: str, encoding_name: str = DEFAULT_ENCODING) -> int:
    """
    Count tokens in text using tiktoken.

    This tokenizer is not exactly the same as sentence-transformers tokenizer,
    but it is good enough for approximate chunk sizing.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def normalize_text(text: str) -> str:
    """
    Normalize line endings and remove excessive blank lines.
    Keeps paragraph structure.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_paragraphs(text: str) -> list[str]:
    """
    Split text into paragraphs by empty lines.
    This works well for Markdown documents.
    """
    text = normalize_text(text)
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def get_last_tokens_text(
    text: str,
    token_count: int,
    encoding_name: str = DEFAULT_ENCODING,
) -> str:
    """
    Take last N tokens from text and decode them back into string.
    Used for overlap between chunks.
    """
    if token_count <= 0:
        return ""

    encoding = tiktoken.get_encoding(encoding_name)
    token_ids = encoding.encode(text)

    if len(token_ids) <= token_count:
        return text.strip()

    return encoding.decode(token_ids[-token_count:]).strip()


def split_long_paragraph(
    paragraph: str,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    encoding_name: str = DEFAULT_ENCODING,
) -> list[str]:
    """
    Fallback for very long paragraphs.

    If one paragraph is longer than chunk_size_tokens,
    we split it by token windows.
    """
    if chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")

    encoding = tiktoken.get_encoding(encoding_name)
    token_ids = encoding.encode(paragraph)

    chunks: list[str] = []
    step = chunk_size_tokens - chunk_overlap_tokens

    for start in range(0, len(token_ids), step):
        end = start + chunk_size_tokens
        chunk_ids = token_ids[start:end]
        chunk_text = encoding.decode(chunk_ids).strip()

        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def chunk_text(
    text: str,
    chunk_size_tokens: int = 500,
    chunk_overlap_tokens: int = 100,
    encoding_name: str = DEFAULT_ENCODING,
) -> list[str]:
    """
    Split text into chunks.

    Strategy:
    1. Split text into paragraphs.
    2. Add paragraphs to the current chunk until it reaches ~500 tokens.
    3. When chunk is full, save it.
    4. Start the next chunk with ~50 tokens overlap from previous chunk.
    """
    if chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")

    paragraphs = split_into_paragraphs(text)

    chunks: list[str] = []
    current_parts: list[str] = []

    for paragraph in paragraphs:
        paragraph_tokens = count_tokens(paragraph, encoding_name)

        # If one paragraph is too long, split it separately.
        if paragraph_tokens > chunk_size_tokens:
            if current_parts:
                current_chunk = "\n\n".join(current_parts).strip()
                chunks.append(current_chunk)
                current_parts = []

            chunks.extend(
                split_long_paragraph(
                    paragraph=paragraph,
                    chunk_size_tokens=chunk_size_tokens,
                    chunk_overlap_tokens=chunk_overlap_tokens,
                    encoding_name=encoding_name,
                )
            )
            continue

        candidate_parts = current_parts + [paragraph]
        candidate_chunk = "\n\n".join(candidate_parts).strip()
        candidate_tokens = count_tokens(candidate_chunk, encoding_name)

        # If paragraph fits into current chunk, keep adding.
        if candidate_tokens <= chunk_size_tokens:
            current_parts.append(paragraph)
            continue

        # Otherwise, save current chunk.
        current_chunk = "\n\n".join(current_parts).strip()
        chunks.append(current_chunk)

        # Start new chunk with overlap from previous chunk.
        overlap_text = get_last_tokens_text(
            text=current_chunk,
            token_count=chunk_overlap_tokens,
            encoding_name=encoding_name,
        )

        current_parts = [overlap_text, paragraph] if overlap_text else [paragraph]

    if current_parts:
        final_chunk = "\n\n".join(current_parts).strip()
        if final_chunk:
            chunks.append(final_chunk)

    return chunks