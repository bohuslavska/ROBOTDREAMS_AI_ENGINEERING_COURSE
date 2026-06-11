from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app.chunking import chunk_text, count_tokens
from app.config import settings
from app.embedder import embed_texts, validate_embedding_dimension
from app.vector_store import recreate_collection, upsert_chunks


SOURCE_PATH = Path("data/source.md")
DOCUMENT_ID = "source"


def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Source file not found: {SOURCE_PATH}")

    text = SOURCE_PATH.read_text(encoding="utf-8")

    chunks = chunk_text(
        text=text,
        chunk_size_tokens=settings.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )

    if not chunks:
        raise ValueError("No chunks were created. Check data/source.md content.")

    chunk_ids = [
        f"{DOCUMENT_ID}::chunk_{index:04d}"
        for index in range(len(chunks))
    ]

    print(f"Source file: {SOURCE_PATH}")
    print(f"Total characters: {len(text):,}")
    print(f"Total chunks: {len(chunks)}")

    print("\nFirst 5 chunks:")
    for index, chunk in enumerate(chunks[:5]):
        print(f"{chunk_ids[index]} | {count_tokens(chunk)} tokens")
        print(chunk[:200].replace("\n", " "))
        print("---")

    print("\nCreating embeddings...")
    embeddings = embed_texts(chunks)

    if len(embeddings) != len(chunks):
        raise ValueError(
            f"Embeddings count mismatch: chunks={len(chunks)}, embeddings={len(embeddings)}"
        )

    for embedding in embeddings:
        validate_embedding_dimension(embedding)

    print(f"Embedding dimension: {len(embeddings[0])}")

    print("\nRecreating Qdrant collection...")
    recreate_collection()

    print("Uploading chunks and embeddings to Qdrant...")
    upsert_chunks(
        chunk_ids=chunk_ids,
        chunks=chunks,
        embeddings=embeddings,
        document_id=DOCUMENT_ID,
    )

    print("Done.")
    print(f"Collection: {settings.qdrant_collection}")


main()