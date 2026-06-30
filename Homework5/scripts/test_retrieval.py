from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app.embedder import embed_query
from app.vector_store import search_chunks

def main() -> None:
    query = "What is prompt injection?"

    query_embedding = embed_query(query)

    results = search_chunks(
        query_embedding=query_embedding,
        top_k=3,
    )

    print(f"Query: {query}")
    print(f"Found chunks: {len(results)}")

    for index, chunk in enumerate(results, start=1):
        print(f"\n--- Result {index} ---")
        print("chunk_id:", chunk["chunk_id"])
        print("score:", chunk["score"])
        print(chunk["text"][:700])


main()