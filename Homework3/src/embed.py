"""
Генерує ембеддинги для documents або queries з JSONL-файлу і зберігає у .npy.

Output:
    <output>.npy        — float32 numpy масив (N, dim), L2-нормалізований
    <output>.ids.json   — паралельний список ID, length == N

Запуск:
    # Корпус (документи)
    python src/embed.py \\
        --input data/corpus.jsonl \\
        --output data/embeddings_corpus.npy

    # Запити (увага: --query-mode!)
    python src/embed.py \\
        --input data/queries.jsonl \\
        --output data/embeddings_queries.npy \\
        --query-mode

    # Швидкий smoke-test (тільки перші 5К)
    python src/embed.py \\
        --input data/corpus.jsonl \\
        --output data/embeddings_corpus.npy \\
        --max-docs 5000 \\
        --force

ВАЖЛИВО про --query-mode:
    BGE — асиметрична модель. Документи ембедяться як є,
    а запити з префіксом "Represent this sentence for searching relevant passages: ".
    Без цього recall впаде на 2-5%.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# Офіційна BGE-інструкція для query-side ембеддингу
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def _read_jsonl(path: Path, query_mode: bool, max_docs: int | None) -> Tuple[List[str], List[str]]:
    """
    Читає JSONL і повертає (texts, ids).

    Для документів: text = title + ". " + text (title може бути порожній).
    Для запитів: text = тільки поле "text".
    """
    texts: List[str] = []
    ids: List[str] = []

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_docs is not None and i >= max_docs:
                break
            row = json.loads(line)

            doc_id = str(row["_id"])

            if query_mode:
                text = row.get("text", "") or ""
            else:
                title = row.get("title", "") or ""
                body = row.get("text", "") or ""
                # Об'єднуємо title+text, якщо title є; інакше просто text.
                # У BeIR/quora title порожній — буде просто body.
                text = f"{title}. {body}".strip(". ").strip() if title else body

            texts.append(text)
            ids.append(doc_id)

    return texts, ids


def _embed(
    texts: List[str],
    model_name: str,
    batch_size: int,
    query_mode: bool,
    normalize: bool,
) -> np.ndarray:
    """Прогнати тексти через модель, повернути (N, dim) float32 numpy."""
    print(f"[model] Завантажую {model_name}...")
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    print(f"[model] OK. dim={dim}, max_seq_length={model.max_seq_length}")

    if query_mode:
        print(f"[query-mode] Додаю BGE-інструкцію до кожного query.")
        texts = [BGE_QUERY_INSTRUCTION + t for t in texts]

    print(f"[encode] Кодую {len(texts):,} текстів (batch_size={batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    ).astype(np.float32)

    print(f"[encode] OK. shape={embeddings.shape}, dtype={embeddings.dtype}")
    return embeddings


def _save(embeddings: np.ndarray, ids: List[str], out_path: Path) -> None:
    """Збереже <out_path>.npy і <out_path>.ids.json."""
    ids_path = out_path.with_suffix(out_path.suffix + ".ids.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)
    with ids_path.open("w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False)

    npy_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[save] {out_path}  ({npy_mb:.1f} MB)")
    print(f"[save] {ids_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate embeddings for a JSONL file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path, help="JSONL з полями _id, text (опц. title)")
    parser.add_argument("--output", required=True, type=Path, help="Куди записати .npy (поряд буде .ids.json)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Sentence-transformers model (default: {DEFAULT_MODEL})")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size для encode (default: 64)")
    parser.add_argument(
        "--query-mode",
        action="store_true",
        help="Додає BGE query-інструкцію (вмикай для queries.jsonl, НЕ вмикай для corpus.jsonl)",
    )
    parser.add_argument("--max-docs", type=int, default=None, help="Обмежити кількість документів (для дебагу)")
    parser.add_argument("--no-normalize", action="store_true", help="Не робити L2-нормалізацію (за замовчуванням робимо)")
    parser.add_argument("--force", action="store_true", help="Перерахувати, навіть якщо output вже є")
    args = parser.parse_args()

    in_path: Path = args.input.resolve()
    out_path: Path = args.output.resolve()
    ids_path = out_path.with_suffix(out_path.suffix + ".ids.json")

    if not in_path.exists():
        raise FileNotFoundError(f"Input не знайдено: {in_path}")

    if out_path.exists() and ids_path.exists() and not args.force:
        out_mb = out_path.stat().st_size / 1024 / 1024
        print(f"[skip] {out_path.name} вже існує ({out_mb:.1f} MB). Use --force to overwrite.")
        return

    print(f"[input]  {in_path}")
    print(f"[output] {out_path}")
    print(f"[mode]   {'QUERIES (with BGE instruction)' if args.query_mode else 'DOCUMENTS'}")

    print("[read] Читаю JSONL...")
    texts, ids = _read_jsonl(in_path, query_mode=args.query_mode, max_docs=args.max_docs)
    print(f"[read] OK. {len(texts):,} рядків.")

    if not texts:
        raise RuntimeError(f"Файл {in_path} порожній — нема що ембедити.")

    embeddings = _embed(
        texts=texts,
        model_name=args.model,
        batch_size=args.batch_size,
        query_mode=args.query_mode,
        normalize=not args.no_normalize,
    )

    # Sanity check: L2-norm близька до 1, якщо normalize=True
    if not args.no_normalize:
        sample_norms = np.linalg.norm(embeddings[:5], axis=1)
        print(f"[sanity] L2-norms перших 5: {sample_norms.round(4).tolist()}  (очікуємо ~1.0)")

    _save(embeddings, ids, out_path)
    print("\nГотово.")


if __name__ == "__main__":
    main()
