"""
Завантажує датасет BeIR/quora з HuggingFace і зберігає у data/.

Output:
    data/corpus.jsonl   — {"_id": str, "title": str, "text": str}   (~523K рядків)
    data/queries.jsonl  — {"_id": str, "text": str}                 (~10K рядків)
    data/qrels.tsv      — query-id<TAB>corpus-id<TAB>score          (golden labels)

Запуск:
    python src/load_data.py
    python src/load_data.py --force        # перевантажити, навіть якщо файли є
    python src/load_data.py --split dev    # використати dev-split замість test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


# Корінь репо = батьківська папка від src/
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

CORPUS_PATH = DATA_DIR / "corpus.jsonl"
QUERIES_PATH = DATA_DIR / "queries.jsonl"
QRELS_PATH = DATA_DIR / "qrels.tsv"


def _need_download(path: Path, force: bool) -> bool:
    """True, якщо файла нема або користувач передав --force."""
    if force:
        return True
    if not path.exists():
        return True
    # Захист від «порожнього» файла, якщо скрипт впав на середині попереднього запуску
    if path.stat().st_size == 0:
        return True
    return False


def _download_corpus(force: bool) -> None:
    if not _need_download(CORPUS_PATH, force):
        print(f"[skip] corpus вже існує: {CORPUS_PATH}")
        return

    print("[1/3] Завантажую corpus (BeIR/quora, ~523K документів)...")
    ds = load_dataset("BeIR/quora", "corpus", split="corpus")

    print(f"       Розмір: {len(ds):,} документів. Записую у {CORPUS_PATH}...")
    with CORPUS_PATH.open("w", encoding="utf-8") as f:
        for row in tqdm(ds, desc="corpus", unit="doc"):
            f.write(json.dumps({
                "_id": str(row["_id"]),
                "title": row.get("title", "") or "",
                "text": row.get("text", "") or "",
            }, ensure_ascii=False) + "\n")
    print(f"       OK → {CORPUS_PATH}")


def _download_queries(force: bool) -> None:
    if not _need_download(QUERIES_PATH, force):
        print(f"[skip] queries вже існують: {QUERIES_PATH}")
        return

    print("[2/3] Завантажую queries (BeIR/quora, ~10K запитів)...")
    ds = load_dataset("BeIR/quora", "queries", split="queries")

    print(f"       Розмір: {len(ds):,} запитів. Записую у {QUERIES_PATH}...")
    with QUERIES_PATH.open("w", encoding="utf-8") as f:
        for row in tqdm(ds, desc="queries", unit="q"):
            f.write(json.dumps({
                "_id": str(row["_id"]),
                "text": row.get("text", "") or "",
            }, ensure_ascii=False) + "\n")
    print(f"       OK → {QUERIES_PATH}")


def _download_qrels(force: bool, split: str) -> None:
    if not _need_download(QRELS_PATH, force):
        print(f"[skip] qrels вже існують: {QRELS_PATH}")
        return

    print(f"[3/3] Завантажую qrels (BeIR/quora-qrels, split={split})...")
    ds = load_dataset("BeIR/quora-qrels", split=split)

    print(f"       Розмір: {len(ds):,} пар. Записую у {QRELS_PATH}...")
    with QRELS_PATH.open("w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")  # TREC-style header
        for row in tqdm(ds, desc="qrels", unit="pair"):
            f.write(f"{row['query-id']}\t{row['corpus-id']}\t{row['score']}\n")
    print(f"       OK → {QRELS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BeIR/quora dataset.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Перезавантажити навіть якщо файли вже існують.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "dev"],
        help="Який split qrels використовувати (default: test).",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"data/ → {DATA_DIR}")

    _download_corpus(args.force)
    _download_queries(args.force)
    _download_qrels(args.force, args.split)

    print("\nГотово. Підсумок:")
    for p in (CORPUS_PATH, QUERIES_PATH, QRELS_PATH):
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  {p.relative_to(REPO_ROOT)}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
