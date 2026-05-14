# 🧪 Homework — Vector DB Benchmark Lab

**Тема:** Lesson 8 · Vector Databases у продакшні

---

## 🎯 Мета

 **виміряти роботу векторних бд власноруч** на реальному масштабі (523K векторів) і побудувати **Pareto-frontier** «recall vs latency».

На виході — **обґрунтований вибір** vector DB і параметрів HNSW для гіпотетичного продакшну.

---

## 📌 Мій звіт

Виконала повний бенчмарк 5 vector DB на датасеті **BeIR/quora (522,931 документів, 2,000 queries із qrels)** з ембеддингами `BAAI/bge-small-en-v1.5` (384 dim, L2-нормалізовані).

Працювала на **старому Intel Mac**, що додало багато операційних труднощів, які не пов'язані безпосередньо з бенчмарком.

### Результати

| БД | Index time | Disk | p50 | p95 | p99 | Recall@10 | MRR@10 |
|---|---|---|---|---|---|---|---|
| FAISS Flat | 5s | 766 MB | 52.8 ms | 61.7 ms | 73.0 ms | 0.9359 | 0.868 |
| FAISS HNSW | 4 хв | 838 MB | **0.29 ms** | **0.41 ms** | **0.49 ms** | 0.9354 | 0.869 |
| Chroma | 20 хв | 934 MB | 2.08 ms | 3.24 ms | 4.16 ms | 0.9343 | 0.868 |
| Qdrant | 8 хв | 1624 MB | 67.1 ms | 81.6 ms | 88.9 ms | 0.9354 | 0.868 |
| pgvector | 38 хв | 1857 MB | 9.8 ms | 15.6 ms | 21.1 ms | 0.9324 | 0.866 |

Графіки: `results/pareto_frontier.png`, `results/latency_distribution.png`, `results/disk_size_chart.png`, `results/terminal_screenshot.png`.

### Ключові інсайти

1. **FAISS HNSW — winner за latency** (0.29 ms p50 на 523K векторах). Це in-process бібліотека з approximate-індексом, який обходить пам'яті у L1 cache.

2. **FAISS Flat виявився повільнішим за HNSW у ~180×.** Він читає 800 MB векторів з RAM на кожен query. На моєму старому Intel Mac повільно, на Apple Silicon було б швидше.

3. **Recall майже однаковий у всіх БД (~0.935)**, навіть у FAISS Flat. Opus пише, що це через особливість датасету: "Причина — структура qrels у Quora: датасет має тисячі майже-дублікатів питань, але qrels позначають тільки **офіційні пари**. Тому навіть exact-пошук знаходить "сусідні" дублікати з вищою similarity, ніж позначений правильним".

4. **Qdrant — повільніший за всіх (67 ms p50).** Це **HTTP overhead** через REST API. На gRPC або з кращою інтеграцією було б 2-3× швидше.

5. **pgvector індексувався 38 хвилин** — це **once-cost**, але показовий для production-планування. SQL-БД не оптимізована під vector workload.

### Trade-off за категоріями

- **exact recall + малі дані** → FAISS Flat
- **Production-grade ANN + найкраща latency** → FAISS HNSW (in-process) або Qdrant (network service)
- **Швидкий MVP без Docker** → Chroma

## 📦 Що треба зробити

### Setup

**Датасет:** [BeIR/quora](https://huggingface.co/datasets/BeIR/quora)
- ~523K документів (duplicate-pair questions)
- ~10K тестових queries з ground truth (`qrels`)
- Реальний production-масштаб, ще влізає на ноут

**Embedding model:** [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) (384d, локально, безкоштовно)
- Або `text-embedding-3-small` через OpenAI (~$0.10 на весь датасет, 1536d)

**Vector DBs (обов'язково 5):**
1. **FAISS Flat** (baseline, 100% recall)
2. **FAISS HNSW**
3. **Qdrant** (Docker)
4. **Chroma** (embedded persistent)
5. **pgvector** (Postgres + HNSW через Docker)

---

## 📊 Метрики, які ОБОВ'ЯЗКОВО виміряти

### 1. Якість пошуку
- **Recall@10** — % релевантних документів у top-10
- **MRR@10** — Mean Reciprocal Rank (де перший правильний)

### 2. Швидкість
- **Indexing time** — скільки секунд/хвилин на побудову індексу
- **Query latency p50 / p95 / p99** — медіана, 95-й, 99-й перцентиль на 1000+ queries

### 3. Ресурси

- **Disk size** — розмір індексу на диску

---

## 📁 Очікувана структура репо

```
homework-vector-db-benchmark/
├── README.md                    # як запустити
├── requirements.txt
├── docker-compose.yml           # Qdrant + Postgres (pgvector)
├── .env.example
├── data/
│   └── .gitignore               # quora dataset не комітити
├── src/
│   ├── load_data.py             # завантаження BeIR/quora
│   ├── embed.py                 # embedding + cache (.npy)
│   ├── benchmarks/
│   │   ├── base.py              # абстрактний VectorDB interface
│   │   ├── faiss_flat.py
│   │   ├── faiss_hnsw.py
│   │   ├── qdrant_db.py
│   │   ├── chroma_db.py
│   │   └── pgvector_db.py
│   ├── metrics.py               # recall@K, MRR, latency percentiles
│   ├── runner.py                # запуск усіх бенчмарків
│   └── plot.py                  # генерація графіків
└── results/
    ├── results.csv              # сирі цифри з усіх БД
    ├── pareto_frontier.png      # графік recall vs latency
    ├── latency_distribution.png # p50 / p95 / p99 порівняння
    ├── disk_size_chart.png      # розмір індексу для кожної БД
    └── benchmark_terminal.png   # скріншот фінального output
```

---

## 🛠️ Базовий шаблон (interface)

Реалізуй спільний інтерфейс для всіх БД, щоб benchmark-runner працював з кожною однаково:

```python
# src/benchmarks/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import numpy as np

class VectorDB(ABC):
    """Спільний інтерфейс для FAISS / Qdrant / Chroma / pgvector."""

    @abstractmethod
    def index(self, vectors: np.ndarray, ids: List[str]) -> None:
        """
        Побудувати індекс з векторів.
        vectors: shape (N, dim), float32, L2-нормалізовані для cosine
        ids: рядкові ID, паралельні до vectors
        """

    @abstractmethod
    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Знайти top-K найближчих векторів.
        query_vec: shape (dim,) — 1D! Wrapper має сам зробити reshape якщо треба.
        Повертає: [(doc_id, score), ...] довжиною top_k
        """

    @abstractmethod
    def disk_size_mb(self) -> float:
        """Розмір індексу на диску в MB (0 якщо in-memory)."""

    def cleanup(self) -> None:
        """Закрити з'єднання, видалити тимчасові файли. За замовчуванням no-op."""
        pass
```

---

## 📈 Канонічний benchmark runner

```python
# src/runner.py
import time
import numpy as np
from typing import Dict, List

WARMUP_QUERIES = 50   # перші N запитів НЕ враховуємо (cold cache, JIT)
NUM_REPEATS = 3       # повторюємо вимір, беремо медіану


def _recall_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    """Recall@K = |retrieved ∩ relevant| / min(K, |relevant|)."""
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]) & relevant)
    return hits / min(k, len(relevant))


def _mrr_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    """MRR@K = 1 / rank першого правильного результату (0 якщо нема)."""
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def benchmark_db(
    db,
    doc_vectors: np.ndarray,        # (N, dim) float32
    doc_ids: List[str],             # parallel ID's до doc_vectors
    query_vectors: np.ndarray,      # (Q, dim) float32
    query_ids: List[str],           # ID кожного запиту з qrels
    qrels: Dict[str, set],          # {query_id: {relevant_doc_id, ...}}
    top_k: int = 10,
) -> Dict:
    # === INDEX ===
    t0 = time.perf_counter()
    db.index(doc_vectors, ids=doc_ids)
    index_time = time.perf_counter() - t0

    # === WARMUP ===
    for q_vec in query_vectors[:WARMUP_QUERIES]:
        db.search(q_vec, top_k=top_k)

    # === MEASURED QUERIES (3 repeats, median) ===
    all_latencies: List[List[float]] = []
    recalls: List[float] = []
    mrrs: List[float] = []

    for repeat in range(NUM_REPEATS):
        latencies = []
        for q_vec, q_id in zip(query_vectors, query_ids):
            t0 = time.perf_counter()
            results = db.search(q_vec, top_k=top_k)
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

            if repeat == 0:
                retrieved_ids = [doc_id for doc_id, _score in results]
                relevant = qrels.get(q_id, set())
                recalls.append(_recall_at_k(retrieved_ids, relevant, top_k))
                mrrs.append(_mrr_at_k(retrieved_ids, relevant, top_k))
        all_latencies.append(latencies)

    # median across repeats per query, тоді percentiles
    latencies_arr = np.median(np.array(all_latencies), axis=0)

    return {
        "index_time_sec": round(index_time, 2),
        "disk_mb": round(db.disk_size_mb(), 1),
        "latency_p50_ms": round(float(np.percentile(latencies_arr, 50)), 3),
        "latency_p95_ms": round(float(np.percentile(latencies_arr, 95)), 3),
        "latency_p99_ms": round(float(np.percentile(latencies_arr, 99)), 3),
        "recall_at_10": round(float(np.mean(recalls)), 4),
        "mrr_at_10": round(float(np.mean(mrrs)), 4),
        "num_queries": len(query_vectors),
    }
```

---

## 📸 Скріншоти

У звіті мають бути **тільки скріншоти**:

- Графік **Pareto frontier** (recall vs latency)
- Графік **Disk size** для кожної БД
- Таблиця з результатами всіх БД
- Скріншот терміналу з фінальним output бенчмарка

---

## 🚀 Швидкий старт

```bash
# 1. Setup
git clone <твій-репо>
cd homework-vector-db-benchmark
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Підняти Qdrant + Postgres з pgvector у Docker
docker compose up -d
docker compose ps    # перевір що hw_qdrant і hw_pgvector запущені

# 3. Завантажити датасет BeIR/quora (~5 хв, один раз)
python src/load_data.py
# → data/corpus.jsonl    (~523K документів)
# → data/queries.jsonl   (~10K запитів)
# → data/qrels.tsv       (golden labels)

# 4. Згенерувати embeddings (~10-30 хв залежно від CPU, один раз)
python src/embed.py \
    --model BAAI/bge-small-en-v1.5 \
    --input data/corpus.jsonl \
    --output data/embeddings.npy

# 5. Запустити бенчмарк всіх 5 БД (~30-60 хв на ноуті)
python src/runner.py --output results/results.csv

# 6. Згенерувати графіки зі скріншотами
python src/plot.py --input results/results.csv --output results/

# 7. Переглянути результати (cross-platform)
# macOS:  open results/pareto_frontier.png
# Linux:  xdg-open results/pareto_frontier.png
# Win:    start results\pareto_frontier.png

# 8. Зупинити Docker після бенчмарка
docker compose down
```

**`docker-compose.yml` у корені репо** має містити сервіси `qdrant` (image `qdrant/qdrant:latest`, порти 6333/6334) і `postgres` (image `pgvector/pgvector:pg16`, порт 5432, env `POSTGRES_USER=bench`, `POSTGRES_PASSWORD=bench`, `POSTGRES_DB=bench`). Готовий приклад — у [docker-compose.yml](docker-compose.yml).

**`requirements.txt`** має включати: `numpy`, `sentence-transformers`, `torch`, `datasets`, `faiss-cpu`, `qdrant-client`, `chromadb`, `psycopg[binary]`, `pgvector`, `psutil`, `matplotlib`, `pandas`. Готовий список — у [requirements.txt](requirements.txt).

---

## 🧭 Як підходити до цього ДЗ

- Це **exploratory / optional** частина курсу.
- **Ок не зробити повністю** — важливо описати підхід і де саме виникли труднощі.
- **Критерії оцінювання:** підхід, спроби, аналіз помилок, висновки.

Це частина навчання через досвід, а не «здав / не здав».

