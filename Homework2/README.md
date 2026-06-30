# Homework — Lesson 06: LLM Engineering

Extraction Agent: витягує структуровані дані (summary, tasks, decisions) з транскриптів зустрічей. Порівняння **Ollama (Llama2 7B, self-hosted)** та **OpenAI (GPT-4o-mini, cloud)**.

## Структура

```
homework/
├── extraction_agent.py     # основний скрипт
├── eval_results.csv        # зведена таблиця метрик
├── ANALYSIS.md             # висновки та аналіз
├── requirements.txt
├── .env.example            # шаблон для API ключа
├── samples/                # 3 тестових тексти
│   ├── simple_meeting.txt
│   ├── chaotic_standup.txt
│   └── technical_sync.txt
├── results/                # JSON-результати кожного запуску
│   ├── simple_ollama.json
│   ├── simple_openai.json
│   ├── chaotic_ollama.json
│   ├── chaotic_openai.json
│   ├── technical_ollama.json
│   └── technical_openai.json
└── screenshots/            # докази виконання
```

## Запуск

1. Встановити Ollama та запустити:
   ```bash
   ollama pull llama2
   ollama serve
   ```

2. Створити `.env` (на основі `.env.example`):
   ```bash
   cp .env.example .env
   # додати свій OPENAI_API_KEY
   ```

3. Встановити залежності:
   ```bash
   pip install -r requirements.txt
   ```

4. Запустити:
   ```bash
   # Окремий запуск
   python extraction_agent.py samples/simple_meeting.txt openai
   python extraction_agent.py samples/chaotic_standup.txt ollama

   # Повний запуск по всіх 3 датасетах та обох провайдерах
   python extraction_agent.py
   ```

## Результати

Див. `eval_results.csv` для метрик та `ANALYSIS.md` для висновків.
