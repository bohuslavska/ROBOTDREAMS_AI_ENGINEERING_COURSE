# Ukrainian Literary Whining Generator

**Ukrainian Literary Whining Generator** — це застосунок, який перетворює сучасні українські скарги на стилізовану літературну прозу.

Ідея проєкту — взяти звичайну фразу на кшталт:

```text
У мене жахливий день на роботі.
```

і згенерувати більш емоційний, художній, “класичний” варіант українською мовою.

У час, коли багато навчальних і pet-проєктів будуються навколо персональних фінансових асистентів, планувальників, HR-ботів або customer-support агентів, мені хотілося зробити щось більш оригінальне й творче. Тому я обрала задачу стилізації українського тексту та fine-tuning мовної моделі під вузький літературний стиль.

Цей проєкт був для мене спробою пройти повний цикл розробки LLM-застосунку:

* підготовка даних;
* формування instruction/SFT dataset;
* fine-tuning;
* локальний inference;
* API;
* UI;
* Docker;
* observability;
* deployment design;
* AWS GPU inference architecture.

---

## Структура проєкту

```text
app/
  api/                  FastAPI backend
  ui/                   Streamlit frontend
  inference_server/     GPU inference microservice для AWS

data/                   Training CSV + SFT jsonl

notebooks/
  01_extract_sad_sentences_updated.ipynb
  02_generate_modern_variants_updated.ipynb
  03_finetune_model.ipynb

models/
  adapters/
    qwen3-8b-lora-v2/   фінальний LoRA adapter

deploy/
  aws/                  AWS deployment scripts and guide
  prometheus/           Prometheus config
  grafana/              Grafana provisioning

docker-compose.yml
docker-compose.prod.yml
Dockerfile.api
Dockerfile.ui
Dockerfile.inference

requirements-api.txt
requirements-ui.txt
requirements-inference.txt

README.md
PROJECT_REPORT.md
SUBMISSION.md
.env.example
```

---

## Що робить застосунок

Користувач вводить сучасну скаргу українською мовою. Модель повертає стилізований літературний варіант.

Приклад вхідного тексту:

```text
У мене жахливий день на роботі.
```

Очікувана логіка роботи:

```text
API отримує текст → передає його в модель → повертає відповідь у літературному стилі.
```

Якщо модель недоступна, API не падає повністю, а повертає fallback-повідомлення:

```text
Ой лихо, моделі розгубила
```

У відповіді також є ознака:

```json
"is_fallback": true
```

Це зроблено для graceful degradation: користувач бачить зрозуміле повідомлення, а система продовжує працювати.

---

## Приклади роботи fine-tuned моделі `qwen3-8b-lora-v2`

Реальні відповіді з локального API у режимі:

```env
INFERENCE_MODE=local
```

| Сучасне, вхід                                              | Класичне, вихід моделі                                   |
| ---------------------------------------------------------- | -------------------------------------------------------- |
| У мене був жахливий день на роботі, я страшенно втомилась. | Тяжка була робота, тяжка, і я втомилася.                 |
| Він мене кинув і навіть не пояснив чому.                   | Він мене покинув і не сказав, чого ради.                 |
| Я одна, і ніхто мене не розуміє, все йде не так.           | Одна, і ніхто не розуміє, що я переживаю, що мені тяжко. |

Ці приклади не є ідеальним відтворенням “високої літератури”, але вони показують, що fine-tuned модель почала змінювати тон, лексику і структуру речень у потрібному напрямку.

---

## Режими inference

У проєкті передбачено кілька режимів inference:

| Режим    | Де працює          | Для чого                               |
| -------- | ------------------ | -------------------------------------- |
| `local`  | Mac + MLX          | Локальна розробка з fine-tuned моделлю |
| `remote` | AWS GPU service    | Production-like deployment             |
| `openai` | OpenAI API         | Опціональний fallback, якщо немає GPU/adapter |
| `mock`   | Hardcoded response | Тести та перевірка інфраструктури      |

Основний бажаний production-like режим:

```env
INFERENCE_MODE=remote
```

У цьому режимі FastAPI backend звертається до окремого GPU inference microservice, який має запускати fine-tuned модель.

---

## Локальний запуск через MLX

```bash
cp .env.example .env
# встановити INFERENCE_MODE=local

source .venv/bin/activate
pip install -r requirements-api.txt

uvicorn app.api.main:app --reload --port 8000
```

В окремому терміналі:

```bash
API_URL=http://localhost:8000 streamlit run app/ui/streamlit_app.py
```

Локальні URL:

| Сервіс   | URL                        |
| -------- | -------------------------- |
| API      | http://localhost:8000      |
| API docs | http://localhost:8000/docs |
| UI       | http://localhost:8501      |

---

## Docker: локальний запуск застосунку з observability

```bash
docker compose up --build
```

| Сервіс     | URL                        |
| ---------- | -------------------------- |
| UI         | http://localhost:8501      |
| API        | http://localhost:8080/docs |
| Grafana    | http://localhost:3000      |
| Prometheus | http://localhost:9090      |

Prometheus і Grafana додані, щоб показати базову observability:

* latency;
* кількість запитів;
* fallback rate;
* доступність моделі;
* metrics endpoint `/metrics`.

---

## Production-like stack з GPU inference

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up --build
```

Цей режим потребує:

* NVIDIA GPU;
* `nvidia-container-toolkit`;
* доступних model adapter files;
* запущеного inference service.

---

## API

Health check:

```bash
curl http://localhost:8080/health
```

Generation endpoint:

```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"text":"У мене жахливий день на роботі."}'
```

Metrics:

```text
GET /metrics
```

---

## Fine-tuning pipeline

Notebooks потрібно запускати в такому порядку:

1. `01_extract_sad_sentences_updated.ipynb`
2. `02_generate_modern_variants_updated.ipynb`
3. `03_finetune_model.ipynb`

Фінальний adapter output:

```text
models/adapters/qwen3-8b-lora-v2/
```

Фінальний training dataset:

```text
3428 SFT pairs
```

У проєкті використовується LoRA adapter, а не повністю merged model у репозиторії.

---

## AWS deployment design

Детальний AWS guide:

```text
deploy/aws/DEPLOY.md
```

Запланована AWS-архітектура:

```text
Користувач
  ↓
Application Load Balancer
  ↓
ECS services:
  - FastAPI backend
  - Streamlit UI
  ↓
Remote GPU inference service on EC2 g5.xlarge
  ↓
Fine-tuned Qwen3-8B LoRA adapter

Додаткові сервіси:
  - S3 для model artifacts
  - ECR для Docker images
  - RDS PostgreSQL для logging/storage
  - Prometheus/Grafana для monitoring
```

Очікувані environment variables для production-like API:

```env
INFERENCE_MODE=remote
INFERENCE_SERVICE_URL=http://GPU_PRIVATE_IP:8080
DATABASE_URL=postgresql+psycopg2://...
FALLBACK_MESSAGE=Ой лихо, моделі розгубила
```

---

## Що було реалізовано

**Локально (працює і перевірено):**

* FastAPI backend;
* Streamlit frontend;
* локальний MLX inference з fine-tuned adapter `qwen3-8b-lora-v2`;
* mock mode для швидких тестів;
* fallback-повідомлення, якщо модель недоступна;
* Prometheus metrics endpoint (`/metrics`);
* Docker Compose (API + UI + Postgres + Grafana + Prometheus);
* notebooks для data pipeline, SFT dataset (~3428 pairs) і fine-tuning;
* submission packaging script.

**Підготовлено для AWS (скрипти, конфіги, Docker images — без повного end-to-end deploy):**

* remote inference mode в API;
* GPU inference microservice (`Dockerfile.inference`);
* AWS guide і скрипти в `deploy/aws/`;
* ECS task definition templates;
* production-like `docker-compose.prod.yml`.

---

## Що не було повністю завершено

Повний AWS production deployment не був завершений end-to-end.

Головний blocker — AWS GPU quota. AWS акаунт мав такий ліміт:

```text
Running On-Demand G and VT instances: 0 vCPU
```

Через це AWS не дозволив запустити `g5.xlarge` GPU instance. Для `g5.xlarge` потрібно підняти quota хоча б до 4 vCPU.

Через цей blocker не було повністю перевірено:

* запуск GPU EC2 inference service на AWS;
* повний шлях запиту:

```text
ALB → ECS API → GPU inference service → fine-tuned model response
```

* фінальний public ALB URL з remote model inference;
* Grafana dashboard на реальному AWS traffic;
* live demo з моделлю, розгорнутою на GPU.

Також виникла неочікувана інженерна проблема: GPU inference Docker image дуже довго збирається і пушиться з Apple Silicon Mac. Inference image залежить від великого CUDA/PyTorch base image і має збиратися під `linux/amd64`, тоді як локальна машина — `arm64`. Я недооцінила, скільки часу може зайняти цей Docker build/push крок перед дедлайном.

---

## Чесний статус проєкту

Поточний стан проєкту:

```text
Працюючий локальний прототип + fine-tuned модель + підготовлена production-like AWS-архітектура.
```

Це не просто notebook з fine-tuning. У проєкті є реальна application structure з API, UI, model adapter, Docker, metrics, fallback behavior і deployment design.

Водночас я не можу чесно стверджувати, що AWS deployment повністю завершений, тому що remote GPU inference не був успішно запущений і протестований end-to-end через AWS quota limitation і нестачу часу.

Найточніший статус:

```text
Локальний inference працює.
API/UI infrastructure реалізована.
Docker/observability підготовлені.
AWS deployment частково підготовлений.
GPU deployment заблокований AWS quota і не завершений у фінальній інтеграції.
```

---

## Що можна запустити без live AWS

Найпростіший режим для перевірки — `mock`. Він дозволяє перевірити API/UI структуру без GPU та зовнішніх сервісів.

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements-api.txt -r requirements-ui.txt

export INFERENCE_MODE=mock
export DATABASE_URL=sqlite:///./demo.db

uvicorn app.api.main:app --port 8000
```

В окремому терміналі:

```bash
API_URL=http://localhost:8000 streamlit run app/ui/streamlit_app.py
```

Це демонструє:

* FastAPI backend;
* Streamlit frontend;
* request/response flow;
* mock mode (без моделі);
* базову структуру застосунку.

Для локального inference з fine-tuned моделлю потрібно використати:

```env
INFERENCE_MODE=local
```

і переконатися, що MLX-compatible adapter/model files доступні локально.

---

## Що включати в submission

Потрібно включити:

```text
app/
data/
notebooks/
models/adapters/qwen3-8b-lora-v2/adapters.safetensors
models/adapters/qwen3-8b-lora-v2/adapter_config.json
README.md
PROJECT_REPORT.md
SUBMISSION.md
docker-compose.yml
docker-compose.prod.yml
Dockerfile.api
Dockerfile.ui
Dockerfile.inference
deploy/
requirements-*.txt
.env.example
scripts/pack-submission.sh
```

Не потрібно включати:

```text
.venv/
.env
pluperfect_grac/
models/merged/
intermediate checkpoints like 0000200_*.safetensors
API keys
AWS credentials
large temporary files
```

---

## Packaging

Щоб створити submission archive:

```bash
cd /path/to/final_project
./scripts/pack-submission.sh
```

Очікуваний результат:

```text
ukrainian-literary-whiner-submission.zip
```

Приблизний розмір з фінальним adapter only:

```text
~100 MB
```

Якщо є обмеження на розмір upload:

```bash
./scripts/pack-submission.sh --no-model
```

У такому випадку adapter можна передати окремо, наприклад через Google Drive.

---

## Перед submission

Checklist:

```text
[ ] README.md оновлений
[ ] PROJECT_REPORT.md містить фінальні числа: 3428 pairs, qwen3-8b-lora-v2
[ ] SUBMISSION.md пояснює, що саме можуть запустити reviewers
[ ] notebooks не містять hardcoded API keys
[ ] .env не включений
[ ] .venv не включений
[ ] intermediate checkpoints не включені
[ ] включені тільки фінальні adapter weights
[ ] zip file має прийнятний розмір
[ ] mock mode запускається
[ ] local API health check працює
```

---

## Фінальна рефлексія

Я свідомо обрала менш стандартну ідею проєкту, бо хотіла попрацювати не тільки з application orchestration, а й з fine-tuning та українськомовною генерацією. Це зробило проєкт цікавішим, але також технічно ризикованішим.

Головний висновок: fine-tuning і локальний inference — це лише частина роботи. Packaging і deployment GPU-based LLM service — це окрема інженерна задача, яка потребує значно більше часу, ніж здається спочатку. Особливо якщо Docker images великі, локальна машина — Apple Silicon, а AWS має GPU quota restrictions.

Якби було більше часу, я б завершила AWS GPU path так:

1. отримати GPU quota increase;
2. запустити `g5.xlarge`;
3. підняти inference server на EC2;
4. підключити ECS API до GPU private IP;
5. протестувати повний шлях `ALB → API → model`;
6. записати коротке demo;
7. зупинити GPU instance після demo, щоб контролювати витрати.

Навіть із незавершеним cloud deployment, проєкт демонструє основну ідею, fine-tuned модель, application layer, observability, fallback behavior і реалістичний deployment plan.
