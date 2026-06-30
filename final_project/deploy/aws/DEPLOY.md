# AWS Deployment — Detailed Guide

Complete step-by-step guide to deploy **Ukrainian Literary Whining Generator** on AWS.

**Stack:**
| Component | AWS service | Purpose |
|---|---|---|
| API | ECS Fargate | FastAPI, calls GPU inference |
| UI | ECS Fargate | Streamlit |
| Inference | EC2 g5.xlarge (GPU) | Your fine-tuned Qwen3 + LoRA v2 |
| Database | RDS PostgreSQL | Log generations |
| Monitoring | ECS Fargate (Prometheus + Grafana) | Metrics dashboard |
| Load balancer | ALB | Public HTTPS access |

**Estimated monthly cost (eu-central-1):** ~$750–900 if GPU runs 24/7. **Stop GPU EC2 when not demoing** → ~$80–100/month for the rest.

---

## Table of contents

1. [Prerequisites](#1-prerequisites)
2. [Architecture diagram](#2-architecture-diagram)
3. [Phase 1 — Prepare model on Mac](#3-phase-1--prepare-model-on-mac)
4. [Phase 2 — Build & push Docker images](#4-phase-2--build--push-docker-images)
5. [Phase 3 — Network & security groups](#5-phase-3--network--security-groups)
6. [Phase 4 — RDS PostgreSQL](#6-phase-4--rds-postgresql)
7. [Phase 5 — GPU inference EC2](#7-phase-5--gpu-inference-ec2)
8. [Phase 6 — ECS cluster & services](#8-phase-6--ecs-cluster--services)
9. [Phase 7 — Application Load Balancer](#9-phase-7--application-load-balancer)
10. [Phase 8 — Observability (Grafana)](#10-phase-8--observability-grafana)
11. [Phase 9 — End-to-end testing](#11-phase-9--end-to-end-testing)
12. [Budget MVP (no GPU)](#12-budget-mvp-no-gpu)
13. [Troubleshooting](#13-troubleshooting)
14. [Teardown & cost saving](#14-teardown--cost-saving)

---

## 1. Prerequisites

On your Mac:

```bash
# AWS CLI
aws --version          # v2 recommended
aws configure          # set Access Key, Secret, region eu-central-1

# Docker
docker --version

# Project venv with mlx-lm (for model export)
cd /path/to/final_project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-api.txt mlx-lm

# Verify fine-tuned adapter exists
ls models/adapters/qwen3-8b-lora-v2/adapters.safetensors
```

You need:
- AWS account with permissions for EC2, ECS, ECR, RDS, S3, ALB, IAM
- A **VPC** with at least 2 subnets in different AZs (default VPC works for learning)
- **~80 GB free** on GPU EC2 disk (model + Docker images)

Copy env template:
```bash
cp deploy/aws/.env.aws.example deploy/aws/.env.aws
# Edit with your account ID, passwords, etc.
```

---

## 2. Architecture diagram

```
                    Internet
                        │
                        ▼
              ┌─────────────────┐
              │   ALB :443/80   │
              └────────┬────────┘
           ┌───────────┼───────────┐
           ▼           ▼           ▼
      ┌────────┐  ┌────────┐  ┌──────────┐
      │ UI:8501│  │API:8000│  │Grafana   │
      │ (ECS)  │  │ (ECS)  │  │:3000 ECS)│
      └────────┘  └───┬────┘  └────┬─────┘
                        │            │
            INFERENCE_MODE=remote     │ scrapes
                        │            ▼
                        ▼       ┌────────────┐
              ┌─────────────────┐│ Prometheus │
              │ GPU EC2 :8080   ││  (ECS)     │
              │ inference svc   │└────────────┘
              │ Qwen3 + LoRA v2 │
              └─────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │ RDS PostgreSQL  │
              │  generations    │
              └─────────────────┘
```

**Security:** GPU port 8080 is **private only** (API security group → GPU security group). RDS is **private only**.

---

## 3. Phase 1 — Prepare model on Mac

MLX adapters run on Mac only. AWS GPU needs the adapter uploaded to S3.

```bash
cd /path/to/final_project
source .venv/bin/activate

# Fuse LoRA into base weights (optional but recommended)
chmod +x deploy/aws/scripts/export-model.sh
./deploy/aws/scripts/export-model.sh whiner-models-YOUR_ACCOUNT_ID
```

This:
1. Runs `mlx_lm.fuse` → `models/merged/qwen3-8b-lora-v2/`
2. Uploads adapter + merged weights to S3

Verify:
```bash
aws s3 ls s3://whiner-models-YOUR_ACCOUNT_ID/adapters/qwen3-8b-lora-v2/
# Should show adapters.safetensors (~74 MB)
```

> **Note:** The GPU inference container uses HuggingFace `Qwen/Qwen3-8B` + PEFT adapter. If PEFT load fails, check inference logs — you may need to convert MLX adapter to HF format or use merged HF weights. As interim, set `INFERENCE_MODE=openai` on the API (see [Budget MVP](#12-budget-mvp-no-gpu)).

---

## 4. Phase 2 — Build & push Docker images

```bash
chmod +x deploy/aws/push-images.sh
./deploy/aws/push-images.sh YOUR_ACCOUNT_ID eu-central-1
```

Creates 3 ECR repos and pushes:
- `whiner-api`
- `whiner-ui`
- `whiner-inference`

Save the printed image URIs into `deploy/aws/.env.aws`:
```
ECR_API=123456789012.dkr.ecr.eu-central-1.amazonaws.com/whiner-api:latest
ECR_UI=...
ECR_INFERENCE=...
```

---

## 5. Phase 3 — Network & security groups

Create 4 security groups in your VPC (via AWS Console → EC2 → Security Groups):

### SG: `whiner-alb`
| Type | Port | Source |
|---|---|---|
| HTTP | 80 | 0.0.0.0/0 |
| HTTPS | 443 | 0.0.0.0/0 |

### SG: `whiner-ecs`
| Type | Port | Source |
|---|---|---|
| Custom TCP | 8000 | whiner-alb SG |
| Custom TCP | 8501 | whiner-alb SG |
| Custom TCP | 3000 | whiner-alb SG (Grafana) |
| All outbound | * | 0.0.0.0/0 |

### SG: `whiner-gpu`
| Type | Port | Source |
|---|---|---|
| Custom TCP | 8080 | whiner-ecs SG only |
| SSH | 22 | Your IP (for setup) |

### SG: `whiner-rds`
| Type | Port | Source |
|---|---|---|
| PostgreSQL | 5432 | whiner-ecs SG |

Write down SG IDs — you'll need them below.

---

## 6. Phase 4 — RDS PostgreSQL

### Option A — AWS Console (easier)

1. RDS → Create database → PostgreSQL 16
2. Template: Free tier (or Dev/Test)
3. DB identifier: `whiner-db`
4. Master username: `postgres`, strong password
5. DB name: `whining`
6. VPC + **private subnets**, SG: `whiner-rds`
7. **Public access: No**

Wait ~10 minutes. Copy endpoint from RDS console.

Set in `deploy/aws/.env.aws`:
```
RDS_ENDPOINT=whiner-db.xxxxx.eu-central-1.rds.amazonaws.com
DATABASE_URL=postgresql+psycopg2://postgres:YOUR_PASSWORD@whiner-db.xxxxx.eu-central-1.rds.amazonaws.com:5432/whining
```

### Option B — CLI script

```bash
chmod +x deploy/aws/scripts/create-rds.sh
./deploy/aws/scripts/create-rds.sh default-vpc-subnet-group sg-RDS_ID YOUR_PASSWORD
```

Tables are created automatically on API startup (`create_tables()`).

---

## 7. Phase 5 — GPU inference EC2

### 7.1 Launch instance

EC2 → Launch instance:
| Setting | Value |
|---|---|
| Name | whiner-gpu |
| AMI | **Deep Learning AMI GPU PyTorch** (Ubuntu 22.04) OR Amazon Linux 2023 + install drivers |
| Instance type | **g5.xlarge** (24 GB GPU) or g4dn.xlarge (cheaper) |
| Key pair | Create/download `.pem` |
| VPC | Same as RDS/ECS |
| Subnet | Private subnet preferred |
| Auto-assign public IP | Enable (for initial setup) OR use bastion |
| Security group | whiner-gpu |
| Storage | **100 GB** gp3 |

### 7.2 Bootstrap GPU instance

SSH in:
```bash
ssh -i whiner.pem ec2-user@GPU_PUBLIC_IP
```

Run bootstrap:
```bash
bash -s < deploy/aws/scripts/gpu-bootstrap.sh
# OR paste gpu-bootstrap.sh contents
exit   # log back in for docker group
ssh -i whiner.pem ec2-user@GPU_PUBLIC_IP
```

### 7.3 Start inference container

On GPU instance (set your values):
```bash
export AWS_REGION=eu-central-1
export S3_BUCKET=whiner-models-YOUR_ACCOUNT_ID
export ECR_INFERENCE=YOUR_ACCOUNT.dkr.ecr.eu-central-1.amazonaws.com/whiner-inference:latest

# Copy start-inference.sh to instance, or run commands manually:
aws s3 sync s3://${S3_BUCKET}/adapters/qwen3-8b-lora-v2/ /opt/whiner/models/adapters/qwen3-8b-lora-v2/

aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.eu-central-1.amazonaws.com

docker run -d --name whiner-inference --gpus all --restart unless-stopped \
  -p 8080:8080 \
  -v /opt/whiner/models/adapters:/app/models/adapters:ro \
  -e USE_MLX=false \
  -e BASE_MODEL=Qwen/Qwen3-8B \
  -e LORA_ADAPTER_PATH=/app/models/adapters/qwen3-8b-lora-v2 \
  -e MODEL_VERSION=qwen3-8b-lora-v2 \
  -e 'FALLBACK_MESSAGE=Ой лихо, моделі розгубила' \
  $ECR_INFERENCE
```

First start downloads ~16 GB base model — **10–20 minutes**.

### 7.4 Verify inference (on GPU instance)

```bash
curl http://localhost:8080/health
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"text":"У мене жахливий день на роботі."}'
```

Note the **private IP** (e.g. `10.0.1.50`):
```bash
hostname -I
```

Set in `deploy/aws/.env.aws`:
```
GPU_INSTANCE_PRIVATE_IP=10.0.1.50
INFERENCE_SERVICE_URL=http://10.0.1.50:8080
```

---

## 8. Phase 6 — ECS cluster & services

### 8.1 Create ECS cluster

```bash
aws ecs create-cluster --cluster-name whiner-cluster --region eu-central-1
```

### 8.2 Create IAM roles (if not exist)

ECS needs:
- **ecsTaskExecutionRole** — pull ECR images, write logs (AWS managed policy `AmazonECSTaskExecutionRolePolicy`)
- **ecsTaskRole** — optional, for S3 access

Console: IAM → Roles → Create → AWS service → Elastic Container Service → Elastic Container Service Task

### 8.3 Register task definitions

Use templates in `deploy/aws/ecs/` — replace placeholders, then:

```bash
aws ecs register-task-definition --cli-input-json file://deploy/aws/ecs/task-api.json
aws ecs register-task-definition --cli-input-json file://deploy/aws/ecs/task-ui.json
aws ecs register-task-definition --cli-input-json file://deploy/aws/ecs/task-prometheus.json
aws ecs register-task-definition --cli-input-json file://deploy/aws/ecs/task-grafana.json
```

### 8.4 Create ECS services

**API service** (Fargate, 512 CPU / 1024 MB):
```bash
aws ecs create-service \
  --cluster whiner-cluster \
  --service-name whiner-api \
  --task-definition whiner-api \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx,subnet-yyy],securityGroups=[sg-ecs],assignPublicIp=ENABLED}" \
  --region eu-central-1
```

Repeat for `whiner-ui`, `whiner-prometheus`, `whiner-grafana`.

**API environment variables (critical):**
```
INFERENCE_MODE=remote
INFERENCE_SERVICE_URL=http://10.0.1.50:8080
MODEL_VERSION=qwen3-8b-lora-v2
DATABASE_URL=postgresql+psycopg2://postgres:PASSWORD@rds-endpoint:5432/whining
FALLBACK_MESSAGE=Ой лихо, моделі розгубила
STORE_USER_INPUTS=true
```

**UI environment:**
```
API_URL=http://INTERNAL_ALB_OR_API_URL:8000
```

> Easiest for first deploy: put API + UI on same host networking or use ALB path routing (see Phase 7).

---

## 9. Phase 7 — Application Load Balancer

### 9.1 Create ALB

EC2 → Load Balancers → Create Application Load Balancer:
- Name: `whiner-alb`
- Scheme: Internet-facing
- SG: `whiner-alb`
- Listeners: HTTP :80 (add HTTPS :443 later with ACM certificate)

### 9.2 Target groups

| Target group | Port | Health check path |
|---|---|---|
| whiner-api-tg | 8000 | `/health` |
| whiner-ui-tg | 8501 | `/_stcore/health` |
| whiner-grafana-tg | 3000 | `/api/health` |

Register ECS tasks as targets (or use ECS service with load balancer integration).

### 9.3 Listener rules

| Path | Forward to |
|---|---|
| `/api/*` | whiner-api-tg (strip prefix or configure FastAPI root_path) |
| `/grafana/*` | whiner-grafana-tg |
| `/*` (default) | whiner-ui-tg |

Update UI `API_URL` to public ALB URL:
```
API_URL=http://whiner-alb-xxxxx.eu-central-1.elb.amazonaws.com
```

For API under `/api` prefix, you may need a reverse proxy or set Streamlit to call `http://alb/api`.

**Simpler alternative for demo:** expose API on `:8080` and UI on `:8501` via two ALB listeners on different ports.

---

## 10. Phase 8 — Observability (Grafana)

After ECS prometheus + grafana services are running:

1. Open Grafana: `http://ALB_DNS:3000` (restrict to your IP in SG!)
2. Login: **admin / admin** (change password)
3. Dashboard **"Ukrainian Literary Whiner"** is auto-provisioned

**Metrics available:**
- Request rate by endpoint
- Generation latency p95
- Fallback rate (`is_fallback=true` → "Ой лихо, моделі розгубила")
- `whiner_model_available` gauge

Prometheus scrapes `api:8000/metrics` inside the VPC.

---

## 11. Phase 9 — End-to-end testing

```bash
ALB=http://YOUR_ALB_DNS

# Health
curl $ALB:8080/health
# Expect: {"status":"ok","inference_mode":"remote","model_available":true}

# Generate
curl -s -X POST $ALB:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"text":"У мене жахливий день на роботі, я втомилась."}' | python3 -m json.tool

# Check NOT fallback
# "is_fallback": false
# "model_version": "qwen3-8b-lora-v2"

# UI
open http://$ALB:8501
```

**Test fallback** (stop GPU container):
```bash
ssh gpu "docker stop whiner-inference"
curl .../generate   # should return "Ой лихо, моделі розгубила", is_fallback: true
ssh gpu "docker start whiner-inference"
```

---

## 12. Budget MVP (no GPU)

Skip GPU EC2 entirely for initial AWS demo:

1. Deploy API + UI + RDS only
2. Set API env:
   ```
   INFERENCE_MODE=openai
   OPENAI_API_KEY=sk-...
   ```
3. Users get good outputs via GPT; swap to `remote` when GPU is ready

Monthly cost: **~$50–80** (ECS + RDS + ALB).

---

## 13. Troubleshooting

| Problem | Fix |
|---|---|
| `model_available: false` | Check GPU SG allows 8080 from ECS SG; verify `INFERENCE_SERVICE_URL` uses **private IP** |
| API returns fallback always | `docker logs whiner-inference` on GPU; model download may still be running |
| RDS connection refused | ECS and RDS must be in same VPC; RDS SG must allow 5432 from ECS SG |
| ECR pull denied | ECS task execution role needs `AmazonECSTaskExecutionRolePolicy` |
| PEFT adapter load fails | MLX adapter ≠ HF format; use OpenAI mode or convert weights |
| UI can't reach API | Check `API_URL` env; CORS is open but URL must be reachable from browser |
| Grafana empty | Prometheus must reach `api:8000/metrics` on same Docker network / VPC |

**Logs:**
```bash
# ECS
aws logs tail /ecs/whiner-api --follow

# GPU inference
ssh gpu "docker logs -f whiner-inference"
```

---

## 14. Teardown & cost saving

**Stop GPU when not in use** (biggest saving):
```bash
aws ec2 stop-instances --instance-ids i-GPU_INSTANCE_ID
```
API gracefully returns fallback message.

**Delete everything:**
```bash
aws ecs update-service --cluster whiner-cluster --service whiner-api --desired-count 0
# Delete ECS services, cluster, ALB, RDS, EC2, ECR images, S3 bucket
```

---

## Quick reference — file map

| File | Purpose |
|---|---|
| `deploy/aws/.env.aws.example` | Production env template |
| `deploy/aws/scripts/export-model.sh` | Fuse + upload model to S3 |
| `deploy/aws/push-images.sh` | Build + push to ECR |
| `deploy/aws/scripts/gpu-bootstrap.sh` | GPU EC2 setup |
| `deploy/aws/scripts/start-inference.sh` | Run inference container |
| `deploy/aws/scripts/create-rds.sh` | Create RDS via CLI |
| `deploy/aws/ecs/task-*.json` | ECS task definition templates |
| `docker-compose.prod.yml` | Local prod-like test with GPU |

---

## Recommended deploy order (checklist)

- [ ] Phase 1: Export model → S3
- [ ] Phase 2: Push Docker images → ECR
- [ ] Phase 3: Create security groups
- [ ] Phase 4: Create RDS, note endpoint
- [ ] Phase 5: Launch GPU EC2, start inference, note private IP
- [ ] Phase 6: ECS cluster + api/ui services with correct env vars
- [ ] Phase 7: ALB + target groups
- [ ] Phase 8: Prometheus + Grafana services
- [ ] Phase 9: Test health, generate, UI, fallback
- [ ] Stop GPU instance when done demoing
