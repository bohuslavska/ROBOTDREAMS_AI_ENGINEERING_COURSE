# AWS Deployment

**Full step-by-step guide:** [DEPLOY.md](./DEPLOY.md)

## Quick start

```bash
# 1. Prepare model (on Mac)
./deploy/aws/scripts/export-model.sh whiner-models-YOUR_ACCOUNT_ID

# 2. Push images to ECR
./deploy/aws/push-images.sh YOUR_ACCOUNT_ID eu-central-1

# 3. Follow DEPLOY.md phases 3–9
```

## Files

| Path | Description |
|---|---|
| [DEPLOY.md](./DEPLOY.md) | **Main guide** — full AWS deployment |
| [.env.aws.example](./.env.aws.example) | Production env template |
| [scripts/](./scripts/) | export-model, gpu-bootstrap, start-inference, create-rds |
| [ecs/](./ecs/) | ECS task definition templates |
| [push-images.sh](./push-images.sh) | Build & push to ECR |

## Budget option

Deploy API + UI + RDS with `INFERENCE_MODE=openai` — skip GPU EC2 (~$50/month). See DEPLOY.md §12.
