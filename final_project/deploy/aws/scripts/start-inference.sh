#!/usr/bin/env bash
# Start inference container on GPU EC2.
# Run on GPU instance after bootstrap + model download.
#
# Usage:
#   export ECR_INFERENCE=123456789012.dkr.ecr.eu-central-1.amazonaws.com/whiner-inference:latest
#   export S3_BUCKET=whiner-models-123456789012
#   export AWS_REGION=eu-central-1
#   ./deploy/aws/scripts/start-inference.sh

set -euo pipefail

ECR_INFERENCE="${ECR_INFERENCE:?Set ECR_INFERENCE}"
S3_BUCKET="${S3_BUCKET:?Set S3_BUCKET}"
AWS_REGION="${AWS_REGION:-eu-central-1}"
ADAPTER_DIR="/opt/whiner/models/adapters/qwen3-8b-lora-v2"

echo "=== Download adapter from S3 ==="
aws s3 sync "s3://${S3_BUCKET}/adapters/qwen3-8b-lora-v2/" "$ADAPTER_DIR/" --region "$AWS_REGION"

echo "=== Login to ECR ==="
ACCOUNT_ID=$(echo "$ECR_INFERENCE" | cut -d. -f1)
aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "=== Pull and run inference ==="
docker pull "$ECR_INFERENCE"
docker rm -f whiner-inference 2>/dev/null || true

docker run -d --name whiner-inference --gpus all \
  --restart unless-stopped \
  -p 8080:8080 \
  -v /opt/whiner/models/adapters:/app/models/adapters:ro \
  -e USE_MLX=false \
  -e BASE_MODEL=Qwen/Qwen3-8B \
  -e LORA_ADAPTER_PATH=/app/models/adapters/qwen3-8b-lora-v2 \
  -e MODEL_VERSION=qwen3-8b-lora-v2 \
  -e 'FALLBACK_MESSAGE=Ой лихо, моделі розгубила' \
  "$ECR_INFERENCE"

echo "Waiting for health..."
for i in $(seq 1 60); do
  if curl -sf http://localhost:8080/health >/dev/null 2>&1; then
    echo "Inference healthy!"
    curl -s http://localhost:8080/health | python3 -m json.tool
    exit 0
  fi
  sleep 10
done

echo "WARNING: health check timed out. Check: docker logs whiner-inference"
