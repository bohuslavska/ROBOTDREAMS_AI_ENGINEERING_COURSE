#!/usr/bin/env bash
# Fuse MLX LoRA adapter into merged weights and upload to S3.
# Run on your Mac BEFORE deploying to AWS.
#
# Usage: ./deploy/aws/scripts/export-model.sh [S3_BUCKET]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

S3_BUCKET="${1:-}"
ADAPTER="models/adapters/qwen3-8b-lora-v2"
MERGED="models/merged/qwen3-8b-lora-v2"

echo "=== Step 1: Fuse MLX adapter ==="
if [[ ! -f "$ADAPTER/adapters.safetensors" ]]; then
  echo "ERROR: $ADAPTER/adapters.safetensors not found. Train v2 first."
  exit 1
fi

source .venv/bin/activate
python -m mlx_lm.fuse \
  --model mlx-community/Qwen3-8B-4bit \
  --adapter-path "$ADAPTER" \
  --save-path "$MERGED"

echo ""
echo "=== Step 2: Upload to S3 ==="
if [[ -z "$S3_BUCKET" ]]; then
  echo "Merged model saved locally at: $MERGED"
  echo "Upload manually:"
  echo "  aws s3 mb s3://YOUR-BUCKET --region eu-central-1"
  echo "  aws s3 sync $ADAPTER/ s3://YOUR-BUCKET/adapters/qwen3-8b-lora-v2/"
  echo "  aws s3 sync $MERGED/   s3://YOUR-BUCKET/merged/qwen3-8b-lora-v2/"
  exit 0
fi

aws s3 mb "s3://${S3_BUCKET}" 2>/dev/null || true
aws s3 sync "$ADAPTER/" "s3://${S3_BUCKET}/adapters/qwen3-8b-lora-v2/"
aws s3 sync "$MERGED/"   "s3://${S3_BUCKET}/merged/qwen3-8b-lora-v2/"

echo ""
echo "Done. Model artifacts in s3://${S3_BUCKET}/"
