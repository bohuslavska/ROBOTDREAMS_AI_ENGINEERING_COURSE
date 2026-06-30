#!/usr/bin/env bash
# Create a submission zip excluding heavy/regeneratable files.
#
# Usage:
#   ./scripts/pack-submission.sh           # include final adapter (~100 MB)
#   ./scripts/pack-submission.sh --no-model  # code + data only (~15 MB)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

INCLUDE_MODEL=true
if [[ "${1:-}" == "--no-model" ]]; then
  INCLUDE_MODEL=false
fi

OUT="ukrainian-literary-whiner-submission.zip"
STAGING=$(mktemp -d)
trap 'rm -rf "$STAGING"' EXIT

NAME="ukrainian-literary-whiner"
DEST="$STAGING/$NAME"
mkdir -p "$DEST"

echo "Staging project..."

# Core code & config
rsync -a \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  app/ deploy/ notebooks/ data/ \
  "$DEST/"

# Root files
cp README.md PROJECT_REPORT.md SUBMISSION.md \
   docker-compose.yml docker-compose.prod.yml \
   Dockerfile.api Dockerfile.ui Dockerfile.inference Dockerfile.prometheus \
   requirements-api.txt requirements-ui.txt requirements-inference.txt \
   .env.example .env.test.local mlx_train_v2.yaml \
   "$DEST/" 2>/dev/null || true

# Models: final adapter only (skip intermediate checkpoints)
mkdir -p "$DEST/models/adapters/qwen3-8b-lora-v2"
if $INCLUDE_MODEL; then
  if [[ -f models/adapters/qwen3-8b-lora-v2/adapters.safetensors ]]; then
    cp models/adapters/qwen3-8b-lora-v2/adapters.safetensors \
       models/adapters/qwen3-8b-lora-v2/adapter_config.json \
       "$DEST/models/adapters/qwen3-8b-lora-v2/"
    echo "Included final adapter (~74 MB)"
  else
    echo "WARNING: adapters.safetensors not found — skipping model"
  fi
else
  mkdir -p "$DEST/models/adapters"
  touch "$DEST/models/adapters/.gitkeep"
  echo "Skipped model (--no-model)"
fi

# Clean notebook junk
rm -rf "$DEST/notebooks/pluperfect_grac" \
       "$DEST/notebooks/generation_checkpoints" \
       "$DEST/notebooks/data" \
       "$DEST/notebooks/.DS_Store" 2>/dev/null || true

# Remove duplicate csv in notebooks if same as data/
rm -f "$DEST/notebooks/modernized_training_pairs_flat.csv" \
      "$DEST/notebooks/sad_sentences.csv" 2>/dev/null || true

# Strip secrets from notebooks (best effort — user should verify)
# Do not copy .env

cd "$STAGING"
zip -rq "$ROOT/$OUT" "$NAME"

SIZE=$(du -sh "$ROOT/$OUT" | cut -f1)
echo ""
echo "Created: $ROOT/$OUT ($SIZE)"
echo "Verify no secrets: unzip -l $OUT | grep -E '\\.env$'  (should be .env.example only)"
