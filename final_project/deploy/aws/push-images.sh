#!/usr/bin/env bash
# Build and push Docker images to AWS ECR
# Usage: ./deploy/aws/push-images.sh <aws-account-id> <aws-region>

set -euo pipefail

ACCOUNT_ID="${1:?AWS account ID required}"
REGION="${2:-eu-central-1}"
REPO_PREFIX="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/whiner"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "Logging in to ECR..."
aws ecr get-login-password --region "$REGION" | \
  docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

for repo in api ui inference; do
  aws ecr describe-repositories --repository-names "whiner-${repo}" --region "$REGION" 2>/dev/null || \
    aws ecr create-repository --repository-name "whiner-${repo}" --region "$REGION"
done

echo "Building images..."
docker build -f Dockerfile.api -t whiner-api .
docker build -f Dockerfile.ui -t whiner-ui .
docker build -f Dockerfile.inference -t whiner-inference .

docker tag whiner-api       "${REPO_PREFIX}-api:latest"
docker tag whiner-ui        "${REPO_PREFIX}-ui:latest"
docker tag whiner-inference "${REPO_PREFIX}-inference:latest"

docker push "${REPO_PREFIX}-api:latest"
docker push "${REPO_PREFIX}-ui:latest"
docker push "${REPO_PREFIX}-inference:latest"

echo "Done. Images pushed:"
echo "  ${REPO_PREFIX}-api:latest"
echo "  ${REPO_PREFIX}-ui:latest"
echo "  ${REPO_PREFIX}-inference:latest"
