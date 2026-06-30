#!/usr/bin/env bash
# Create CloudWatch log groups for ECS services.
# Usage: ./deploy/aws/scripts/setup-logs.sh

set -euo pipefail
REGION="${AWS_REGION:-eu-central-1}"

for svc in whiner-api whiner-ui whiner-prometheus whiner-grafana; do
  aws logs create-log-group --log-group-name "/ecs/${svc}" --region "$REGION" 2>/dev/null || true
  echo "Log group: /ecs/${svc}"
done

echo "Done."
