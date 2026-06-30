#!/usr/bin/env bash
# Create RDS PostgreSQL for the whiner app.
# Usage: ./deploy/aws/scripts/create-rds.sh <subnet-group> <security-group-id> <password>

set -euo pipefail

SUBNET_GROUP="${1:?RDS subnet group name}"
SG_ID="${2:?Security group ID (must allow 5432 from ECS SG)}"
PASSWORD="${3:?RDS master password}"
REGION="${AWS_REGION:-eu-central-1}"

aws rds create-db-instance \
  --region "$REGION" \
  --db-instance-identifier whiner-db \
  --db-instance-class db.t4g.micro \
  --engine postgres \
  --engine-version 16 \
  --master-username postgres \
  --master-user-password "$PASSWORD" \
  --allocated-storage 20 \
  --db-name whining \
  --vpc-security-group-ids "$SG_ID" \
  --db-subnet-group-name "$SUBNET_GROUP" \
  --backup-retention-period 1 \
  --no-publicly-accessible \
  --storage-encrypted

echo "RDS creation started. Wait ~5–10 min, then:"
echo "  aws rds describe-db-instances --db-instance-identifier whiner-db --query 'DBInstances[0].Endpoint'"
