#!/usr/bin/env bash
# Run ONCE on a fresh GPU EC2 instance (Amazon Linux 2023 + NVIDIA).
# Installs Docker, nvidia-container-toolkit, AWS CLI.
#
# Usage (on GPU EC2 as ec2-user):
#   curl -sSL https://raw.githubusercontent.com/.../gpu-bootstrap.sh | bash
#   OR copy this file and run: bash gpu-bootstrap.sh

set -euo pipefail

echo "=== Installing Docker ==="
sudo yum update -y
sudo yum install -y docker
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker ec2-user

echo "=== Installing NVIDIA Container Toolkit ==="
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo yum install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "=== Verify GPU ==="
nvidia-smi || echo "WARNING: nvidia-smi failed — check AMI has GPU drivers"

echo "=== Install AWS CLI v2 (if missing) ==="
if ! command -v aws &>/dev/null; then
  curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
  unzip -q /tmp/awscliv2.zip -d /tmp
  sudo /tmp/aws/install
fi

sudo mkdir -p /opt/whiner/models/adapters
echo ""
echo "Bootstrap complete. Log out and back in for docker group."
echo "Next: download model from S3 and start inference container."
