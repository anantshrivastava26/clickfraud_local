#!/bin/bash
# EC2 bootstrap (Amazon Linux 2023): Docker + 2 GB swap so image builds fit on a t3.micro.
# -e stop on error, -u error on unset variables, -x log every command (visible in the instance's
# cloud-init log), pipefail fail a pipeline if any stage fails.
set -euxo pipefail
dnf install -y docker
systemctl enable --now docker
# Let the default login user run docker without sudo (takes effect at the next login).
usermod -aG docker ec2-user
# Docker image builds need more memory than a t3.micro's 1 GB, so add a 2 GB swap file to avoid
# out-of-memory kills. The guard keeps the script safe to re-run.
if [ ! -f /swapfile ]; then
  dd if=/dev/zero of=/swapfile bs=1M count=2048
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
mkdir -p /opt/clickfraud-monitor
chown ec2-user:ec2-user /opt/clickfraud-monitor
