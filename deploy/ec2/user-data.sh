#!/bin/bash
# EC2 bootstrap (Amazon Linux 2023): Docker + 2 GB swap so image builds fit on a t3.micro.
set -euxo pipefail
dnf install -y docker
systemctl enable --now docker
usermod -aG docker ec2-user
if [ ! -f /swapfile ]; then
  dd if=/dev/zero of=/swapfile bs=1M count=2048
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
mkdir -p /opt/clickfraud-monitor
chown ec2-user:ec2-user /opt/clickfraud-monitor
