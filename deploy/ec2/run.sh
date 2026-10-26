#!/bin/bash
# (Re)start the stack on the EC2 host. Everything is prefixed clickfraud- so it
# can share a host with other projects.
#
# Two modes:
#   BACKEND_IMAGE / FRONTEND_IMAGE set  -> pull those images (used by GitHub Actions via SSM)
#   unset                               -> build from the source tree next to this script
# Stop on the first error (-e) or unset variable (-u), and fail a pipeline if any stage fails.
set -euo pipefail

NET=clickfraud-net
HTTP_PORT=${HTTP_PORT:-80}

if [[ -n "${BACKEND_IMAGE:-}" && -n "${FRONTEND_IMAGE:-}" ]]; then
  # The registry host is everything before the first '/' of the image name. The instance role
  # provides a short-lived ECR token for the login below.
  REGISTRY=${BACKEND_IMAGE%%/*}
  REGION=${AWS_REGION:-ap-south-1}
  aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"
  docker pull "$BACKEND_IMAGE"
  docker pull "$FRONTEND_IMAGE"
else
  # Source mode: build from the repository checkout this script lives in.
  cd "$(dirname "$0")/../.."
  BACKEND_IMAGE=clickfraud-backend:latest
  FRONTEND_IMAGE=clickfraud-frontend:latest
  docker build -t "$BACKEND_IMAGE" backend
  docker build -t "$FRONTEND_IMAGE" frontend
fi

# Both containers join one user-defined network, so the frontend reaches the backend by container
# name (clickfraud-backend:8000).
docker network inspect $NET >/dev/null 2>&1 || docker network create $NET
# Replace the running containers. This means a brief gap in service while they restart, in exchange
# for a very simple rollout.
docker rm -f clickfraud-backend clickfraud-frontend >/dev/null 2>&1 || true

# Backend: publishes no host port, so only the frontend container can reach it. --restart
# unless-stopped brings both containers back after an instance reboot.
docker run -d --name clickfraud-backend --network $NET --restart unless-stopped \
  -e ALLOWED_ORIGINS="*" \
  "$BACKEND_IMAGE"

# Frontend (nginx): the only published port. It serves the UI and proxies /api to the backend.
docker run -d --name clickfraud-frontend --network $NET --restart unless-stopped \
  -e BACKEND_URL=http://clickfraud-backend:8000 \
  -p "$HTTP_PORT":80 \
  "$FRONTEND_IMAGE"

# The t3.micro has little disk: delete unused images older than three days. Never fatal.
docker image prune -af --filter "until=72h" >/dev/null || true
docker ps --filter name=clickfraud-
