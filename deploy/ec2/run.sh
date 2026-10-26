#!/bin/bash
# (Re)start the stack on the EC2 host. Everything is prefixed clickfraud- so it
# can share a host with other projects.
#
# Two modes:
#   BACKEND_IMAGE / FRONTEND_IMAGE set  -> pull those images (used by GitHub Actions via SSM)
#   unset                               -> build from the source tree next to this script
set -euo pipefail

NET=clickfraud-net
HTTP_PORT=${HTTP_PORT:-80}

if [[ -n "${BACKEND_IMAGE:-}" && -n "${FRONTEND_IMAGE:-}" ]]; then
  REGISTRY=${BACKEND_IMAGE%%/*}
  REGION=${AWS_REGION:-ap-south-1}
  aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"
  docker pull "$BACKEND_IMAGE"
  docker pull "$FRONTEND_IMAGE"
else
  cd "$(dirname "$0")/../.."
  BACKEND_IMAGE=clickfraud-backend:latest
  FRONTEND_IMAGE=clickfraud-frontend:latest
  docker build -t "$BACKEND_IMAGE" backend
  docker build -t "$FRONTEND_IMAGE" frontend
fi

docker network inspect $NET >/dev/null 2>&1 || docker network create $NET
docker rm -f clickfraud-backend clickfraud-frontend >/dev/null 2>&1 || true

docker run -d --name clickfraud-backend --network $NET --restart unless-stopped \
  -e ALLOWED_ORIGINS="*" \
  "$BACKEND_IMAGE"

docker run -d --name clickfraud-frontend --network $NET --restart unless-stopped \
  -e BACKEND_URL=http://clickfraud-backend:8000 \
  -p "$HTTP_PORT":80 \
  "$FRONTEND_IMAGE"

docker image prune -af --filter "until=72h" >/dev/null || true
docker ps --filter name=clickfraud-
