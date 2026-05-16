#!/usr/bin/env bash
set -euo pipefail

command="${1:-up}"
image="${TURBOSEARCH_POSTGRES_IMAGE:-turbosearch-postgres:pg18}"
container="${TURBOSEARCH_POSTGRES_CONTAINER:-turbosearch-postgres}"
port="${TURBOSEARCH_POSTGRES_PORT:-55432}"
database="${TURBOSEARCH_POSTGRES_DB:-turbosearch}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

build_image() {
  docker build \
    --file "${repo_root}/docker/postgres/Dockerfile" \
    --tag "${image}" \
    "${repo_root}"
}

wait_ready() {
  for _ in $(seq 1 60); do
    if docker exec "${container}" pg_isready -U postgres -d "${database}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done

  docker logs "${container}"
  echo "Postgres did not become ready in time." >&2
  return 1
}

case "${command}" in
  build)
    build_image
    ;;
  up)
    build_image
    if docker container inspect "${container}" >/dev/null 2>&1; then
      docker start "${container}" >/dev/null
    else
      docker run \
        --detach \
        --name "${container}" \
        --env POSTGRES_PASSWORD=postgres \
        --env POSTGRES_DB="${database}" \
        --publish "127.0.0.1:${port}:5432" \
        "${image}" >/dev/null
    fi
    wait_ready
    echo "postgres://postgres:postgres@127.0.0.1:${port}/${database}"
    ;;
  down)
    docker rm --force "${container}" >/dev/null 2>&1 || true
    ;;
  logs)
    docker logs "${container}"
    ;;
  url)
    echo "postgres://postgres:postgres@127.0.0.1:${port}/${database}"
    ;;
  *)
    echo "usage: scripts/postgres-db.sh [build|up|down|logs|url]" >&2
    exit 1
    ;;
esac
