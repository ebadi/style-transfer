#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "Stopping running containers..."
docker compose --profile training down --remove-orphans 2>/dev/null || true

echo "Building lora-trainer image..."
docker compose --profile training build lora-trainer

echo "Starting training..."
docker compose --profile training up lora-trainer
