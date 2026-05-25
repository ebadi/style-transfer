#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

START=$(date +%s)

docker compose --profile inference up --abort-on-container-exit --exit-code-from worker

END=$(date +%s)
ELAPSED=$(( END - START ))
printf "\nInference completed in %d min %d sec\n" $(( ELAPSED / 60 )) $(( ELAPSED % 60 ))
