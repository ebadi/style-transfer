#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODE=${1:-video}

if [[ "$MODE" != "video" && "$MODE" != "frame" ]]; then
    echo "Usage: ./infer.sh [video|frame]"
    echo ""
    echo "  video  — AnimateDiff pipeline: entire video processed as one ComfyUI job (default)"
    echo "  frame  — Per-frame pipeline: frame-by-frame with temporal blending"
    exit 1
fi

echo "Starting inference pipeline (mode: $MODE)..."
echo "  Input:    $(grep 'video:' config.yaml | head -1 | awk '{print $2}')"
echo "  Style:    $(grep 'sample_images_dir' config.yaml | awk '{print $2}')"
echo "  Output:   $(grep 'video:' config.yaml | tail -1 | awk '{print $2}')"
echo ""

export PIPELINE_MODE="$MODE"
docker compose --profile inference up --build
