#!/usr/bin/env bash
# Runs 4 inference jobs back-to-back, each with one parameter changed from baseline.
# ComfyUI starts once and stays running for all tests.
# Outputs saved to data/output/test_<name>.mp4
set -euo pipefail

cd "$(dirname "$0")"

# Backup original config — always restored on exit
cp config.yaml config.yaml.bak
trap 'cp config.yaml.bak config.yaml && rm -f config.yaml.bak' EXIT

# Start ComfyUI once (rebuild only if image changed)
echo "Starting ComfyUI..."
docker compose up -d --build comfyui

echo "Waiting for ComfyUI to be healthy..."
until [ "$(docker inspect --format='{{.State.Health.Status}}' styletransfer_comfyui 2>/dev/null)" = "healthy" ]; do
    echo -n "."; sleep 5
done
echo " ready."
echo ""

# Apply one change on top of the baseline config and set the output filename
apply_and_run() {
    local name="$1"
    local py_patch="$2"
    local output="/data/output/test_${name}.mp4"

    echo "=== Test: ${name} ==="

    cp config.yaml.bak config.yaml
    python3 - <<EOF
import yaml
with open('config.yaml') as f:
    c = yaml.safe_load(f)
${py_patch}
c['output']['video'] = '${output}'
with open('config.yaml', 'w') as f:
    yaml.dump(c, f, sort_keys=False)
EOF

    docker compose --profile inference run --no-deps --rm worker
    echo "Saved: ${output}"
    echo ""
}

apply_and_run "denoise_0.45"      "c['style_transfer']['denoising_strength'] = 0.45"
apply_and_run "controlnet_0.90"   "c['style_transfer']['controlnet_strength'] = 0.90"
apply_and_run "ipadapter_0.40"    "c['style_transfer']['ipadapter_weight'] = 0.40"
apply_and_run "sampler_dpmpp2m"   "c['style_transfer']['sampler'] = 'dpmpp_2m'"

echo "All tests complete. Results:"
ls -lh data/output/test_*.mp4 2>/dev/null || echo "  (no output files found)"
