#!/usr/bin/env bash
# Downloads all pre-trained model weights required by the pipeline.
# Run this once before starting the containers.
# Total download size: ~8 GB

set -euo pipefail

MODELS_DIR="$(dirname "$0")/models"
mkdir -p \
    "$MODELS_DIR/checkpoints" \
    "$MODELS_DIR/controlnet" \
    "$MODELS_DIR/ipadapter" \
    "$MODELS_DIR/clip_vision" \
    "$MODELS_DIR/vae" \
    "$MODELS_DIR/loras"

HF="https://huggingface.co"

download() {
    local url="$1"
    local dest="$2"
    if [ -f "$dest" ]; then
        echo "  Already exists: $(basename "$dest")"
        return
    fi
    echo "  Downloading $(basename "$dest") ..."
    wget -q --show-progress -O "$dest" "$url"
}

# ── 1. Base checkpoint: Realistic Vision V5.1 (photorealism-focused SD 1.5 model) ──
echo "[1/5] Realistic Vision V5.1 (~2.1 GB)"
download \
    "$HF/SG161222/Realistic_Vision_V5.1_noVAE/resolve/main/Realistic_Vision_V5.1_fp16-no-ema.safetensors" \
    "$MODELS_DIR/checkpoints/realisticVisionV51.safetensors"

# ── 2. ControlNet canny (structure preservation) ──
echo "[2/5] ControlNet canny (~0.7 GB)"
download \
    "$HF/lllyasviel/control_v11p_sd15_canny/resolve/main/diffusion_pytorch_model.fp16.safetensors" \
    "$MODELS_DIR/controlnet/control_v11p_sd15_canny.safetensors"

# ── 3. IP-Adapter Plus for SD1.5 (style injection) ──
echo "[3/5] IP-Adapter Plus SD1.5 (~0.3 GB)"
download \
    "$HF/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors" \
    "$MODELS_DIR/ipadapter/ip-adapter-plus_sd15.safetensors"

# ── 4. CLIP Vision ViT-H (required by IP-Adapter) ──
echo "[4/5] CLIP Vision ViT-H (~2.4 GB)"
download \
    "$HF/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors" \
    "$MODELS_DIR/clip_vision/clip_vision_vit_h.safetensors"

# ── 5. VAE (improves colour accuracy with Realistic Vision) ──
echo "[5/5] SD VAE ft-mse (~0.3 GB)"
download \
    "$HF/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors" \
    "$MODELS_DIR/vae/vae-ft-mse-840000.safetensors"

echo ""
echo "All models downloaded to $MODELS_DIR"
echo "You can now run the pipeline with docker compose."
