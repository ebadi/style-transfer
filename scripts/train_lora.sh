#!/usr/bin/env bash
set -euo pipefail

CONFIG=/config.yaml

# Parse config values
get_cfg() { python3 -c "import yaml,sys; c=yaml.safe_load(open('$CONFIG')); print($1)"; }

STEPS=$(get_cfg "c['lora']['training_steps']")
RANK=$(get_cfg "c['lora']['rank']")
LR=$(get_cfg "c['lora']['learning_rate']")
OUTPUT_NAME=$(get_cfg "c['lora']['output_name']")
BASE_MODEL=$(get_cfg "c['lora']['base_model_hf']")
RESOLUTION=$(get_cfg "c['lora']['resolution']")
CAPTION=$(get_cfg "c['lora']['instance_prompt']")

echo "========================================="
echo "  LoRA Training"
echo "  Base model : $BASE_MODEL"
echo "  Steps      : $STEPS"
echo "  Rank       : $RANK"
echo "  Resolution : ${RESOLUTION}x${RESOLUTION}"
echo "========================================="

# Step 1: prepare dataset
echo ""
echo "[1/2] Preparing dataset..."
python3 /scripts/prepare_dataset.py

# Remove any stale .txt files left by previous runs — the DreamBooth
# dataloader iterates all files and tries to open each as an image.
find /data/prepared_dataset -name "*.txt" -delete

# Step 2: train
echo ""
echo "[2/2] Training LoRA..."
accelerate launch \
    --config_file /accelerate_config.yaml \
    /diffusers/examples/dreambooth/train_dreambooth_lora.py \
    --pretrained_model_name_or_path="$BASE_MODEL" \
    --instance_data_dir="/data/prepared_dataset" \
    --output_dir="/tmp/lora_output" \
    --instance_prompt="$CAPTION" \
    --resolution="$RESOLUTION" \
    --train_batch_size=2 \
    --gradient_accumulation_steps=4 \
    --learning_rate="$LR" \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=100 \
    --max_train_steps="$STEPS" \
    --rank="$RANK" \
    --mixed_precision="fp16" \
    --enable_xformers_memory_efficient_attention \
    --use_8bit_adam \
    --checkpointing_steps=500 \
    --seed=42

# Copy LoRA weights to shared models volume
mkdir -p /models/loras
LORA_SRC="/tmp/lora_output/pytorch_lora_weights.safetensors"
LORA_DST="/models/loras/${OUTPUT_NAME}.safetensors"

cp "$LORA_SRC" "$LORA_DST"
echo ""
echo "Done. LoRA saved to $LORA_DST"
