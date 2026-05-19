FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-dev \
    git wget \
    && rm -rf /var/lib/apt/lists/*

# PyTorch with CUDA 11.8
RUN pip3 install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Training dependencies
RUN pip3 install --no-cache-dir \
    diffusers[torch]==0.27.2 \
    transformers==4.38.2 \
    accelerate==0.27.2 \
    peft==0.10.0 \
    bitsandbytes==0.42.0 \
    Pillow \
    datasets \
    PyYAML \
    safetensors \
    scipy \
    ftfy \
    tensorboard

RUN pip3 install --no-cache-dir xformers==0.0.23.post1 \
    --index-url https://download.pytorch.org/whl/cu118

# Clone diffusers for training scripts
RUN git clone --depth 1 --branch v0.27.2 \
    https://github.com/huggingface/diffusers.git /diffusers && \
    pip3 install --no-cache-dir -e "/diffusers[torch]"

# Pin versions that transitive deps may have upgraded to incompatible releases:
#   huggingface_hub>=0.24 removed cached_download used by diffusers v0.27.2
#   numpy 2.x breaks torch 2.1.2 which was compiled against numpy 1.x
RUN pip3 install --no-cache-dir --force-reinstall \
    "huggingface_hub==0.23.4" \
    "numpy<2"

RUN mkdir -p /scripts /data/images /data/prepared_dataset /models/loras

# Write accelerate config to an explicit path passed via --config_file.
# `accelerate config default` runs at build time without a GPU and writes
# use_cpu: true — using --config_file bypasses that entirely.
RUN cat > /accelerate_config.yaml << 'EOF'
compute_environment: LOCAL_MACHINE
distributed_type: NO
downcast_bf16: no
gpu_ids: '0'
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

CMD ["bash", "/scripts/train_lora.sh"]
