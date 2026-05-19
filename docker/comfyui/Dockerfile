FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-dev \
    git wget curl \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 11.8 (2.4.x is the last series to ship cu118 wheels
# and the minimum required by current ComfyUI master for torch.library.custom_op)
RUN pip3 install --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu118

WORKDIR /app

RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd ComfyUI && git checkout 264b003286c731f5d219d747622643e9dd50503b

WORKDIR /app/ComfyUI

RUN pip3 install --no-cache-dir -r requirements.txt

# requirements.txt pulls in a torchaudio wheel built for cu130; override with cu118
RUN pip3 install --no-cache-dir --force-reinstall \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu118

# Install ComfyUI Manager (optional but useful)
WORKDIR /app/ComfyUI/custom_nodes
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git && \
    pip3 install --no-cache-dir -r ComfyUI-Manager/requirements.txt

# IP-Adapter Plus custom nodes (no requirements.txt — deps covered by ComfyUI base)
RUN git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git

# ControlNet preprocessors (provides CannyEdgePreprocessor and others)
RUN git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git && \
    pip3 install --no-cache-dir -r comfyui_controlnet_aux/requirements.txt

WORKDIR /app/ComfyUI

EXPOSE 8188

CMD ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--lowvram"]
