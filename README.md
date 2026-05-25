# Sim-to-Real Style Transfer

Converts a Gazebo simulator video into photorealistic factory footage using your own factory images as a style reference.

## Prerequisites

- Docker and Docker Compose (v2)
- NVIDIA Container Toolkit
- NVIDIA GPU with ≥ 8 GB VRAM
- ~20 GB free disk space

## Usage

### 1. Place your data

```
data/
├── input/
│   └── your_video.mp4        ← Gazebo recording (MP4 or AVI)
└── sample_images/
    └── *.jpg / *.png         ← real factory photos
```

Update `config.yaml` with the correct video filename under `input.video`.

### 2. Download pre-trained models

```bash
chmod +x *.sh
./download_models.sh
```

### 3. Build Docker images

```bash
docker compose build
```

### 4. Train the LoRA

```bash
./train.sh
```

Trains a LoRA on your factory images. Output: `models/loras/factory_lora.safetensors`.

> **Note:** With the default 1500-step config, training takes ~2 hours on an Intel Core i9-10900KF + NVIDIA GeForce RTX 2070 SUPER (8 GB VRAM) with 32 GB RAM.

### 5. Run style transfer

```bash
./infer.sh
```

Output video: `data/output/stylized_video.mp4`.
