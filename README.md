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
chmod +x download_models.sh
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

### 5. Run style transfer

```bash
docker compose --profile inference up
```

Output video: `data/output/stylized_video.mp4`.

## Configuration

All parameters are in `config.yaml`. Key settings:

| Parameter | Effect |
|---|---|
| `lora.training_steps` | More steps = stronger style learning |
| `style_transfer.style_reference` | Filename of the factory image to use as style reference (null = first image found) |
| `style_transfer.denoising_strength` | Higher = more stylization, less geometric fidelity |
| `style_transfer.controlnet_strength` | Higher = closer to original Gazebo geometry |
| `style_transfer.ipadapter_weight` | Higher = stronger colour/texture from the reference image |
