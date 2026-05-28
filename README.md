# Sim-to-Real Style Transfer

Converts Gazebo simulator video into photorealistic factory footage using your own factory images as a style reference.

Two inference pipelines are available and share the same training, Docker containers, and configuration:

| | Video pipeline | Image pipeline |
|---|---|---|
| **How it works** | Entire video processed as one AnimateDiff job inside ComfyUI | Frames extracted by ffmpeg, each submitted as a separate ComfyUI job, reassembled |
| **Temporal consistency** | AnimateDiff motion module (temporal attention across frames) | Python temporal blending (α carry-over between frames) |
| **Speed** | One job — faster end-to-end | One job per frame — slower |
| **Best for** | Smooth, coherent video output | Fine-grained per-frame control |
| **Invoke** | `./infer.sh video` | `./infer.sh frame` |

---

## Prerequisites

- Docker and Docker Compose v2
- NVIDIA Container Toolkit
- NVIDIA GPU with ≥ 8 GB VRAM
- ~20 GB free disk space

---

## Setup

### 1. Place your data

```
data/
├── input/
│   └── your_video.mp4        ← Gazebo recording (MP4 or AVI)
└── sample_images/
    └── *.jpg / *.png         ← real factory reference photos
```

Update `input.video` in `config.yaml` to match your video filename.

### 2. Download pre-trained models (~11 GB)

```bash
chmod +x *.sh
./download_models.sh
```

Downloads: Realistic Vision V5.1, ControlNet canny, IP-Adapter Plus, CLIP Vision ViT-H, VAE, AnimateDiff v2 motion module.

---

## Usage

### Train the LoRA  *(shared by both pipelines)*

Learns your factory style from the images in `data/sample_images/`:

```bash
./train.sh
```

Output: `models/loras/factory_lora.safetensors`

Training is optional — the pipeline still runs without it (set `lora_strength: 0.0` in `config.yaml` to disable).

---

### Run inference

```bash
./infer.sh video    # AnimateDiff video pipeline (default)
./infer.sh frame    # per-frame pipeline
./infer.sh          # same as video
```

Output: `data/output/stylized_video.mp4`

---

## Configuration (`config.yaml`)

### Shared parameters

| Key | Description |
|---|---|
| `input.video` | Path to Gazebo input video |
| `input.sample_images_dir` | Directory of factory reference photos |
| `style_transfer.lora_strength` | How strongly the trained LoRA is applied (0–1) |
| `style_transfer.controlnet_strength` | Structure preservation strength (0–1) |
| `style_transfer.ipadapter_weight` | Runtime style injection from reference image (0–1) |
| `style_transfer.denoising_strength` | img2img strength — higher = more style, less structure |
| `style_transfer.sampler` | Diffusion sampler (`dpmpp_2m` recommended) |
| `style_transfer.processing_resolution` | Working resolution in pixels |

### Video pipeline only

| Key | Description |
|---|---|
| `animatediff.motion_module` | AnimateDiff motion module filename |
| `animatediff.context_length` | Frames per temporal window (default: 16) |
| `animatediff.context_overlap` | Overlap between windows — increase to reduce seam artifacts |

### Image pipeline only

| Key | Description |
|---|---|
| `style_transfer.temporal_blend_alpha` | Frame carry-over factor to reduce flicker (0 = off, 0.15 recommended) |
| `video.codec` | Output codec (default: `libx264`) |
| `video.quality` | CRF quality — lower is better (18 = high quality) |

---

## Parameter testing

`test_params.sh` runs four inference jobs back-to-back with individual parameter variations, saving each result with a descriptive filename:

```bash
./test_params.sh
```

Output files: `data/output/test_denoise_0.45.mp4`, `test_controlnet_0.90.mp4`, etc.

---

## Project structure

```
├── config.yaml                  # all pipeline configuration
├── train.sh                     # LoRA training entry point
├── infer.sh                     # inference entry point (accepts video|frame)
├── download_models.sh           # one-time model download
├── test_params.sh               # parameter sweep utility
│
├── docker/
│   ├── comfyui/                 # ComfyUI + custom nodes (AnimateDiff, VHS, IP-Adapter, ControlNet)
│   ├── lora-trainer/            # DreamBooth LoRA training container
│   └── worker/                  # lightweight orchestration container
│
├── workflows/
│   ├── style_transfer_video.json    # video pipeline workflow (AnimateDiff + VHS)
│   └── style_transfer_frame.json   # frame pipeline workflow (per-frame)
│
├── scripts/
│   ├── run_pipeline.py          # unified pipeline script (mode via PIPELINE_MODE)
│   ├── train_lora.sh            # DreamBooth training script
│   └── prepare_dataset.py       # image preprocessing for training
│
├── models/                      # model weights (populated by download_models.sh + train.sh)
│   ├── checkpoints/
│   ├── loras/
│   ├── controlnet/
│   ├── ipadapter/
│   ├── clip_vision/
│   ├── vae/
│   └── animatediff_models/
│
└── data/
    ├── input/                   # input Gazebo video
    ├── sample_images/           # factory reference photos
    └── output/                  # stylized output videos
```
