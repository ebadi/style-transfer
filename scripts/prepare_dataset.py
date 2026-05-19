"""
Resizes factory images for LoRA training.
Reads from /data/images, writes resized PNGs to /data/prepared_dataset.
No .txt caption files are written — the DreamBooth dataloader opens every file
in the directory as an image, so mixing in .txt files causes PIL errors.
The instance prompt is passed directly via --instance_prompt at training time.
"""

import sys
from pathlib import Path

import yaml
from PIL import Image


def center_crop_and_resize(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    short = min(w, h)
    left = (w - short) // 2
    top = (h - short) // 2
    img = img.crop((left, top, left + short, top + short))
    return img.resize((size, size), Image.LANCZOS)


def prepare_dataset(
    images_dir: str, output_dir: str, caption: str, resolution: int
) -> int:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
    images = [p for p in Path(images_dir).iterdir() if p.suffix in extensions]

    if not images:
        print(f"ERROR: no images found in {images_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(images)} images — resizing to {resolution}x{resolution}")

    for img_path in images:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  Skipping {img_path.name}: {e}")
            continue

        img = center_crop_and_resize(img, resolution)

        out_img = output_path / (img_path.stem + ".png")
        img.save(out_img, format="PNG")

    saved = len(list(output_path.glob("*.png")))
    print(f"Dataset ready: {saved} images in {output_dir}")
    return saved


if __name__ == "__main__":
    with open("/config.yaml") as f:
        cfg = yaml.safe_load(f)

    lora_cfg = cfg["lora"]
    prepare_dataset(
        images_dir="/data/images",
        output_dir="/data/prepared_dataset",
        caption=lora_cfg.get(
            "instance_prompt", "factory_style industrial warehouse photorealistic"
        ),
        resolution=int(lora_cfg.get("resolution", 512)),
    )
