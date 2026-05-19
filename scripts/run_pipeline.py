"""
Orchestrates the full style-transfer pipeline:
  1. Extract frames from the Gazebo video
  2. For each frame: submit a ComfyUI job (ControlNet + IP-Adapter + LoRA)
  3. Reassemble the stylised frames into an output video at the original resolution
"""

import copy
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests
import yaml
from tqdm import tqdm


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────


def load_config(path="/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# Video helpers (ffprobe / ffmpeg)
# ──────────────────────────────────────────────


def get_video_info(video_path: str) -> tuple[int, int, float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "json",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    stream = json.loads(result.stdout)["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    fps = int(num) / int(den)
    return stream["width"], stream["height"], fps


def extract_frames(video_path: str, frames_dir: str, fps: float) -> list[Path]:
    Path(frames_dir).mkdir(parents=True, exist_ok=True)
    pattern = str(Path(frames_dir) / "frame_%06d.png")
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-vf", f"fps={fps:.6f}", pattern, "-y"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return sorted(Path(frames_dir).glob("frame_*.png"))


def assemble_video(
    frames_dir: str,
    output_path: str,
    fps: float,
    orig_w: int,
    orig_h: int,
    codec: str,
    crf: int,
):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pattern = str(Path(frames_dir) / "styled_%06d.png")
    subprocess.run(
        [
            "ffmpeg",
            "-framerate",
            f"{fps:.6f}",
            "-i",
            pattern,
            "-vf",
            f"scale={orig_w}:{orig_h}:flags=lanczos",
            "-c:v",
            codec,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            output_path,
            "-y",
        ],
        check=True,
    )


# ──────────────────────────────────────────────
# Resolution helpers
# ──────────────────────────────────────────────


def compute_proc_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    """Scale so the longer side = max_side, both dims divisible by 8."""
    if max(width, height) <= max_side:
        return (width // 8) * 8, (height // 8) * 8
    scale = max_side / max(width, height)
    return (int(width * scale) // 8) * 8, (int(height * scale) // 8) * 8


# ──────────────────────────────────────────────
# ComfyUI API helpers
# ──────────────────────────────────────────────


def wait_for_comfyui(url: str, timeout: int = 180):
    print(f"Waiting for ComfyUI at {url} ...", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/system_stats", timeout=5)
            if r.status_code == 200:
                print("ComfyUI is ready.")
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(3)
    raise RuntimeError(f"ComfyUI did not become ready within {timeout}s")


def submit_workflow(url: str, workflow: dict, client_id: str) -> str:
    payload = {"prompt": workflow, "client_id": client_id}
    r = requests.post(f"{url}/prompt", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["prompt_id"]


def wait_for_job(url: str, prompt_id: str, timeout: int = 600) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{url}/history/{prompt_id}", timeout=10)
        if r.status_code == 200:
            history = r.json()
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    msgs = status.get("messages", [])
                    raise RuntimeError(f"ComfyUI job failed: {msgs}")
                if entry.get("outputs"):
                    return entry
        time.sleep(2)
    raise TimeoutError(f"Job {prompt_id} did not complete within {timeout}s")


def download_output(url: str, history_entry: dict, dest: Path):
    outputs = history_entry.get("outputs", {})
    for node_output in outputs.values():
        for image in node_output.get("images", []):
            r = requests.get(
                f"{url}/view",
                params={
                    "filename": image["filename"],
                    "subfolder": image.get("subfolder", ""),
                    "type": "output",
                },
                timeout=60,
            )
            r.raise_for_status()
            dest.write_bytes(r.content)
            return
    raise RuntimeError("No images found in ComfyUI output for job")


# ──────────────────────────────────────────────
# Workflow patching
# ──────────────────────────────────────────────


def patch_workflow(
    base_workflow: dict,
    frame_name: str,
    style_name: str,
    proc_w: int,
    proc_h: int,
    orig_w: int,
    orig_h: int,
    cfg: dict,
) -> dict:
    wf = copy.deepcopy(base_workflow)
    lora_name = f"{cfg['lora']['output_name']}.safetensors"
    st = cfg["style_transfer"]

    # Model / LoRA
    wf["1"]["inputs"]["ckpt_name"] = st["base_model"]
    wf["2"]["inputs"]["lora_name"] = lora_name
    wf["2"]["inputs"]["strength_model"] = st["lora_strength"]
    wf["2"]["inputs"]["strength_clip"] = st["lora_strength"]

    # Prompts
    wf["3"]["inputs"]["text"] = st["prompt"]
    wf["4"]["inputs"]["text"] = st["negative_prompt"]

    # Input frame → scale to processing resolution
    wf["6"]["inputs"]["image"] = frame_name
    wf["7"]["inputs"]["width"] = proc_w
    wf["7"]["inputs"]["height"] = proc_h
    wf["8"]["inputs"]["resolution"] = max(proc_w, proc_h)

    # Style reference
    wf["11"]["inputs"]["image"] = style_name
    wf["12"]["inputs"]["weight"] = st["ipadapter_weight"]

    # ControlNet
    wf["13"]["inputs"]["strength"] = st["controlnet_strength"]

    # Sampler
    wf["15"]["inputs"]["denoise"] = st["denoising_strength"]
    wf["15"]["inputs"]["steps"] = st["steps"]
    wf["15"]["inputs"]["cfg"] = st["cfg"]
    wf["15"]["inputs"]["sampler_name"] = st["sampler"]
    wf["15"]["inputs"]["scheduler"] = st["scheduler"]
    wf["15"]["inputs"]["seed"] = 42

    # Output scale → back to original resolution
    wf["17"]["inputs"]["width"] = orig_w
    wf["17"]["inputs"]["height"] = orig_h

    # Output filename (no extension — ComfyUI appends counter + .png)
    wf["18"]["inputs"]["filename_prefix"] = "styled_out"

    return wf


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main():
    cfg = load_config()
    comfyui_url = os.getenv("COMFYUI_URL", "http://localhost:8188")
    comfyui_input_dir = Path(os.getenv("COMFYUI_INPUT_DIR", "/app/ComfyUI/input"))
    comfyui_output_dir = Path(os.getenv("COMFYUI_OUTPUT_DIR", "/app/ComfyUI/output"))

    input_video = cfg["input"]["video"]
    factory_dir = cfg["input"]["sample_images_dir"]
    output_video = cfg["output"]["video"]

    # ── pick a single style reference image ───────────────────────────────
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    style_images = sorted(p for p in Path(factory_dir).iterdir() if p.suffix in exts)
    if not style_images:
        sys.exit(f"No factory images found in {factory_dir}")

    ref_name = cfg["style_transfer"].get("style_reference")
    if ref_name:
        style_ref_image = Path(factory_dir) / ref_name
        if not style_ref_image.exists():
            sys.exit(f"style_reference '{ref_name}' not found in {factory_dir}")
    else:
        style_ref_image = style_images[0]
    print(f"Style reference: {style_ref_image.name}  ({len(style_images)} images available)")

    # ── video info ─────────────────────────────────────────────────────────
    orig_w, orig_h, fps = get_video_info(input_video)
    if cfg["video"]["fps"] != "auto":
        fps = float(cfg["video"]["fps"])
    print(f"Input: {orig_w}x{orig_h} @ {fps:.2f} fps  ({input_video})")

    # ── processing resolution ──────────────────────────────────────────────
    max_res = cfg["style_transfer"]["processing_resolution"]
    if str(max_res).lower() == "original":
        proc_w, proc_h = (orig_w // 8) * 8, (orig_h // 8) * 8
    else:
        proc_w, proc_h = compute_proc_size(orig_w, orig_h, int(max_res))
    print(f"Processing at: {proc_w}x{proc_h}")

    # ── extract frames ─────────────────────────────────────────────────────
    frames_in_dir = "/tmp/frames_input"
    frames_out_dir = "/tmp/frames_output"
    Path(frames_out_dir).mkdir(parents=True, exist_ok=True)

    print("Extracting frames...")
    frames = extract_frames(input_video, frames_in_dir, fps)
    print(f"Extracted {len(frames)} frames")

    # ── wait for ComfyUI ───────────────────────────────────────────────────
    wait_for_comfyui(comfyui_url)

    # ── load base workflow ─────────────────────────────────────────────────
    with open("/workflows/style_transfer.json") as f:
        base_workflow = json.load(f)

    client_id = str(uuid.uuid4())

    # Upload the style reference once — same image reused for every frame
    style_name = f"style_ref{style_ref_image.suffix}"
    shutil.copy2(style_ref_image, comfyui_input_dir / style_name)

    # ── process frames ─────────────────────────────────────────────────────
    print(f"\nProcessing {len(frames)} frames (this is the slow part)...")
    failed = []

    for idx, frame_path in enumerate(tqdm(frames, unit="frame")):
        frame_name = f"in_{idx:06d}.png"

        shutil.copy2(frame_path, comfyui_input_dir / frame_name)

        workflow = patch_workflow(
            base_workflow,
            frame_name,
            style_name,
            proc_w,
            proc_h,
            orig_w,
            orig_h,
            cfg,
        )

        try:
            prompt_id = submit_workflow(comfyui_url, workflow, client_id)
            history = wait_for_job(comfyui_url, prompt_id)
            out_path = Path(frames_out_dir) / f"styled_{idx:06d}.png"
            download_output(comfyui_url, history, out_path)
        except Exception as e:
            print(f"\n  Frame {idx} failed: {e} — copying original as fallback")
            shutil.copy2(frame_path, Path(frames_out_dir) / f"styled_{idx:06d}.png")
            failed.append(idx)

    if failed:
        print(
            f"\nWarning: {len(failed)} frames fell back to original: {failed[:10]}{'...' if len(failed) > 10 else ''}"
        )

    # ── assemble output video ──────────────────────────────────────────────
    print("\nAssembling output video...")
    assemble_video(
        frames_out_dir,
        output_video,
        fps,
        orig_w,
        orig_h,
        cfg["video"]["codec"],
        cfg["video"]["quality"],
    )
    print(f"\nDone. Output saved to: {output_video}")


if __name__ == "__main__":
    main()
