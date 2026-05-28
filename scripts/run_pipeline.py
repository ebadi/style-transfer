"""
Unified style-transfer pipeline.
Select mode via PIPELINE_MODE env var (default: video):

  video — AnimateDiff pipeline: entire video processed as a temporal batch via
          ComfyUI VideoHelperSuite + AnimateDiff. One ComfyUI job per run.

  image — Per-frame pipeline: ffmpeg extracts frames, each frame is submitted
          as a separate ComfyUI job, temporal blending reduces flicker, ffmpeg
          reassembles the output video.
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


# ── Config ───────────────────────────────────────────────────────────────────

def load_config(path="/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ── ComfyUI API ───────────────────────────────────────────────────────────────

def wait_for_comfyui(url: str, timeout: int = 180):
    print(f"Waiting for ComfyUI at {url} ...", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{url}/system_stats", timeout=5).status_code == 200:
                print("ComfyUI is ready.")
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(3)
    raise RuntimeError(f"ComfyUI did not become ready within {timeout}s")


def load_workflow(url: str, path: str) -> dict:
    """Load a workflow file. If it is in LiteGraph graph format, convert to API format via
    the /workflow/convert endpoint (added by comfyui-workflow-to-api-converter-endpoint)."""
    with open(path) as f:
        data = json.load(f)
    if "nodes" in data and "links" in data:
        r = requests.post(f"{url}/workflow/convert", json=data, timeout=30)
        r.raise_for_status()
        return r.json()
    return data


def submit_workflow(url: str, workflow: dict, client_id: str) -> str:
    r = requests.post(f"{url}/prompt", json={"prompt": workflow, "client_id": client_id}, timeout=30)
    r.raise_for_status()
    return r.json()["prompt_id"]


def wait_for_job(url: str, prompt_id: str, timeout: int = 7200) -> dict:
    deadline = time.time() + timeout
    last_dot = time.time()
    while time.time() < deadline:
        r = requests.get(f"{url}/history/{prompt_id}", timeout=10)
        if r.status_code == 200:
            history = r.json()
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI job failed: {status.get('messages', [])}")
                if entry.get("outputs"):
                    print()
                    return entry
        if time.time() - last_dot >= 30:
            print(".", end="", flush=True)
            last_dot = time.time()
        time.sleep(2)
    raise TimeoutError(f"Job {prompt_id} did not complete within {timeout}s")


# ── Shared helpers ────────────────────────────────────────────────────────────

def get_video_info(video_path: str) -> tuple[int, int, float]:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    stream = json.loads(result.stdout)["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    return stream["width"], stream["height"], int(num) / int(den)


def override_fps(raw_fps: float, cfg: dict) -> float:
    return float(cfg["video"]["fps"]) if cfg["video"]["fps"] != "auto" else raw_fps


def select_style_ref(factory_dir: Path, cfg: dict) -> Path:
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    style_images = sorted(p for p in factory_dir.iterdir() if p.suffix in exts)
    if not style_images:
        sys.exit(f"No factory images found in {factory_dir}")
    ref_name  = cfg["style_transfer"].get("style_reference")
    style_ref = (factory_dir / ref_name) if ref_name else style_images[0]
    if ref_name and not style_ref.exists():
        sys.exit(f"style_reference '{ref_name}' not found in {factory_dir}")
    print(f"Style reference: {style_ref.name}  ({len(style_images)} images available)")
    return style_ref


def download_output(url: str, history_entry: dict, dest: Path, timeout: int = 60):
    for node_output in history_entry.get("outputs", {}).values():
        for image in node_output.get("images", []):
            r = requests.get(
                f"{url}/view",
                params={"filename": image["filename"], "subfolder": image.get("subfolder", ""), "type": "output"},
                timeout=timeout,
            )
            r.raise_for_status()
            dest.write_bytes(r.content)
            return
    raise RuntimeError("No output found in ComfyUI job output")


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE PIPELINE  (per-frame, with temporal blending)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_frames(video_path: str, frames_dir: str, fps: float) -> list[Path]:
    Path(frames_dir).mkdir(parents=True, exist_ok=True)
    pattern = str(Path(frames_dir) / "frame_%06d.png")
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-vf", f"fps={fps:.6f}", pattern, "-y"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return sorted(Path(frames_dir).glob("frame_*.png"))


def assemble_video(frames_dir: str, output_path: str, fps: float,
                   orig_w: int, orig_h: int, codec: str, crf: int):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-framerate", f"{fps:.6f}",
            "-i", str(Path(frames_dir) / "styled_%06d.png"),
            "-vf", f"scale={orig_w}:{orig_h}:flags=lanczos",
            "-c:v", codec, "-crf", str(crf), "-pix_fmt", "yuv420p",
            output_path, "-y",
        ],
        check=True,
    )


def compute_proc_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    if max(width, height) <= max_side:
        return (width // 8) * 8, (height // 8) * 8
    scale = max_side / max(width, height)
    return (int(width * scale) // 8) * 8, (int(height * scale) // 8) * 8



def patch_workflow_image(wf: dict, frame_name: str, style_name: str,
                         proc_w: int, proc_h: int, orig_w: int, orig_h: int,
                         cfg: dict) -> dict:
    wf = copy.deepcopy(wf)
    st = cfg["style_transfer"]

    wf["1"]["inputs"]["ckpt_name"]       = st["base_model"]
    wf["2"]["inputs"]["lora_name"]       = f"{cfg['lora']['output_name']}.safetensors"
    wf["2"]["inputs"]["strength_model"]  = st["lora_strength"]
    wf["2"]["inputs"]["strength_clip"]   = st["lora_strength"]
    wf["3"]["inputs"]["text"]            = st["prompt"]
    wf["4"]["inputs"]["text"]            = st["negative_prompt"]
    wf["6"]["inputs"]["image"]           = frame_name
    wf["7"]["inputs"]["width"]           = proc_w
    wf["7"]["inputs"]["height"]          = proc_h
    wf["8"]["inputs"]["resolution"]      = max(proc_w, proc_h)
    wf["11"]["inputs"]["image"]          = style_name
    wf["12"]["inputs"]["weight"]         = st["ipadapter_weight"]
    wf["13"]["inputs"]["strength"]       = st["controlnet_strength"]
    wf["15"]["inputs"]["denoise"]        = st["denoising_strength"]
    wf["15"]["inputs"]["steps"]          = st["steps"]
    wf["15"]["inputs"]["cfg"]            = st["cfg"]
    wf["15"]["inputs"]["sampler_name"]   = st["sampler"]
    wf["15"]["inputs"]["scheduler"]      = st["scheduler"]
    wf["15"]["inputs"]["seed"]           = 42
    wf["17"]["inputs"]["width"]          = orig_w
    wf["17"]["inputs"]["height"]         = orig_h
    wf["18"]["inputs"]["filename_prefix"] = "styled_out"
    return wf


def run_image_pipeline(cfg: dict, comfyui_url: str,
                       comfyui_input: Path, comfyui_output: Path):
    try:
        import numpy as np
        from PIL import Image
        from tqdm import tqdm
    except ImportError as e:
        sys.exit(f"Image pipeline requires numpy, Pillow and tqdm: {e}")

    input_video  = cfg["input"]["video"]
    factory_dir  = cfg["input"]["sample_images_dir"]
    output_video = cfg["output"]["video"]
    frames_dir   = cfg["output"].get("frames_dir", "/data/output/frames/")

    style_ref = select_style_ref(Path(factory_dir), cfg)

    orig_w, orig_h, fps = get_video_info(input_video)
    fps = override_fps(fps, cfg)
    print(f"Input: {orig_w}x{orig_h} @ {fps:.2f} fps")

    max_res = cfg["style_transfer"]["processing_resolution"]
    proc_w, proc_h = compute_proc_size(orig_w, orig_h, int(max_res)) \
        if str(max_res).lower() != "original" else ((orig_w // 8) * 8, (orig_h // 8) * 8)
    print(f"Processing at: {proc_w}x{proc_h}")

    frames_in  = "/tmp/frames_input"
    frames_out = "/tmp/frames_output"
    Path(frames_out).mkdir(parents=True, exist_ok=True)

    print("Extracting frames...")
    frames = extract_frames(input_video, frames_in, fps)
    print(f"Extracted {len(frames)} frames")

    base_wf = load_workflow(comfyui_url, "/workflows/style_transfer_frame.json")

    client_id  = str(uuid.uuid4())
    style_name = f"style_ref{style_ref.suffix}"
    shutil.copy2(style_ref, comfyui_input / style_name)

    blend_alpha = float(cfg["style_transfer"].get("temporal_blend_alpha", 0.0))
    prev_styled: "np.ndarray | None" = None
    failed = []

    print(f"\nProcessing {len(frames)} frames...")
    for idx, frame_path in enumerate(tqdm(frames, unit="frame")):
        frame_name = f"in_{idx:06d}.png"
        shutil.copy2(frame_path, comfyui_input / frame_name)
        workflow = patch_workflow_image(base_wf, frame_name, style_name,
                                        proc_w, proc_h, orig_w, orig_h, cfg)
        try:
            prompt_id = submit_workflow(comfyui_url, workflow, client_id)
            history   = wait_for_job(comfyui_url, prompt_id, timeout=600)
            out_path  = Path(frames_out) / f"styled_{idx:06d}.png"
            download_output(comfyui_url, history, out_path)

            curr = np.array(Image.open(out_path)).astype(np.float32)
            if blend_alpha > 0 and prev_styled is not None:
                blended = (1.0 - blend_alpha) * curr + blend_alpha * prev_styled
                Image.fromarray(blended.clip(0, 255).astype(np.uint8)).save(out_path)
                prev_styled = blended
            else:
                prev_styled = curr
        except Exception as e:
            print(f"\n  Frame {idx} failed: {e} — using original as fallback")
            shutil.copy2(frame_path, Path(frames_out) / f"styled_{idx:06d}.png")
            failed.append(idx)

    if failed:
        print(f"\nWarning: {len(failed)} frames fell back to original.")

    print("\nAssembling output video...")
    assemble_video(frames_out, output_video, fps, orig_w, orig_h,
                   cfg["video"]["codec"], cfg["video"]["quality"])
    print(f"Done. Output: {output_video}")


# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO PIPELINE  (AnimateDiff — entire video as one ComfyUI job)
# ═══════════════════════════════════════════════════════════════════════════════


def patch_workflow_video(wf: dict, video_name: str, style_name: str,
                         fps: float, cfg: dict) -> dict:
    wf = copy.deepcopy(wf)
    st = cfg["style_transfer"]
    ad = cfg["animatediff"]

    wf["1"]["inputs"]["ckpt_name"]          = st["base_model"]
    wf["18"]["inputs"]["lora_name"]         = f"{cfg['lora']['output_name']}.safetensors"
    wf["18"]["inputs"]["strength_model"]    = st["lora_strength"]
    wf["18"]["inputs"]["strength_clip"]     = st["lora_strength"]
    wf["2"]["inputs"]["text"]               = st["prompt"]
    wf["3"]["inputs"]["text"]               = st["negative_prompt"]
    wf["4"]["inputs"]["context_length"]     = ad["context_length"]
    wf["4"]["inputs"]["context_overlap"]    = ad["context_overlap"]
    wf["5"]["inputs"]["model_name"]         = ad["motion_module"]
    wf["8"]["inputs"]["image"]              = style_name
    wf["9"]["inputs"]["weight"]             = st["ipadapter_weight"]
    wf["10"]["inputs"]["video"]             = video_name
    wf["10"]["inputs"]["custom_width"]      = st["processing_resolution"]
    wf["11"]["inputs"]["control_net_name"]  = st.get("controlnet_model", "control_v11p_sd15_canny.safetensors")
    wf["12"]["inputs"]["resolution"]        = st["processing_resolution"]
    wf["13"]["inputs"]["strength"]          = st["controlnet_strength"]
    wf["15"]["inputs"]["denoise"]           = st["denoising_strength"]
    wf["15"]["inputs"]["steps"]             = st["steps"]
    wf["15"]["inputs"]["cfg"]               = st["cfg"]
    wf["15"]["inputs"]["sampler_name"]      = st["sampler"]
    wf["15"]["inputs"]["scheduler"]         = st["scheduler"]
    wf["17"]["inputs"]["frame_rate"]        = int(round(fps))
    return wf


def run_video_pipeline(cfg: dict, comfyui_url: str,
                       comfyui_input: Path, comfyui_output: Path):
    input_video  = Path(cfg["input"]["video"])
    factory_dir  = Path(cfg["input"]["sample_images_dir"])
    output_video = Path(cfg["output"]["video"])

    style_ref = select_style_ref(factory_dir, cfg)

    _, _, fps = get_video_info(str(input_video))
    fps = override_fps(fps, cfg)
    print(f"Input video: {input_video.name} @ {fps:.2f} fps")

    video_dest = f"input_{input_video.name}"
    style_dest = f"style_ref{style_ref.suffix}"
    shutil.copy2(input_video, comfyui_input / video_dest)
    shutil.copy2(style_ref,   comfyui_input / style_dest)
    print(f"Uploaded: {video_dest}, {style_dest}")

    base_wf  = load_workflow(comfyui_url, "/workflows/style_transfer_video.json")
    workflow = patch_workflow_video(base_wf, video_dest, style_dest, fps, cfg)
    client_id = str(uuid.uuid4())

    print("Submitting video pipeline job...")
    prompt_id = submit_workflow(comfyui_url, workflow, client_id)
    history   = wait_for_job(comfyui_url, prompt_id)

    output_video.parent.mkdir(parents=True, exist_ok=True)
    download_output(comfyui_url, history, output_video, timeout=300)
    print(f"Done. Output: {output_video}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    cfg  = load_config()
    mode = os.getenv("PIPELINE_MODE", "video").lower()
    url  = os.getenv("COMFYUI_URL", "http://localhost:8188")
    inp  = Path(os.getenv("COMFYUI_INPUT_DIR",  "/app/ComfyUI/input"))
    out  = Path(os.getenv("COMFYUI_OUTPUT_DIR", "/app/ComfyUI/output"))

    print(f"Pipeline mode: {mode}")
    wait_for_comfyui(url)

    if mode == "video":
        run_video_pipeline(cfg, url, inp, out)
    elif mode == "frame":
        run_image_pipeline(cfg, url, inp, out)
    else:
        sys.exit(f"Unknown PIPELINE_MODE '{mode}'. Use 'video' or 'frame'.")


if __name__ == "__main__":
    main()
