"""Drives the image-edit -> img2vid pipeline against the two servers unified_server.py starts
(design_tune_server.py for FLUX.2, img2vid backend_server for Wan2.2), one after the other.

This talks to both servers over HTTP exactly like design_tune_client.py and backent_client.py do; it
does not merge them. Sequencing (and, if --serialize-gpu was passed to unified_server.py, cross-process
GPU exclusion) happens on the server side.

Example:
    python my/unified_client.py smile_v1 my/samples/flux/faces/face_000_yellow_upper.png \\
        --layout patient_only --preset natural --wan-backend wan-turbo --frame-num 17 \\
        --output-dir outputs/unified_run1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests


DEFAULT_WAN_PROMPT = (
    "A photorealistic close-up portrait video of the same person in the input image. "
    "The person's crooked, uneven, and discolored teeth gradually transform into beautifully restored, "
    "naturally aligned, clean white teeth with realistic dental anatomy and proportions. "
    "The person transitions smoothly through several natural facial expressions: a subtle closed-mouth smile, "
    "a gentle smile revealing the upper teeth, a warm confident smile, and finally a broad joyful smile "
    "showing the beautifully restored teeth. Static frontal camera, no head rotation, no camera movement.")
DEFAULT_WAN_NEG_PROMPT = (
    "deformed teeth, crooked restored teeth, missing teeth, extra teeth, duplicated teeth, unnatural tooth shapes, "
    "oversized teeth, excessively white teeth, flickering teeth, changing tooth alignment, "
    "inconsistent dental anatomy, distorted mouth, asymmetrical lips, facial identity changes, face morphing, "
    "blurry teeth, artificial smile, uncanny valley, camera movement, head rotation, low quality, "
    "temporal flickering")


def fail(response: requests.Response) -> None:
    try:
        detail = response.json()
    except ValueError:
        detail = response.text
    sys.exit(f"HTTP {response.status_code}: {detail}")


def design_tune_infer(args: argparse.Namespace, output_path: Path) -> dict:
    parameters = {
        "style_name": args.style, "layout": args.layout, "seed": args.seed,
        "mode": "preset", "preset": args.preset,
    }
    if args.steps is not None:
        parameters["num_steps"] = args.steps
    files = [("images", (args.image.name, args.image.read_bytes(), "application/octet-stream"))]
    started = time.time()
    response = requests.post(
        f"{args.flux_server}/api/v1/infer",
        data={"parameters": json.dumps(parameters, ensure_ascii=False)},
        files=files,
        timeout=600,
    )
    if not response.ok:
        fail(response)
    output_path.write_bytes(response.content)
    metadata = json.loads(response.headers["X-Design-Tune-Metadata"])
    metadata["_elapsed_sec"] = round(time.time() - started, 1)
    return metadata


def wan_generate(args: argparse.Namespace, image_path: Path, output_path: Path) -> dict:
    turbo = args.wan_backend == "wan-turbo"
    sampling_steps = args.sampling_steps if args.sampling_steps is not None else (4 if turbo else 50)
    guide_scale = args.guide_scale if args.guide_scale is not None else (1.0 if turbo else 5.0)
    data = {
        "prompt": args.wan_prompt or DEFAULT_WAN_PROMPT,
        "negative_prompt": args.wan_negative_prompt or DEFAULT_WAN_NEG_PROMPT,
        "size": args.size, "frame_num": str(args.frame_num), "sampling_steps": str(sampling_steps),
        "guide_scale": str(guide_scale), "shift": str(args.shift), "seed": str(args.wan_seed),
        "fps": str(args.fps),
    }
    with open(image_path, "rb") as handle:
        files = {"image": (image_path.name, handle, "application/octet-stream")}
        started = time.time()
        response = requests.post(f"{args.wan_server}/generate", data=data, files=files, timeout=args.wan_timeout)
    if response.status_code != 200:
        sys.exit(f"HTTP {response.status_code}: {response.text}")
    # Server returns the video itself as the body; metadata rides in a header (same pattern as
    # design_tune_infer's X-Design-Tune-Metadata above).
    output_path.write_bytes(response.content)
    result = json.loads(response.headers.get("X-Img2Vid-Metadata", "{}"))
    result["_elapsed_sec"] = round(time.time() - started, 1)
    result["local_video_path"] = str(output_path)
    return result


def start_vram_monitor(log_path: Path) -> subprocess.Popen:
    command = (
        "while true; do echo \"$(date +%H:%M:%S) $(nvidia-smi --query-gpu=memory.used "
        f"--format=csv,noheader)\"; sleep 1; done > {log_path} 2>&1"
    )
    return subprocess.Popen(command, shell=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("style")
    parser.add_argument("image", type=Path)
    parser.add_argument("--flux-server", default="http://127.0.0.1:5000")
    parser.add_argument("--wan-server", default="http://127.0.0.1:8000")
    parser.add_argument("--layout", default="patient_only")
    parser.add_argument("--preset", default="natural")
    parser.add_argument("--steps", type=int, help="FLUX.2 num_inference_steps")
    parser.add_argument("--seed", type=int, default=0, help="FLUX.2 seed")
    parser.add_argument("--wan-backend", default="wan-turbo", choices=["wan", "wan-turbo"])
    parser.add_argument("--wan-prompt")
    parser.add_argument("--wan-negative-prompt")
    parser.add_argument("--size", default="704*1280", choices=["704*1280", "1280*704"])
    parser.add_argument("--frame-num", type=int, default=37, dest="frame_num", help="4n+1")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--sampling-steps", type=int, dest="sampling_steps")
    parser.add_argument("--guide-scale", type=float, dest="guide_scale")
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--wan-seed", type=int, default=42)
    parser.add_argument("--wan-timeout", type=float, default=3600)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/unified_run"))
    parser.add_argument("--vram-log", action="store_true", help="Poll nvidia-smi every second into <output-dir>/vram_log.txt")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    monitor = start_vram_monitor(args.output_dir / "vram_log.txt") if args.vram_log else None

    try:
        print(f"[1/2] design_tune_server: {args.style}/{args.layout}/{args.preset} on {args.image}")
        edited_path = args.output_dir / "flux_edit.png"
        flux_metadata = design_tune_infer(args, edited_path)
        print(f"      saved {edited_path} ({flux_metadata['_elapsed_sec']}s)")

        print(f"[2/2] img2vid backend_server ({args.wan_backend}) on {edited_path}")
        video_path = args.output_dir / "video.mp4"
        wan_result = wan_generate(args, edited_path, video_path)
        print(f"      saved {video_path} ({wan_result['_elapsed_sec']}s)")
    finally:
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=5)

    summary = {"flux": flux_metadata, "wan": wan_result}
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done. summary: {summary_path}")


if __name__ == "__main__":
    main()
