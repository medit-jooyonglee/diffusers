from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageChops, ImageEnhance

from diffusers import Flux2KleinPipeline

from .embedding_library import EmbeddingLibrary
from .inference import ROLE_ARGUMENTS, load_transformer_override, resolve_layout_and_images


DEFAULT_MODEL = "black-forest-labs/FLUX.2-klein-4B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--text-encoder-model")
    parser.add_argument("--text-encoder-revision")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--entry-id", required=True)
    parser.add_argument("--patient", type=Path, required=True)
    parser.add_argument("--tooth-reference", type=Path)
    parser.add_argument("--mouth-reference", type=Path)
    parser.add_argument("--pose-reference", type=Path)
    parser.add_argument("--extra-reference", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/smile_design_parity"))
    return parser.parse_args()


def run_pipeline(
    pipeline: Flux2KleinPipeline,
    call_kwargs: dict,
) -> tuple[Image.Image, torch.Tensor]:
    captured = {}

    def capture_latents(_, __, ___, callback_kwargs):
        captured["latents"] = callback_kwargs["latents"].detach().float().cpu()
        return callback_kwargs

    with torch.inference_mode():
        image = pipeline(
            **call_kwargs,
            callback_on_step_end=capture_latents,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images[0]
    if "latents" not in captured:
        raise RuntimeError("Pipeline did not expose final latents")
    return image, captured["latents"]


def main() -> None:
    args = parse_args()
    library = EmbeddingLibrary(args.library)
    record = library.entry_metadata(args.entry_id)
    layout_args = SimpleNamespace(layout=record["layout"])
    for role, argument in ROLE_ARGUMENTS.items():
        setattr(layout_args, argument, getattr(args, argument))
    layout, reference_images, reference_records = resolve_layout_and_images(layout_args, library)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    text_encoder_source = args.text_encoder_model or args.model
    text_encoder_revision = args.text_encoder_revision or args.revision
    encoding_pipeline = Flux2KleinPipeline.from_pretrained(
        text_encoder_source,
        revision=text_encoder_revision,
        transformer=None,
        vae=None,
        scheduler=None,
        dtype=dtype,
        local_files_only=args.local_files_only,
    )
    transformer = load_transformer_override(args.model, args.revision, dtype, args.local_files_only)
    pipeline = Flux2KleinPipeline.from_pretrained(
        args.model,
        revision=args.revision,
        transformer=transformer,
        text_encoder=encoding_pipeline.text_encoder,
        tokenizer=encoding_pipeline.tokenizer,
        dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(args.device)
    del encoding_pipeline
    pipeline.set_progress_bar_config(dynamic_ncols=True)
    steps = args.steps if args.steps is not None else 4 if pipeline.config.get("is_distilled", False) else 50
    guidance_scale = (
        args.guidance_scale
        if args.guidance_scale is not None
        else 1.0 if pipeline.config.get("is_distilled", False) else 4.0
    )
    shared = {
        "image": reference_images,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "num_images_per_prompt": 1,
        "max_sequence_length": library.manifest["max_sequence_length"],
        "text_encoder_out_layers": tuple(library.manifest["text_encoder_out_layers"]),
    }
    if args.height is not None:
        shared["height"] = args.height
    if args.width is not None:
        shared["width"] = args.width
    runtime_image, runtime_latents = run_pipeline(
        pipeline,
        shared
        | {
            "prompt": record["prompt"],
            "generator": torch.Generator(device=args.device).manual_seed(args.seed),
        },
    )
    cached_image, cached_latents = run_pipeline(
        pipeline,
        shared
        | {
            "prompt_embeds": library.load(args.entry_id, args.device, dtype),
            "negative_prompt_embeds": library.load_negative(args.device, dtype),
            "generator": torch.Generator(device=args.device).manual_seed(args.seed),
        },
    )
    runtime_pixels = np.asarray(runtime_image).astype(np.float32)
    cached_pixels = np.asarray(cached_image).astype(np.float32)
    pixel_difference = np.abs(runtime_pixels - cached_pixels)
    latent_difference = (runtime_latents - cached_latents).abs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    runtime_path = args.output_dir / f"runtime_{run_id}.png"
    cached_path = args.output_dir / f"cached_{run_id}.png"
    difference_path = args.output_dir / f"difference_x8_{run_id}.png"
    runtime_image.save(runtime_path)
    cached_image.save(cached_path)
    difference = ImageChops.difference(runtime_image.convert("RGB"), cached_image.convert("RGB"))
    ImageEnhance.Brightness(difference).enhance(8.0).save(difference_path)
    report = {
        "entry_id": args.entry_id,
        "layout": layout,
        "prompt": record["prompt"],
        "references": reference_records,
        "seed": args.seed,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "latent_max_abs": latent_difference.max().item(),
        "latent_mean_abs": latent_difference.mean().item(),
        "pixel_max_abs": float(pixel_difference.max()),
        "pixel_mean_abs": float(pixel_difference.mean()),
        "runtime_image": str(runtime_path),
        "cached_image": str(cached_path),
        "difference_image": str(difference_path),
    }
    report_path = args.output_dir / f"parity_{run_id}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
