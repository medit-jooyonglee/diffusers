from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageFilter
from PIL.PngImagePlugin import PngInfo

from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel

from .embedding_library import EmbeddingLibrary


DEFAULT_MODEL = "black-forest-labs/FLUX.2-klein-4B"
ROLE_ARGUMENTS = {
    "patient": "patient",
    "tooth": "tooth_reference",
    "mouth": "mouth_reference",
    "pose": "pose_reference",
    "extra": "extra_reference",
}
CONTROL_ARGUMENTS = {
    "whitening": "whitening",
    "smile_intensity": "smile_intensity",
    "tooth_roundness": "tooth_roundness",
    "gum_exposure": "gum_exposure",
    "mouth_opening": "mouth_opening",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--layout")
    parser.add_argument("--patient", type=Path, required=True)
    parser.add_argument("--tooth-reference", type=Path)
    parser.add_argument("--mouth-reference", type=Path)
    parser.add_argument("--pose-reference", type=Path)
    parser.add_argument("--extra-reference", type=Path)
    parser.add_argument("--mode", choices=("preset", "anchor", "grid", "delta"), default="preset")
    parser.add_argument("--preset", default="natural")
    parser.add_argument("--attribute", choices=tuple(CONTROL_ARGUMENTS))
    parser.add_argument("--value", type=float)
    parser.add_argument("--grid", default="whitening_smile")
    parser.add_argument("--whitening", type=float)
    parser.add_argument("--smile-intensity", type=float)
    parser.add_argument("--tooth-roundness", type=float)
    parser.add_argument("--gum-exposure", type=float)
    parser.add_argument("--mouth-opening", type=float)
    parser.add_argument("--delta-strength", action="append", default=[])
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--mask-feather", type=float, default=12.0)
    parser.add_argument("--steps", type=positive_int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--height", type=multiple_of_16)
    parser.add_argument("--width", type=multiple_of_16)
    parser.add_argument("--num-images", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-library-model-mismatch", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/smile_design"))
    parser.add_argument("--filename-prefix", default="smile")
    return parser.parse_args()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def multiple_of_16(value: str) -> int:
    parsed = positive_int(value)
    if parsed % 16:
        raise argparse.ArgumentTypeError("must be divisible by 16")
    return parsed


def parse_strengths(values: list[str]) -> dict[str, float]:
    strengths = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--delta-strength must use ATTRIBUTE=VALUE, got {value!r}")
        name, raw_strength = value.split("=", 1)
        name = name.strip()
        if name not in CONTROL_ARGUMENTS:
            raise ValueError(f"Unknown delta strength attribute: {name}")
        strengths[name] = float(raw_strength)
    return strengths


def supplied_control_values(args: argparse.Namespace) -> dict[str, float]:
    return {
        name: float(getattr(args, argument))
        for name, argument in CONTROL_ARGUMENTS.items()
        if getattr(args, argument) is not None
    }


def resolve_layout_and_images(
    args: argparse.Namespace, library: EmbeddingLibrary
) -> tuple[str, list[Image.Image], list[dict[str, str]]]:
    supplied = {
        role: getattr(args, argument)
        for role, argument in ROLE_ARGUMENTS.items()
        if getattr(args, argument) is not None
    }
    inferred_layout = "patient_only" if list(supplied) == ["patient"] else "_".join(supplied)
    layout = args.layout or inferred_layout
    layout_config = library.manifest["layouts"].get(layout)
    if layout_config is None:
        raise ValueError(f"Reference combination {inferred_layout!r} has no cached layout; requested={layout!r}")
    expected_roles = layout_config["roles"]
    if list(supplied) != expected_roles:
        raise ValueError(f"Layout {layout!r} requires reference roles {expected_roles}, got {list(supplied)}")
    images = []
    records = []
    for role in expected_roles:
        path = Path(supplied[role]).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{role} reference image not found: {path}")
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
        records.append({"role": role, "file": str(path.resolve())})
    return layout, images, records


def resolve_embedding(
    args: argparse.Namespace,
    library: EmbeddingLibrary,
    layout: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, Any]]:
    values = supplied_control_values(args)
    if args.mode == "preset":
        if values or args.attribute is not None or args.value is not None:
            raise ValueError("Preset mode does not accept slider values or --attribute/--value")
        embedding = library.preset(layout, args.preset, device=args.device, dtype=dtype)
        return embedding, {"mode": "preset", "preset": args.preset}
    if args.mode == "anchor":
        if args.attribute is None or args.value is None:
            raise ValueError("Anchor mode requires --attribute and --value")
        if values:
            raise ValueError("Anchor mode uses --attribute/--value, not named slider arguments")
        embedding = library.interpolate_anchor(layout, args.attribute, args.value, args.device, dtype)
        return embedding, {"mode": "anchor", "attribute": args.attribute, "value": args.value}
    if args.attribute is not None or args.value is not None:
        raise ValueError("--attribute/--value are only valid in anchor mode")
    if args.mode == "grid":
        grid = library.manifest["grids"].get(args.grid)
        if grid is None:
            raise ValueError(f"Unknown grid: {args.grid}")
        required = set(grid["attributes"])
        if set(values) != required:
            raise ValueError(f"Grid {args.grid!r} requires exactly these sliders: {sorted(required)}")
        embedding = library.interpolate_grid(layout, args.grid, values, args.device, dtype)
        return embedding, {"mode": "grid", "grid": args.grid, "values": values}
    if not values:
        raise ValueError("Delta mode requires at least one named slider argument")
    strengths = parse_strengths(args.delta_strength)
    embedding = library.compose_deltas(layout, values, strengths, args.device, dtype)
    return embedding, {"mode": "delta", "values": values, "strengths": strengths}


def load_product_pipeline(args: argparse.Namespace, dtype: torch.dtype) -> Flux2KleinPipeline:
    transformer = load_transformer_override(
        args.model, args.revision, dtype, args.local_files_only
    )
    pipeline = Flux2KleinPipeline.from_pretrained(
        args.model,
        revision=args.revision,
        transformer=transformer,
        text_encoder=None,
        tokenizer=None,
        dtype=dtype,
        local_files_only=args.local_files_only,
    )
    if pipeline.text_encoder is not None or pipeline.tokenizer is not None:
        raise RuntimeError("Product pipeline unexpectedly loaded a text encoder or tokenizer")
    if args.cpu_offload:
        if not args.device.startswith("cuda"):
            raise ValueError("--cpu-offload requires a CUDA device")
        gpu_id = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
        pipeline.enable_model_cpu_offload(gpu_id=gpu_id)
    else:
        pipeline.to(args.device)
    pipeline.set_progress_bar_config(dynamic_ncols=True)
    return pipeline


def load_transformer_override(
    model: str,
    revision: str | None,
    dtype: torch.dtype,
    local_files_only: bool,
) -> Flux2Transformer2DModel | None:
    model_path = Path(model).expanduser()
    if not model_path.is_dir():
        return None
    transformer_directory = model_path / "transformer"
    diffusers_weights = (
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.safetensors.index.json",
        "diffusion_pytorch_model.bin",
        "diffusion_pytorch_model.bin.index.json",
    )
    if any((transformer_directory / name).is_file() for name in diffusers_weights):
        return None
    candidates = sorted(model_path.glob("*.safetensors"))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError(f"Expected one root transformer checkpoint in {model_path}, found {candidates}")
    return Flux2Transformer2DModel.from_single_file(
        candidates[0],
        config=str(model_path),
        subfolder="transformer",
        revision=revision,
        dtype=dtype,
        local_files_only=local_files_only,
    )


def composite_with_mask(generated: Image.Image, patient: Image.Image, mask_path: Path, feather: float) -> Image.Image:
    if feather < 0:
        raise ValueError("--mask-feather must be non-negative")
    with Image.open(mask_path) as mask_image:
        mask = mask_image.convert("L").resize(generated.size, Image.Resampling.LANCZOS)
    if feather:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    original = patient.convert("RGB").resize(generated.size, Image.Resampling.LANCZOS)
    return Image.composite(generated.convert("RGB"), original, mask)


def save_png(image: Image.Image, path: Path, metadata: dict[str, Any]) -> None:
    png_info = PngInfo()
    for key, value in metadata.items():
        rendered = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        png_info.add_text(key, rendered)
    image.save(path, pnginfo=png_info)


def main() -> None:
    args = parse_args()
    library = EmbeddingLibrary(args.library)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    layout, reference_images, reference_records = resolve_layout_and_images(args, library)
    prompt_embeds, mix = resolve_embedding(args, library, layout, dtype)
    negative_prompt_embeds = library.load_negative(device=args.device, dtype=dtype)
    pipeline_load_started = time.perf_counter()
    pipeline = load_product_pipeline(args, dtype)
    pipeline_load_seconds = time.perf_counter() - pipeline_load_started
    if prompt_embeds.shape[-1] != pipeline.transformer.config.joint_attention_dim:
        raise ValueError(
            f"Embedding dimension {prompt_embeds.shape[-1]} does not match transformer dimension "
            f"{pipeline.transformer.config.joint_attention_dim}"
        )
    library_model = library.manifest.get("model")
    if not args.allow_library_model_mismatch and library_model != args.model:
        raise ValueError(
            f"Embedding library model {library_model!r} differs from runtime model {args.model!r}; "
            "use the identical model argument or --allow-library-model-mismatch"
        )
    steps = args.steps if args.steps is not None else 4 if pipeline.config.get("is_distilled", False) else 50
    guidance_scale = (
        args.guidance_scale
        if args.guidance_scale is not None
        else 1.0 if pipeline.config.get("is_distilled", False) else 4.0
    )
    generators = [
        torch.Generator(device=args.device).manual_seed(args.seed + index)
        for index in range(args.num_images)
    ]
    call_kwargs: dict[str, Any] = {
        "image": reference_images,
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "num_images_per_prompt": args.num_images,
        "generator": generators,
    }
    if args.height is not None:
        call_kwargs["height"] = args.height
    if args.width is not None:
        call_kwargs["width"] = args.width
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    generation_started = time.perf_counter()
    with torch.inference_mode():
        images = pipeline(**call_kwargs).images
    generation_seconds = time.perf_counter() - generation_started
    peak_allocated = (
        torch.cuda.max_memory_allocated(torch.device(args.device)) if args.device.startswith("cuda") else 0
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(torch.device(args.device)) if args.device.startswith("cuda") else 0
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    records = []
    for index, image in enumerate(images):
        if args.mask is not None:
            image = composite_with_mask(image, reference_images[0], args.mask, args.mask_feather)
        path = args.output_dir / f"{args.filename_prefix}_{run_id}_{index:03d}_seed{args.seed + index}.png"
        metadata = {
            "layout": layout,
            "mix": mix,
            "seed": args.seed + index,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "references": reference_records,
            "text_encoder_loaded": False,
        }
        save_png(image, path, metadata)
        records.append({"file": str(path), **metadata})
        print(f"saved={path}", flush=True)
    report = {
        "model": args.model,
        "library": str(args.library.resolve()),
        "template_version": library.manifest["template_version"],
        "layout": layout,
        "mix": mix,
        "references": reference_records,
        "pipeline_load_seconds": pipeline_load_seconds,
        "generation_seconds": generation_seconds,
        "peak_cuda_allocated_gib": peak_allocated / 2**30,
        "peak_cuda_reserved_gib": peak_reserved / 2**30,
        "text_encoder_loaded": False,
        "images": records,
    }
    report_path = args.output_dir / f"manifest_{run_id}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
