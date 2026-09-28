#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import torch
from PIL.PngImagePlugin import PngInfo

from diffusers import DPMSolverMultistepScheduler, Flux2KleinPipeline, Flux2Transformer2DModel


DEFAULT_BASE_MODEL = "black-forest-labs/FLUX.2-klein-base-4B"
CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)$")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def nonzero_int(value: str) -> int:
    parsed = int(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("must not be zero")
    return parsed


def image_size(value: str) -> int:
    parsed = positive_int(value)
    if parsed % 16:
        raise argparse.ArgumentTypeError("must be divisible by 16")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FLUX.2 Klein inference with a capacity-distilled student transformer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--student",
        required=True,
        help="Student transformer directory, checkpoint directory, or distillation output directory.",
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--revision")
    prompt_group = parser.add_argument_group("prompt input")
    prompt_group.add_argument("--prompt", action="append", default=[])
    prompt_group.add_argument("--prompt-file", action="append", default=[])
    prompt_group.add_argument("--negative-prompt", default="")
    generation = parser.add_argument_group("generation")
    generation.add_argument(
        "--num-images",
        type=nonzero_int,
        default=1,
        help=(
            "Positive values create this many images with incrementing seeds. A negative value creates one image "
            "per input prompt and reuses the exact --seed for every image."
        ),
    )
    generation.add_argument("--steps", type=positive_int, default=50)
    generation.add_argument("--guidance-scale", type=float, default=4.0)
    generation.add_argument("--height", type=image_size)
    generation.add_argument("--width", type=image_size)
    generation.add_argument("--seed", type=int, default=0)
    generation.add_argument(
        "--scheduler",
        choices=("model", "dpm_solver"),
        default="model",
    )
    generation.add_argument("--output-dir", type=Path, default=Path("outputs/distilled_student"))
    generation.add_argument("--filename-prefix", default="student")
    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", default="cuda")
    runtime.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    runtime.add_argument("--cpu-offload", action="store_true")
    runtime.add_argument("--compile", action="store_true")
    model_source = parser.add_mutually_exclusive_group()
    model_source.add_argument("--local-files-only", dest="local_files_only", action="store_true")
    model_source.add_argument("--allow-download", dest="local_files_only", action="store_false")
    parser.set_defaults(local_files_only=True)
    args = parser.parse_args()
    if (args.height is None) != (args.width is None):
        parser.error("--height and --width must be specified together")
    if not args.device.startswith("cuda") and args.cpu_offload:
        parser.error("--cpu-offload requires a CUDA device")
    if args.cpu_offload and args.compile:
        parser.error("--cpu-offload and --compile cannot be used together")
    return args


def read_prompts(args: argparse.Namespace) -> list[str]:
    prompts = [prompt.strip() for prompt in args.prompt if prompt.strip()]
    for file_name in args.prompt_file:
        path = Path(file_name).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        prompts.extend(
            line.strip()
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if not prompts:
        raise ValueError("At least one --prompt or --prompt-file entry is required")
    return prompts


def has_transformer_weights(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    names = (
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.safetensors.index.json",
        "diffusion_pytorch_model.bin",
        "diffusion_pytorch_model.bin.index.json",
    )
    return any((path / name).is_file() for name in names)


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if path.is_dir() and match:
            checkpoints.append((int(match.group(1)), path))
    return max(checkpoints, default=(None, None))[1]


def resolve_student_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Student path does not exist or is not a directory: {path}")

    candidates = [path / "student", path, path / "transformer"]
    checkpoint = latest_checkpoint(path)
    if checkpoint is not None:
        candidates.append(checkpoint / "transformer")
    for candidate in candidates:
        if has_transformer_weights(candidate):
            return candidate

    raise FileNotFoundError(
        f"No Diffusers transformer weights found under {path}. Expected student/config.json "
        "or checkpoint-N/transformer/config.json with diffusion_pytorch_model weights."
    )


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def configure_scheduler(pipe: Flux2KleinPipeline, scheduler_name: str) -> None:
    if scheduler_name == "model":
        return
    source_config = pipe.scheduler.config
    pipe.scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=source_config.num_train_timesteps,
        algorithm_type="dpmsolver++",
        solver_order=2,
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        flow_shift=getattr(source_config, "shift", 1.0),
        use_dynamic_shifting=True,
        time_shift_type=getattr(source_config, "time_shift_type", "exponential"),
    )


def load_pipeline(
    args: argparse.Namespace,
    student_path: Path,
    dtype: torch.dtype,
) -> Flux2KleinPipeline:
    student = Flux2Transformer2DModel.from_pretrained(
        student_path,
        dtype=dtype,
        local_files_only=True,
    )
    if student.config.guidance_embeds:
        raise ValueError(
            "The capacity-distilled base student must have guidance_embeds=False; "
            "this transformer appears to use embedded guidance."
        )

    pipe = Flux2KleinPipeline.from_pretrained(
        args.base_model,
        transformer=student,
        dtype=dtype,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    if pipe.config.get("is_distilled", False):
        raise ValueError(
            "Use FLUX.2-klein-base-4B as --base-model. The step-distilled FLUX.2-klein-4B "
            "does not match this 50-step CFG student."
        )
    configure_scheduler(pipe, args.scheduler)
    if args.cpu_offload:
        gpu_id = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
        pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    else:
        pipe.to(args.device)
    if args.compile:
        pipe.transformer = torch.compile(pipe.transformer)
    pipe.set_progress_bar_config(dynamic_ncols=True)
    return pipe


@torch.inference_mode()
def encode_prompt(
    pipe: Flux2KleinPipeline,
    prompt: str,
    device: torch.device,
) -> torch.Tensor:
    prompt_embeds, _ = pipe.encode_prompt(prompt=prompt, device=device)
    return prompt_embeds


def save_png(image, path: Path, metadata: dict) -> None:
    png_info = PngInfo()
    for key, value in metadata.items():
        png_info.add_text(key, str(value))
    image.save(path, pnginfo=png_info)


def main() -> None:
    args = parse_args()
    prompts = read_prompts(args)
    fixed_seed_mode = args.num_images < 0
    num_images = len(prompts) if fixed_seed_mode else args.num_images
    student_path = resolve_student_path(args.student)
    dtype = resolve_dtype(args.dtype)

    print(f"Student transformer: {student_path}", flush=True)
    print(f"Base pipeline: {args.base_model}", flush=True)
    print(f"Local files only: {args.local_files_only}", flush=True)
    pipe = load_pipeline(args, student_path, dtype)
    execution_device = pipe._execution_device
    negative_prompt_embeds = None
    if args.guidance_scale > 1:
        negative_prompt_embeds = encode_prompt(
            pipe,
            args.negative_prompt.strip(),
            execution_device,
        )
    print(f"Pipeline: {type(pipe).__name__}", flush=True)
    print(f"Scheduler: {type(pipe.scheduler).__name__}", flush=True)
    print(
        f"Transformer layers: {len(pipe.transformer.transformer_blocks)} double, "
        f"{len(pipe.transformer.single_transformer_blocks)} single",
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prompt_embedding_cache = {}
    records = []
    for index in range(num_images):
        prompt = prompts[index % len(prompts)]
        if prompt not in prompt_embedding_cache:
            prompt_embedding_cache[prompt] = encode_prompt(pipe, prompt, execution_device)
        seed = args.seed if fixed_seed_mode else args.seed + index
        generator = torch.Generator(device=execution_device).manual_seed(seed)
        call_kwargs = {
            "prompt_embeds": prompt_embedding_cache[prompt],
            "negative_prompt_embeds": negative_prompt_embeds,
            "guidance_scale": args.guidance_scale,
            "num_inference_steps": args.steps,
            "generator": generator,
            "output_type": "pil",
        }
        if args.height is not None:
            call_kwargs["height"] = args.height
            call_kwargs["width"] = args.width

        print(
            f"Generating {index + 1}/{num_images}: seed={seed}, prompt={prompt!r}",
            flush=True,
        )
        image = pipe(**call_kwargs).images[0]
        filename = f"{args.filename_prefix}_{run_id}_{index:04d}_seed{seed}.png"
        output_path = args.output_dir / filename 
        metadata = {
            "prompt": prompt,
            "negative_prompt": args.negative_prompt,
            "seed": seed,
            "student": str(student_path),
            "base_model": args.base_model,
            "pipeline": type(pipe).__name__,
            "scheduler": type(pipe.scheduler).__name__,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "dtype": args.dtype,
            "width": image.width,
            "height": image.height,
        }
        save_png(image, output_path, metadata)
        records.append({"file": str(output_path), **metadata})
        print(f"Saved: {output_path}", flush=True)

    manifest_path = args.output_dir / f"manifest_{run_id}.json"
    manifest = {
        "student": str(student_path),
        "base_model": args.base_model,
        "num_images": num_images,
        "requested_num_images": args.num_images,
        "fixed_seed_mode": fixed_seed_mode,
        "arguments": vars(args) | {"output_dir": str(args.output_dir)},
        "images": records,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error


# CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 \
# python my/predict_distillation.py \
#   --student /data1/jooyonglee/training/diffuers/distillation/flux2/checkpoint-270000/ \
#   --num-images 300 \
#   --steps 50 \
#   --guidance-scale 4.0 \
#   --height 512 \
#   --width 512 \
#   --seed 42 \
#   --scheduler model \
#   --dtype bf16 \
#   --device cuda:0 \
#   --output-dir ./outputs/distilled-studentpt
  
  
