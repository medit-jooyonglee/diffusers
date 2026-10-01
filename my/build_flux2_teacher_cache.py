import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import torch

from diffusers import Flux2KleinPipeline


DEFAULT_TEACHER = "black-forest-labs/FLUX.2-klein-base-4B"
SAFE_ID_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def multiple_of_16(value):
    value = positive_int(value)
    if value % 16:
        raise argparse.ArgumentTypeError("must be divisible by 16")
    return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--steps", type=positive_int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--height", type=multiple_of_16, default=512)
    parser.add_argument("--width", type=multiple_of_16, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=positive_int)
    parser.add_argument("--preview-every", type=nonnegative_int, default=100)
    parser.add_argument("--log-every", type=positive_int, default=10)
    parser.add_argument("--num-shards", type=positive_int, default=1)
    parser.add_argument("--shard-index", type=nonnegative_int, default=0)
    parser.add_argument("--save-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.shard_index >= args.num_shards:
        parser.error("--shard-index must be smaller than --num-shards")
    if args.guidance_scale <= 1:
        parser.error("--guidance-scale must be greater than 1 for true CFG teacher cache generation")
    return args


def load_prompt_records(path):
    if not path.is_file():
        raise FileNotFoundError(f"Prompt JSONL not found: {path}")
    records = []
    prompt_ids = set()
    prompts = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
        if isinstance(payload, str):
            prompt = payload.strip()
            payload = {"prompt": prompt}
        elif isinstance(payload, dict):
            prompt = str(payload.get("prompt", "")).strip()
        else:
            raise ValueError(f"Expected a string or object at {path}:{line_number}")
        if not prompt:
            raise ValueError(f"Empty prompt at {path}:{line_number}")
        raw_id = str(payload.get("prompt_id") or hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16])
        prompt_id = SAFE_ID_PATTERN.sub("-", raw_id).strip("-._")
        if not prompt_id:
            raise ValueError(f"Invalid prompt_id at {path}:{line_number}: {raw_id!r}")
        if prompt_id in prompt_ids:
            raise ValueError(f"Duplicate prompt_id at {path}:{line_number}: {prompt_id}")
        if prompt in prompts:
            raise ValueError(f"Duplicate prompt at {path}:{line_number}: {prompt[:120]}")
        prompt_ids.add(prompt_id)
        prompts.add(prompt)
        records.append({"prompt_id": prompt_id, "prompt": prompt, "source": payload})
    if not records:
        raise ValueError(f"No prompts found in {path}")
    return records


def atomic_torch_save(value, path):
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_text_write(text, path):
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def cache_fingerprint(record, args):
    payload = {
        "prompt": record["prompt"],
        "teacher_model": args.teacher_model,
        "revision": args.revision,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "dtype": args.dtype,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def sample_status(sample_dir, record, args):
    required = [sample_dir / "latent.pt", sample_dir / "prompt.txt", sample_dir / "metadata.json"]
    if args.save_embeddings:
        required.append(sample_dir / "embedding.pt")
    if not (sample_dir / "complete.json").is_file() or not all(path.is_file() for path in required):
        return "incomplete"
    try:
        metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "incomplete"
    return "complete" if metadata.get("cache_fingerprint") == cache_fingerprint(record, args) else "mismatch"


def unpack_normalized_latents(packed_latents, pipeline, height, width):
    latent_height = 2 * (height // (pipeline.vae_scale_factor * 2))
    latent_width = 2 * (width // (pipeline.vae_scale_factor * 2))
    spatial_height = latent_height // 2
    spatial_width = latent_width // 2
    batch_size, sequence_length, channels = packed_latents.shape
    if sequence_length != spatial_height * spatial_width:
        raise ValueError(
            f"Packed latent sequence has length {sequence_length}, expected "
            f"{spatial_height * spatial_width} for {height}x{width}"
        )
    return packed_latents.permute(0, 2, 1).reshape(batch_size, channels, spatial_height, spatial_width)


def should_save_preview(global_index, args):
    if args.preview_every == 0:
        return False
    block_index, block_offset = divmod(global_index, args.preview_every)
    return block_offset == block_index % args.num_shards


def load_pipeline(args, dtype):
    pipeline = Flux2KleinPipeline.from_pretrained(
        args.teacher_model,
        revision=args.revision,
        dtype=dtype,
        local_files_only=args.local_files_only,
    )
    if pipeline.config.get("is_distilled", False):
        raise ValueError("Teacher cache requires FLUX.2-klein-base-4B, not the step-distilled model")
    if pipeline.transformer.config.guidance_embeds:
        raise ValueError("Expected the FLUX.2 Klein base teacher with guidance_embeds=False")
    pipeline.to(args.device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_batch(pipeline, batch, args, dtype):
    prompts = [item["prompt"] for item in batch]
    seeds = [args.seed + item["global_index"] for item in batch]
    generators = [torch.Generator(device=args.device).manual_seed(seed) for seed in seeds]
    prompt_embeds, _ = pipeline.encode_prompt(prompt=prompts, device=args.device)
    negative_prompt_embeds, _ = pipeline.encode_prompt(prompt=[""] * len(batch), device=args.device)
    captured = {}

    def capture_final_latents(_, __, ___, callback_kwargs):
        captured["packed_latents"] = callback_kwargs["latents"].detach()
        return callback_kwargs

    save_preview = any(should_save_preview(item["global_index"], args) for item in batch)
    with torch.inference_mode():
        output = pipeline(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=1,
            generator=generators,
            output_type="pil" if save_preview else "latent",
            callback_on_step_end=capture_final_latents,
            callback_on_step_end_tensor_inputs=["latents"],
        )
    if "packed_latents" not in captured:
        raise RuntimeError("The pipeline did not expose final latents through callback_on_step_end")
    latents = unpack_normalized_latents(
        captured["packed_latents"], pipeline, args.height, args.width
    ).to(device="cpu", dtype=dtype)
    embeddings = prompt_embeds.detach().to(device="cpu", dtype=dtype)
    previews = output.images if save_preview else [None] * len(batch)
    return latents, embeddings, previews, seeds


def save_batch(batch, latents, embeddings, previews, seeds, pipeline, args):
    manifest_records = []
    for offset, item in enumerate(batch):
        sample_dir = item["sample_dir"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": 1,
            "prompt_id": item["prompt_id"],
            "prompt": item["prompt"],
            "seed": seeds[offset],
            "teacher_model": args.teacher_model,
            "revision": args.revision,
            "scheduler": type(pipeline.scheduler).__name__,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "height": args.height,
            "width": args.width,
            "dtype": args.dtype,
            "latent_format": "normalized_patchified_chw",
            "cache_fingerprint": cache_fingerprint(item, args),
            "source": item["source"],
        }
        atomic_torch_save(latents[offset].contiguous(), sample_dir / "latent.pt")
        if args.save_embeddings:
            atomic_torch_save(embeddings[offset].contiguous(), sample_dir / "embedding.pt")
        atomic_text_write(item["prompt"] + "\n", sample_dir / "prompt.txt")
        atomic_text_write(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", sample_dir / "metadata.json")
        if previews[offset] is not None and should_save_preview(item["global_index"], args):
            preview_path = sample_dir / "preview.png"
            previews[offset].save(preview_path)
        atomic_text_write(json.dumps({"complete": True}) + "\n", sample_dir / "complete.json")
        manifest_records.append(metadata)
    return manifest_records


def main():
    args = parse_args()
    records = load_prompt_records(args.prompt_jsonl)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    selected = []
    skipped = 0
    for global_index, record in enumerate(records):
        if global_index % args.num_shards != args.shard_index:
            continue
        sample_dir = args.output_dir / record["prompt_id"]
        if not args.overwrite:
            status = sample_status(sample_dir, record, args)
            if status == "complete":
                skipped += 1
                continue
            if status == "mismatch":
                raise ValueError(
                    f"Cache configuration mismatch for {sample_dir}. Use a new --output-dir or --overwrite."
                )
        selected.append({**record, "global_index": global_index, "sample_dir": sample_dir})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(
        f"Prompts={len(records):,}, shard={args.shard_index}/{args.num_shards}, "
        f"pending={len(selected):,}, skipped={skipped:,}"
    )
    if not selected:
        return
    pipeline = load_pipeline(args, dtype)
    written = []
    started_at = time.monotonic()
    shard_total = skipped + len(selected)
    for start in range(0, len(selected), args.batch_size):
        batch = selected[start : start + args.batch_size]
        latents, embeddings, previews, seeds = build_batch(pipeline, batch, args, dtype)
        written.extend(save_batch(batch, latents, embeddings, previews, seeds, pipeline, args))
        del latents, embeddings, previews
        processed = start + len(batch)
        if processed == len(selected) or processed == len(batch) or processed % args.log_every < len(batch):
            elapsed = time.monotonic() - started_at
            rate = processed / elapsed
            eta = (len(selected) - processed) / rate
            completed = skipped + processed
            print(
                f"[progress] shard={args.shard_index}/{args.num_shards} "
                f"completed={completed:,}/{shard_total:,} ({completed / shard_total:.1%}) "
                f"this_run={processed:,}/{len(selected):,} rate={rate:.3f} samples/s "
                f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}",
                flush=True,
            )
    manifest_records = []
    for global_index, record in enumerate(records):
        if global_index % args.num_shards != args.shard_index:
            continue
        sample_dir = args.output_dir / record["prompt_id"]
        if sample_status(sample_dir, record, args) == "complete":
            manifest_records.append(json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8")))
    shard_manifest = args.output_dir / f"manifest-shard-{args.shard_index:05d}.jsonl"
    atomic_text_write(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in manifest_records),
        shard_manifest,
    )
    print(
        f"Wrote {len(written):,} samples; shard cache now has {len(manifest_records):,} complete samples; "
        f"manifest={shard_manifest}"
    )


if __name__ == "__main__":
    main()
