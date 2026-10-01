from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from diffusers import Flux2KleinPipeline, __version__ as diffusers_version

from .prompt_templates import build_prompt_entries, load_experiment_config, reference_roles


DEFAULT_MODEL = "black-forest-labs/FLUX.2-klein-4B"
DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "smile_design" / "smile_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--text-encoder-model")
    parser.add_argument("--text-encoder-revision")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layout", action="append", default=[])
    parser.add_argument("--kind", action="append", choices=("base", "preset", "anchor", "grid"), default=[])
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--max-sequence-length", type=positive_int, default=512)
    parser.add_argument("--text-encoder-out-layers", type=int, nargs="+", default=[9, 18, 27])
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_json_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def model_config_fingerprint(model: str, revision: str | None) -> str:
    path = Path(model).expanduser()
    digest = hashlib.sha256()
    digest.update(str(path.resolve() if path.exists() else model).encode("utf-8"))
    digest.update(str(revision).encode("utf-8"))
    if path.is_dir():
        for relative in ("model_index.json", "text_encoder/config.json", "tokenizer/tokenizer_config.json"):
            candidate = path / relative
            if candidate.is_file():
                digest.update(relative.encode("utf-8"))
                digest.update(candidate.read_bytes())
    return digest.hexdigest()


def entry_file(prompt: str) -> Path:
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return Path("embeddings") / f"{prompt_hash}.pt"


def load_encoder_pipeline(args: argparse.Namespace, dtype: torch.dtype) -> Flux2KleinPipeline:
    source = args.text_encoder_model or args.model
    revision = args.text_encoder_revision or args.revision
    pipeline = Flux2KleinPipeline.from_pretrained(
        source,
        revision=revision,
        transformer=None,
        vae=None,
        scheduler=None,
        dtype=dtype,
        local_files_only=args.local_files_only,
    )
    if pipeline.text_encoder is None or pipeline.tokenizer is None:
        raise ValueError("The source model must include its Qwen text encoder and tokenizer")
    pipeline.text_encoder.eval().requires_grad_(False)
    pipeline.to(args.device)
    return pipeline


def main() -> None:
    args = parse_args()
    config, config_hash = load_experiment_config(args.config)
    entries = build_prompt_entries(
        config,
        layouts=args.layout or None,
        kinds=set(args.kind) if args.kind else None,
    )
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    text_encoder_model = args.text_encoder_model or args.model
    text_encoder_revision = args.text_encoder_revision or args.revision
    fingerprint = model_config_fingerprint(text_encoder_model, text_encoder_revision)
    manifest = {
        "schema_version": 1,
        "template_version": config["template_version"],
        "template_config_sha256": config_hash,
        "model": args.model,
        "revision": args.revision,
        "text_encoder_model": text_encoder_model,
        "text_encoder_revision": text_encoder_revision,
        "text_encoder_config_fingerprint": fingerprint,
        "diffusers_version": diffusers_version,
        "dtype": args.dtype,
        "max_sequence_length": args.max_sequence_length,
        "text_encoder_out_layers": args.text_encoder_out_layers,
        "embedding_shape": None,
        "negative_embedding": "negative.pt",
        "controls": config["controls"],
        "grids": config["grids"],
        "layouts": {name: {"roles": reference_roles(config, name)} for name in config["layouts"]},
        "entries": {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        compatibility_keys = (
            "schema_version",
            "template_version",
            "template_config_sha256",
            "model",
            "revision",
            "text_encoder_model",
            "text_encoder_revision",
            "text_encoder_config_fingerprint",
            "dtype",
            "max_sequence_length",
            "text_encoder_out_layers",
        )
        mismatches = [key for key in compatibility_keys if existing.get(key) != manifest.get(key)]
        if mismatches:
            raise ValueError(
                f"Existing embedding library is incompatible in {mismatches}; use a new --output-dir"
            )
        manifest = existing

    pending = []
    for entry in entries:
        relative_path = entry_file(entry.prompt)
        current = manifest["entries"].get(entry.entry_id)
        if current and current.get("prompt") != entry.prompt and not args.overwrite:
            raise ValueError(f"Entry prompt changed for {entry.entry_id}; use --overwrite or a new library")
        manifest["entries"][entry.entry_id] = {
            "file": str(relative_path),
            "kind": entry.kind,
            "layout": entry.layout,
            "prompt": entry.prompt,
            "prompt_sha256": hashlib.sha256(entry.prompt.encode("utf-8")).hexdigest(),
            "values": entry.values,
            "preset": entry.preset,
            "grid": entry.grid,
        }
        if args.overwrite or not (args.output_dir / relative_path).is_file():
            pending.append((entry, relative_path))

    negative_path = args.output_dir / manifest["negative_embedding"]
    needs_negative = args.overwrite or not negative_path.is_file()
    pipeline = load_encoder_pipeline(args, dtype) if needs_negative or pending else None
    if needs_negative:
        with torch.inference_mode():
            negative, _ = pipeline.encode_prompt(
                prompt="",
                device=args.device,
                max_sequence_length=args.max_sequence_length,
                text_encoder_out_layers=tuple(args.text_encoder_out_layers),
            )
        negative = negative.detach().to(device="cpu", dtype=dtype).contiguous()
        atomic_torch_save({"prompt_embeds": negative}, negative_path)
        manifest["embedding_shape"] = list(negative.shape)

    unique_pending = {}
    for entry, relative_path in pending:
        unique_pending[str(relative_path)] = (entry, relative_path)
    pending = list(unique_pending.values())
    print(f"entries={len(entries)}, pending_unique_files={len(pending)}", flush=True)
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        prompts = [entry.prompt for entry, _ in batch]
        with torch.inference_mode():
            embeddings, _ = pipeline.encode_prompt(
                prompt=prompts,
                device=args.device,
                max_sequence_length=args.max_sequence_length,
                text_encoder_out_layers=tuple(args.text_encoder_out_layers),
            )
        embeddings = embeddings.detach().to(device="cpu", dtype=dtype)
        if manifest["embedding_shape"] is None:
            manifest["embedding_shape"] = [1, *embeddings.shape[1:]]
        for offset, (_, relative_path) in enumerate(batch):
            tensor = embeddings[offset : offset + 1].contiguous()
            if list(tensor.shape) != manifest["embedding_shape"]:
                raise ValueError(
                    f"Embedding shape changed: {list(tensor.shape)} vs {manifest['embedding_shape']}"
                )
            atomic_torch_save({"prompt_embeds": tensor}, args.output_dir / relative_path)
        print(f"encoded={min(start + len(batch), len(pending))}/{len(pending)}", flush=True)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json_save(manifest, manifest_path)
    version = {key: value for key, value in manifest.items() if key != "entries"}
    version["entry_count"] = len(manifest["entries"])
    atomic_json_save(version, args.output_dir / "version.json")
    print(f"manifest={manifest_path}, entries={len(manifest['entries'])}", flush=True)


if __name__ == "__main__":
    main()
