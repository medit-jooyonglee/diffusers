"""FLUX.2 klein-4B(step-distilled, 4 step) 이미지 편집 일괄 생성.

입력 이미지 + 텍스트 프롬프트로 편집 결과를 만들어 distillation 학습 데이터로 쓴다.
klein-4B 스냅샷에는 텍스트 인코더 가중치가 없어 klein-base-4B의 Qwen3 인코더를 쓴다 (설정 파일이 같다).

입력 (둘 중 하나):
  --input-jsonl  한 줄에 {"id": "...", "image": "a.png" 또는 ["a.png", "ref.png"], "prompt": "...", "seed": 0(선택)}
                 image를 빼면 text-to-image ("width"/"height" 선택, 기본 1024)
  --images-dir + --prompt  폴더의 모든 이미지에 같은 프롬프트 (스모크 테스트용)

출력: <output-dir>/<id>.png, <output-dir>/results.jsonl (이미 있는 id는 건너뛴다 — 중단 후 다시 실행하면 이어서 생성)

예:
  python my/flux2_klein_edit_batch.py --images-dir my/samples/faces/samples --limit 2 \\
      --prompt "Make the teeth a natural bright white." --output-dir outputs/klein_edit_smoke
  python my/flux2_klein_edit_batch.py --input-jsonl edits.jsonl --output-dir outputs/klein_edit \\
      --fp8-compute --compile regional --num-shards 2 --shard-index 0 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# 텍스트 인코더와 transformer를 같은 GPU에 올리면 24GB에 빠듯해 메모리 조각화로 OOM이 나기 쉽다
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image, ImageOps
from transformers import AutoTokenizer, Qwen3ForCausalLM

from diffusers import Flux2KleinPipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smile_design.inference import load_transformer_override  # noqa: E402


DEFAULT_MODEL = (
    "/data1/jooyonglee/pretrained/flux2/models--black-forest-labs--FLUX.2-klein-4B/"
    "snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294"
)
DEFAULT_TEXT_ENCODER_MODEL = (
    "/data1/jooyonglee/pretrained/flux2/models--black-forest-labs--FLUX.2-klein-base-4B/"
    "snapshots/a3b4f4849157f664bdbc776fd7453c2783562f4d"
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-jsonl", type=Path)
    source.add_argument("--images-dir", type=Path)
    parser.add_argument("--prompt", help="--images-dir와 함께 쓰는 공통 프롬프트")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--text-encoder-model", default=DEFAULT_TEXT_ENCODER_MODEL)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="distilled 모델은 CFG를 쓰지 않는다")
    parser.add_argument("--seed", type=int, default=0, help="입력에 seed가 없을 때 쓰는 seed")
    parser.add_argument("--max-area", type=int, default=1024 * 1024, help="생성 해상도 면적 상한 (원본 비율 유지, 16의 배수)")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--fp8-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="transformer 가중치를 fp8로 저장, bf16으로 계산. 24GB GPU에서는 끄면 1024² 입력에서 OOM이 날 수 있다",
    )
    parser.add_argument("--fp8-compute", action="store_true", help="torchao fp8 연산 (--compile과 함께 써야 빨라진다)")
    parser.add_argument("--compile", nargs="?", const="full", choices=("full", "regional"))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, help="처음 N개만 (스모크 테스트용)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.images_dir and not args.prompt:
        parser.error("--images-dir requires --prompt")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    return args


def load_items(args: argparse.Namespace) -> list[dict]:
    if args.images_dir:
        paths = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        items = [{"id": p.stem, "image": [str(p)], "prompt": args.prompt} for p in paths]
    else:
        items = []
        base = args.input_jsonl.parent
        for number, line in enumerate(args.input_jsonl.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            # image가 없으면 text-to-image (width/height로 크기 지정, 기본 1024x1024)
            images = record.get("image") or []
            images = images if isinstance(images, list) else [images]
            images = [str(Path(p) if Path(p).is_absolute() else base / p) for p in images]
            items.append({"id": str(record.get("id", f"{number:06d}")), "image": images, "prompt": record["prompt"],
                          "seed": record.get("seed"), "width": record.get("width", 1024),
                          "height": record.get("height", 1024)})
    items = items[args.shard_index :: args.num_shards]
    return items[: args.limit] if args.limit else items


def generation_size(width: int, height: int, max_area: int, multiple: int = 16) -> tuple[int, int]:
    scale = min(1.0, math.sqrt(max_area / (width * height)))
    return (max(multiple, round(width * scale / multiple) * multiple), max(multiple, round(height * scale / multiple) * multiple))


def load_pipeline(args: argparse.Namespace) -> Flux2KleinPipeline:
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    transformer = load_transformer_override(args.model, None, dtype, True)
    kwargs = {"transformer": transformer} if transformer is not None else {}
    pipeline = Flux2KleinPipeline.from_pretrained(
        args.model,
        **kwargs,
        text_encoder=Qwen3ForCausalLM.from_pretrained(args.text_encoder_model, subfolder="text_encoder", dtype=dtype, local_files_only=True),
        tokenizer=AutoTokenizer.from_pretrained(args.text_encoder_model, subfolder="tokenizer", local_files_only=True),
        dtype=dtype,
        local_files_only=True,
    )
    if args.fp8_compute:
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow, quantize_

        quantize_(pipeline.transformer, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))
    elif args.fp8_weights:
        pipeline.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=dtype)
    pipeline.to(args.device)
    if args.compile == "regional":
        pipeline.transformer.compile_repeated_blocks(dynamic=True)
    elif args.compile == "full":
        pipeline.transformer.compile(dynamic=True)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def main() -> None:
    args = parse_args()
    items = load_items(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    done = set()
    if results_path.exists() and not args.overwrite:
        records = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        done = {r["id"] for r in records if r["status"] == "ok"}  # 실패한 건은 다시 시도한다
    todo = [item for item in items if item["id"] not in done]
    print(f"{len(items)} items in shard {args.shard_index}/{args.num_shards}, {len(done & {i['id'] for i in items})} done, {len(todo)} to run", flush=True)
    if not todo:
        return

    started = time.time()
    pipeline = load_pipeline(args)
    print(f"pipeline loaded in {time.time() - started:.1f}s", flush=True)

    with results_path.open("a", encoding="utf-8") as results, torch.inference_mode():
        for index, item in enumerate(todo, 1):
            record = {"id": item["id"], "prompt": item["prompt"], "image": item["image"]}
            try:
                images = [ImageOps.exif_transpose(Image.open(p)).convert("RGB") for p in item["image"]]
                source_size = images[0].size if images else (item["width"], item["height"])
                width, height = generation_size(*source_size, args.max_area)
                seed = args.seed if item.get("seed") is None else int(item["seed"])
                t0 = time.time()
                output = pipeline(
                    image=images or None,
                    prompt=item["prompt"],
                    height=height,
                    width=width,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    generator=torch.Generator(device=args.device).manual_seed(seed),
                ).images[0]
                elapsed = time.time() - t0
                path = args.output_dir / f"{item['id']}.png"
                output.save(path)
                record.update(status="ok", output=str(path), seed=seed, input_size=list(source_size),
                              size=[width, height], seconds=round(elapsed, 2))
            except Exception as error:  # 한 건이 실패해도 나머지는 계속 생성
                record.update(status="error", error=f"{type(error).__name__}: {error}")
            results.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.flush()
            print(f"[{index}/{len(todo)}] {item['id']} {record['status']} {record.get('seconds', '')}", flush=True)
    print(f"done in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
