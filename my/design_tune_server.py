from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import re
import subprocess
import sys
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import yaml
from flask import Flask, Response, jsonify, request, send_file
from PIL import Image, ImageFilter, ImageOps, PngImagePlugin

from diffusers import Flux2KleinInpaintPipeline, Flux2KleinPipeline

from smile_design.embedding_library import EmbeddingLibrary
from smile_design.face_mask import MASK_REGIONS, FaceMasker, parse_regions
from smile_design.mice import MiceAttention, MiceEdit, PromptLengthCounter
from smile_design.inference import load_transformer_override
from smile_design.prompt_templates import build_prompt_entries, load_experiment_config, reference_roles


ROOT = Path(__file__).resolve().parents[1]
app_logger = logging.getLogger("design_tune")
DEFAULT_MODEL = (
    "/data1/jooyonglee/pretrained/flux2/models--black-forest-labs--FLUX.2-klein-4B/"
    "snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294"
)
DEFAULT_TEXT_ENCODER_MODEL = (
    "/data1/jooyonglee/pretrained/flux2/models--black-forest-labs--FLUX.2-klein-base-4B/"
    "snapshots/a3b4f4849157f664bdbc776fd7453c2783562f4d"
)
STYLE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_error(message: str, status: int, **details: Any):
    payload = {"error": message}
    payload.update(details)
    return jsonify(payload), status


def parse_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def torch_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def resolve_mask_options(value: Any, args: argparse.Namespace) -> dict[str, Any]:
    # 요청의 parameters.mask: 생략하면 서버 기본값, true/false, 또는 옵션 객체
    # mode "inpaint" (기본): Flux2KleinInpaintPipeline으로 regions 영역만 latent 단계에서 실제로 재생성
    #   (docs/smile_tune/final_report.md 0장 — composite보다 훨씬 자연스럽고, 극단적 케이스도 복원됨).
    #   edits가 있으면 영역마다 순차적으로 진짜 inpaint를 적용한다 (MICE의 "여러 영역 동시 조합"을
    #   attention bias 대신 latent mask로 구현한 것 — 0.4절). 없으면 preset/anchor/grid/delta로 만든
    #   프롬프트 하나로 regions 전체를 한 번에 inpaint한다. layout에 두 번째 이미지(tooth 등)가
    #   있으면 자동으로 image_reference로 사용한다.
    # mode "mice": edits(영역별 지시문)를 attention bias로 영역에 묶어 한 번에 생성 (smile_design/mice.py).
    #   레거시 — inpaint의 edits 방식이 더 정확하고 단순하다 (0.4절). 여러 영역을 "한 번의 forward"로
    #   묶어야만 하는 경우에만 쓴다.
    # mode "composite": 전체 생성 후 regions 영역만 원본에 RGB로 합성 (레거시, inpaint보다 품질이 낮음)
    options = {
        "enabled": args.mask,
        "mode": args.mask_mode,
        "regions": parse_regions(args.mask_regions),
        "dilate": args.mask_dilate,
        "feather": args.mask_feather,
        "edits": [],
        "global": "default",
        "kernel_size": 11,
        "temperature": 3.0,
        "mice_steps": None,
        "composite": False,
        "text_tokens": 200,
        "strength": 1.0,
    }
    if value is None:
        return options
    if isinstance(value, bool):
        return {**options, "enabled": value}
    if not isinstance(value, dict):
        raise ValueError("parameters.mask must be a boolean or an object")
    unknown = sorted(set(value) - set(options))
    if unknown:
        raise ValueError(f"Unknown mask options {unknown}; use {sorted(options)}")
    if "enabled" in value:
        options["enabled"] = bool(value["enabled"])
    else:
        options["enabled"] = True
    if "regions" in value:
        options["regions"] = parse_regions(value["regions"])
    if "dilate" in value:
        options["dilate"] = int(value["dilate"])
    if "feather" in value:
        options["feather"] = float(value["feather"])
    if "mode" in value:
        if value["mode"] not in ("composite", "mice", "inpaint"):
            raise ValueError("mask.mode must be 'composite', 'mice', or 'inpaint'")
        options["mode"] = value["mode"]
    if options["mode"] == "inpaint" and "strength" in value:
        options["strength"] = float(value["strength"])
    if not 0.0 < options["strength"] <= 1.0:
        raise ValueError("mask.strength must be in (0, 1]")
    if options["mode"] in ("mice", "inpaint") and "edits" in value:
        edits = value["edits"]
        if not isinstance(edits, list) or not edits or not all(isinstance(e, dict) and "prompt" in e for e in edits):
            raise ValueError('mask.edits must be a non-empty list of {"prompt": name, "region": optional}')
        options["edits"] = edits
    if options["mode"] == "mice" and not options["edits"]:
        raise ValueError('mask.mode "mice" requires a non-empty mask.edits')
    if options["mode"] == "mice":
        options["global"] = str(value.get("global", options["global"]))
        options["kernel_size"] = int(value.get("kernel_size", options["kernel_size"]))
        options["temperature"] = float(value.get("temperature", options["temperature"]))
        options["mice_steps"] = None if value.get("mice_steps") is None else int(value["mice_steps"])
        options["composite"] = bool(value.get("composite", False))
        # 지시문당 사용할 텍스트 토큰 수 (padding 포함). 0이면 실제 토큰만 사용 (품질이 떨어짐, 실험 결과)
        options["text_tokens"] = int(value.get("text_tokens", options["text_tokens"]))
    return options


def safe_style_name(name: str) -> str:
    if not STYLE_NAME_PATTERN.fullmatch(name):
        raise ValueError("Style name must use only letters, numbers, dot, underscore, or hyphen")
    return name


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


class StyleStore:
    def __init__(self, styles_dir: Path, libraries_dir: Path):
        self.styles_dir = styles_dir.resolve()
        self.libraries_dir = libraries_dir.resolve()
        self.styles_dir.mkdir(parents=True, exist_ok=True)
        self.libraries_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def _style_files(self) -> list[Path]:
        return sorted(set(self.styles_dir.glob("*.yaml")) | set(self.styles_dir.glob("*.yml")))

    def _metadata(self, path: Path) -> dict[str, Any]:
        config, config_hash = load_experiment_config(path)
        return {
            "name": str(config.get("template_version") or path.stem),
            "file_name": path.name,
            "config_hash": config_hash,
            "template_version": config["template_version"],
            "layouts": list(config["layouts"]),
            "controls": list(config["controls"]),
            "presets": list(config["presets"]),
            "grids": list(config["grids"]),
            "updated_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        }

    def resolve(self, name: str) -> Path:
        safe_style_name(name)
        direct = [self.styles_dir / f"{name}.yaml", self.styles_dir / f"{name}.yml"]
        alternate = name.replace("-", "_")
        direct.extend([self.styles_dir / f"{alternate}.yaml", self.styles_dir / f"{alternate}.yml"])
        for path in direct:
            if path.is_file():
                return path
        for path in self._style_files():
            try:
                config, _ = load_experiment_config(path)
            except Exception:
                continue
            if config.get("template_version") == name:
                return path
        raise FileNotFoundError(f"Style not found: {name}")

    def list(self) -> list[dict[str, Any]]:
        result = []
        for path in self._style_files():
            try:
                metadata = self._metadata(path)
                library = self.library_path(metadata["name"], path)
                metadata["library_ready"] = (library / "manifest.json").is_file()
                metadata["library"] = str(library)
                result.append(metadata)
            except Exception as error:
                result.append({"name": path.stem, "file_name": path.name, "valid": False, "error": str(error)})
        return result

    def read(self, name: str) -> tuple[Path, str, dict[str, Any], str]:
        with self.lock:
            path = self.resolve(name)
            content = path.read_text(encoding="utf-8")
            config, config_hash = load_experiment_config(path)
            return path, content, config, config_hash

    def validate_text(self, content: str) -> tuple[dict[str, Any], str]:
        temporary = self.styles_dir / f".validate-{uuid.uuid4().hex}.yaml"
        try:
            temporary.write_text(content, encoding="utf-8")
            config, config_hash = load_experiment_config(temporary)
            build_prompt_entries(config)
            for layout in config["layouts"]:
                reference_roles(config, layout)
            return config, config_hash
        finally:
            temporary.unlink(missing_ok=True)

    def save(self, name: str, content: str) -> tuple[Path, dict[str, Any], str]:
        safe_style_name(name)
        config, config_hash = self.validate_text(content)
        with self.lock:
            try:
                path = self.resolve(name)
            except FileNotFoundError:
                path = self.styles_dir / f"{name}.yaml"
            atomic_write_text(path, content)
        return path, config, config_hash

    def delete(self, name: str) -> Path:
        with self.lock:
            path = self.resolve(name)
            path.unlink()
            return path

    def library_path(self, name: str, style_path: Path | None = None) -> Path:
        safe_style_name(name)
        path = style_path or self.resolve(name)
        _, config_hash = load_experiment_config(path)
        # 요청 이름(smile_v1 / smile-v1 / template_version)과 무관하게 YAML 파일 이름으로 경로를 통일
        return self.libraries_dir / path.stem / config_hash


@dataclass
class BuildJob:
    job_id: str
    style_name: str
    library: str
    status: str = "queued"
    created_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    error: str | None = None
    log: str = ""
    skipped: bool = False  # 이미 빌드된 라이브러리라 빌드 프로세스를 띄우지 않음

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "style_name": self.style_name,
            "library": self.library,
            "status": self.status,
            "skipped": self.skipped,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "return_code": self.return_code,
            "error": self.error,
            "log": self.log,
        }


def _library_complete(library: Path, config: dict[str, Any], layouts: list[str] | None, kinds: list[str] | None) -> bool:
    """manifest에 요청 범위의 모든 항목이 있고 파일도 남아 있으면 True."""
    manifest_path = library / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = build_prompt_entries(config, layouts=layouts, kinds=set(kinds) if kinds else None)
    except (ValueError, json.JSONDecodeError):
        return False
    entries = manifest.get("entries", {})
    for entry in expected:
        record = entries.get(entry.entry_id)
        if record is None or record.get("prompt") != entry.prompt or not (library / record["file"]).is_file():
            return False
    return (library / manifest.get("negative_embedding", "negative.pt")).is_file()


class BuildManager:
    def __init__(self, args: argparse.Namespace, store: StyleStore):
        self.args = args
        self.store = store
        self.jobs: dict[str, BuildJob] = {}
        self.lock = threading.RLock()
        self.gpu_lock: threading.RLock | None = None

    def start(self, style_name: str, options: dict[str, Any]) -> BuildJob:
        style_path, style_content, config, _ = self.store.read(style_name)
        library = self.store.library_path(style_name, style_path)
        layouts = options.get("layouts")
        kinds = options.get("kinds")
        if layouts is not None:
            if not isinstance(layouts, list) or not all(isinstance(item, str) for item in layouts):
                raise ValueError("layouts must be a list of strings")
            unknown = set(layouts) - set(config["layouts"])
            if unknown:
                raise ValueError(f"Unknown layouts: {sorted(unknown)}")
        if kinds is not None:
            allowed = {"base", "preset", "anchor", "grid", "mice"}
            if not isinstance(kinds, list) or not all(isinstance(item, str) for item in kinds):
                raise ValueError("kinds must be a list of strings")
            unknown = set(kinds) - allowed
            if unknown:
                raise ValueError(f"Unknown embedding kinds: {sorted(unknown)}")
        batch_size = int(options.get("batch_size", self.args.build_batch_size))
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        overwrite = bool(options.get("overwrite", False))
        with self.lock:
            # 같은 라이브러리(= 같은 YAML 내용)를 빌드 중이면 새 프로세스를 띄우지 않고 그 작업을 돌려준다
            for existing in self.jobs.values():
                if existing.library == str(library) and existing.status in {"queued", "running"}:
                    return existing
            if not overwrite and _library_complete(library, config, layouts, kinds):
                job = BuildJob(uuid.uuid4().hex, style_name, str(library), status="succeeded", skipped=True)
                job.started_at = job.finished_at = utc_now()
                job.log = "Embedding library is already built for this style content; use overwrite to rebuild"
                self.jobs[job.job_id] = job
                return job
            # 확인과 등록을 같은 락 안에서 해야 동시 요청이 둘 다 통과하지 않는다
            job = BuildJob(uuid.uuid4().hex, style_name, str(library))
            self.jobs[job.job_id] = job
        snapshot_path = library / "style.yaml"
        atomic_write_text(snapshot_path, style_content)
        thread = threading.Thread(
            target=self._run,
            args=(
                job,
                snapshot_path,
                layouts or [],
                kinds or [],
                batch_size,
                overwrite,
            ),
            daemon=True,
            name=f"embedding-build-{job.job_id[:8]}",
        )
        thread.start()
        return job

    def get(self, job_id: str) -> BuildJob:
        with self.lock:
            try:
                return self.jobs[job_id]
            except KeyError as error:
                raise FileNotFoundError(f"Build job not found: {job_id}") from error

    def _run(
        self,
        job: BuildJob,
        style_path: Path,
        layouts: list[str],
        kinds: list[str],
        batch_size: int,
        overwrite: bool,
    ) -> None:
        job.status = "running"
        job.started_at = utc_now()
        command = [
            sys.executable,
            str(ROOT / "my" / "build_smile_embedding_library.py"),
            "--model",
            self.args.model,
            "--text-encoder-model",
            self.args.text_encoder_model,
            "--config",
            str(style_path),
            "--output-dir",
            job.library,
            "--batch-size",
            str(batch_size),
            "--dtype",
            self.args.dtype,
            "--device",
            self.args.build_device or self.args.device,
        ]
        if self.args.local_files_only:
            command.append("--local-files-only")
        else:
            command.append("--no-local-files-only")
        for layout in layouts:
            command.extend(["--layout", layout])
        for kind in kinds:
            command.extend(["--kind", kind])
        if overwrite:
            command.append("--overwrite")
        try:
            # 빌드는 별도 프로세스라 종료 시 text encoder VRAM이 모두 반환된다.
            # 추론과 같은 GPU에서 빌드할 때만 GPU 락을 잡는다 (다른 GPU면 빌드 중에도 추론 가능)
            same_device = (self.args.build_device or self.args.device) == self.args.device
            lock = (self.gpu_lock if same_device else None) or threading.RLock()
            with lock:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=self.args.build_timeout,
                    check=False,
                )
            job.return_code = completed.returncode
            job.log = completed.stdout[-self.args.job_log_characters :]
            if completed.returncode == 0:
                job.status = "succeeded"
            else:
                job.status = "failed"
                job.error = f"Embedding builder exited with code {completed.returncode}"
        except subprocess.TimeoutExpired as error:
            job.status = "failed"
            job.error = f"Embedding build exceeded {self.args.build_timeout} seconds"
            output = error.stdout or ""
            job.log = output[-self.args.job_log_characters :]
        except Exception as error:
            job.status = "failed"
            job.error = str(error)
            job.log = traceback.format_exc()[-self.args.job_log_characters :]
        finally:
            job.finished_at = utc_now()


class InferenceEngine:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.dtype = torch_dtype(args.dtype)
        self.pipeline: Flux2KleinPipeline | None = None
        self.inpaint_pipeline: Flux2KleinInpaintPipeline | None = None
        self.lock = threading.RLock()
        self.masker = FaceMasker(args.face_landmarker_model)
        self.prompt_length: PromptLengthCounter | None = None

    @property
    def loaded(self) -> bool:
        return self.pipeline is not None

    def load(self) -> Flux2KleinPipeline:
        with self.lock:
            if self.pipeline is not None:
                return self.pipeline
            transformer = load_transformer_override(
                self.args.model,
                self.args.revision,
                self.dtype,
                self.args.local_files_only,
            )
            # load_transformer_override는 root에 단일 safetensors 파일뿐인 체크포인트(klein-4B)에서만
            # override를 반환한다. 이미 diffusers 폴더 형식(예: klein-base-4B)이면 None을 반환하는데,
            # from_pretrained에 transformer=None을 그대로 넘기면 "이 컴포넌트는 없다"로 해석되어
            # transformer가 아예 로드되지 않는다 — kwarg 자체를 생략해 정상적으로 로드되게 한다.
            transformer_kwargs = {"transformer": transformer} if transformer is not None else {}
            pipeline = Flux2KleinPipeline.from_pretrained(
                self.args.model,
                revision=self.args.revision,
                **transformer_kwargs,
                text_encoder=None,
                tokenizer=None,
                dtype=self.dtype,
                local_files_only=self.args.local_files_only,
            )
            if pipeline.text_encoder is not None or pipeline.tokenizer is not None:
                raise RuntimeError("Inference pipeline unexpectedly loaded a text encoder or tokenizer")
            if self.args.fp8_weights:
                # transformer 가중치를 fp8로 보관하고 레이어 계산 시에만 bf16으로 올림 (VRAM 약 3.8GB 절약)
                pipeline.transformer.enable_layerwise_casting(
                    storage_dtype=torch.float8_e4m3fn, compute_dtype=self.dtype
                )
            if self.args.cpu_offload:
                if not self.args.device.startswith("cuda"):
                    raise ValueError("CPU offload requires a CUDA device")
                gpu_id = int(self.args.device.split(":", 1)[1]) if ":" in self.args.device else 0
                pipeline.enable_model_cpu_offload(gpu_id=gpu_id)
            else:
                pipeline.to(self.args.device)
            pipeline.set_progress_bar_config(dynamic_ncols=True)
            self.pipeline = pipeline
            return pipeline

    def load_inpaint(self) -> Flux2KleinInpaintPipeline:
        # 메인 파이프라인과 transformer/vae 모듈 인스턴스를 그대로 공유한다 (fp8 캐스팅 포함, VRAM 추가 없음)
        with self.lock:
            if self.inpaint_pipeline is not None:
                return self.inpaint_pipeline
            pipeline = self.load()
            inpaint = Flux2KleinInpaintPipeline(
                scheduler=pipeline.scheduler,
                vae=pipeline.vae,
                text_encoder=None,
                tokenizer=None,
                transformer=pipeline.transformer,
                is_distilled=pipeline.config.get("is_distilled", False),
            )
            inpaint.to(self.args.device) if not self.args.cpu_offload else None
            inpaint.set_progress_bar_config(dynamic_ncols=True)
            self.inpaint_pipeline = inpaint
            return inpaint

    def infer(
        self,
        library: EmbeddingLibrary,
        config: dict[str, Any],
        parameters: dict[str, Any],
        images: list[Image.Image],
    ) -> tuple[Image.Image, dict[str, Any]]:
        layout = parameters.get("layout")
        if not isinstance(layout, str) or layout not in config["layouts"]:
            raise ValueError(f"Unknown or missing layout: {layout!r}")
        roles = reference_roles(config, layout)
        if len(images) != len(roles):
            raise ValueError(f"Layout {layout!r} requires {len(roles)} images in order {roles}, got {len(images)}")
        mask_options = resolve_mask_options(parameters.get("mask"), self.args)
        is_inpaint = mask_options["enabled"] and mask_options["mode"] == "inpaint"
        if is_inpaint and mask_options["edits"]:
            # 여러 영역을 한 요청에서 조합: attention bias(mice) 대신 영역마다 순차적인 진짜 inpaint로 처리한다
            return self._inpaint_sequential(library, config, layout, images, parameters, mask_options)
        mask_image, mask_metadata, mice = None, {"enabled": False}, None
        if mask_options["enabled"] and mask_options["mode"] == "mice":
            mice, mask_image, mask_metadata = self._mice(library, config, layout, images[0], mask_options)
            prompt_embeds, mix = mice.prompt_embeds, {"mode": "mice"}
        else:
            if mask_options["enabled"]:
                # 생성 전에 마스크를 만들어 얼굴이 없으면 GPU 추론 없이 바로 실패
                mask_image, mask_metadata = self.masker.build(
                    images[0], mask_options["regions"], mask_options["dilate"], mask_options["feather"]
                )
                if is_inpaint:
                    mask_metadata["mode"] = "inpaint"
                    mask_metadata["strength"] = mask_options["strength"]
            prompt_embeds, mix = self._embedding(library, layout, parameters)
        original_size = images[0].size
        width, height = internal_size(
            original_size[0], original_size[1], self.args.dimension_multiple, self.args.max_generation_area
        )
        pipeline = self.load_inpaint() if is_inpaint else self.load()
        steps = int(parameters.get("num_steps", parameters.get("steps", self.args.steps)))
        guidance_scale = float(parameters.get("guidance_scale", self.args.guidance_scale))
        # step-distilled 모델은 CFG를 쓰지 않으므로 negative 임베딩을 GPU에 올리지 않음
        negative_prompt_embeds = None
        if guidance_scale > 1.0 and not pipeline.config.get("is_distilled", False):
            negative_prompt_embeds = library.load_negative(self.args.device, self.dtype)
        seed = int(parameters.get("seed", 0))
        if steps < 1:
            raise ValueError("steps must be at least 1")
        step_warning = None
        if not pipeline.config.get("is_distilled", False) and steps < self.args.min_steps_non_distilled:
            # klein-4B(distilled)는 4 step용으로 따로 학습된 가중치라 적은 step에서도 수렴하지만,
            # klein-base-4B 같은 비증류 모델은 같은 step 수로는 뭉개진 결과만 나온다 (실측 확인, final_report.md 0장)
            step_warning = (
                f"Model is not step-distilled but num_inference_steps={steps} "
                f"(< {self.args.min_steps_non_distilled}); expect blurry/incoherent output. "
                "Use the distilled klein-4B checkpoint for low step counts, or raise steps for this model."
            )
            app_logger.warning(f"[design_tune] {step_warning}")
        generator = torch.Generator(device=self.args.device).manual_seed(seed)
        if is_inpaint:
            # Flux2KleinInpaintPipeline: mask_image가 latent 단계에서 영역을 제한한다 (RGB 후처리 합성보다 정확함).
            # layout에 두 번째 이미지가 있으면(예: patient_tooth) 치아 모양 참조로 그대로 넘긴다
            call_kwargs = {
                "image": images[0],
                "mask_image": mask_image,
                "image_reference": images[1] if len(images) > 1 else None,
                "strength": mask_options["strength"],
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
                "height": height,
                "width": width,
                "num_images_per_prompt": 1,
                "generator": generator,
            }
        else:
            call_kwargs = {
                "image": images,
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
                "height": height,
                "width": width,
                "num_images_per_prompt": 1,
                "generator": generator,
            }
        with self.lock, torch.inference_mode():
            hook = mice.install(pipeline.transformer) if mice else None
            try:
                generated = pipeline(**call_kwargs).images[0]
            finally:
                if hook is not None:
                    hook.remove()
        if generated.size != original_size:
            generated = generated.resize(original_size, Image.Resampling.LANCZOS)
        if mask_image is not None:
            # 마스크 밖은 원본 환자 픽셀을 그대로 유지 (inpaint도 resize 과정의 오차를 없애기 위해 한 번 더 적용)
            generated = Image.composite(generated.convert("RGB"), images[0], mask_image)
        metadata = {
            "style_name": parameters["style_name"],
            "layout": layout,
            "roles": roles,
            "mix": mix,
            "seed": seed,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "input_size": list(original_size),
            "internal_size": [width, height],
            "output_size": list(generated.size),
            "text_encoder_loaded": False,
            "mask": mask_metadata,
            "step_warning": step_warning,
        }
        return generated, metadata

    def _inpaint_sequential(
        self,
        library: EmbeddingLibrary,
        config: dict[str, Any],
        layout: str,
        images: list[Image.Image],
        parameters: dict[str, Any],
        mask_options: dict[str, Any],
    ) -> tuple[Image.Image, dict[str, Any]]:
        # mode "inpaint" + edits: MICE의 "여러 영역을 한 번에 조합" 역할을 attention bias 대신 영역마다
        # 순차적인 진짜 inpaint 호출로 구현한다 (final_report.md 0.4절). 각 호출은 자기 영역의 텍스트
        # 임베딩만 쓰고 (MICE처럼 전역 지시문과 합치지 않음 — 바깥 픽셀은 latent mask가 이미 보존한다),
        # 이전 호출의 출력 위에 이어서 다음 영역을 inpaint한다. 끝나면 전체 영역 합집합으로 원본과
        # 한 번 더 합성해 바깥 픽셀을 보존한다 (resize 오차 방지).
        regions_config = config.get("mice", {}).get("regions", {})
        if not regions_config:
            raise ValueError("This style has no mice section; add mice.global / mice.regions and rebuild")
        roles = reference_roles(config, layout)
        patient = images[0]
        reference = images[1] if len(images) > 1 else None
        original_size = patient.size
        width, height = internal_size(
            original_size[0], original_size[1], self.args.dimension_multiple, self.args.max_generation_area
        )
        pipeline = self.load_inpaint()
        steps = int(parameters.get("num_steps", parameters.get("steps", self.args.steps)))
        guidance_scale = float(parameters.get("guidance_scale", self.args.guidance_scale))
        negative_prompt_embeds = None
        if guidance_scale > 1.0 and not pipeline.config.get("is_distilled", False):
            negative_prompt_embeds = library.load_negative(self.args.device, self.dtype)
        seed = int(parameters.get("seed", 0))
        if steps < 1:
            raise ValueError("steps must be at least 1")
        step_warning = None
        if not pipeline.config.get("is_distilled", False) and steps < self.args.min_steps_non_distilled:
            step_warning = (
                f"Model is not step-distilled but num_inference_steps={steps} "
                f"(< {self.args.min_steps_non_distilled}); expect blurry/incoherent output."
            )
            app_logger.warning(f"[design_tune] {step_warning}")
        masks, _ = self.masker.region_masks(patient)
        current, union, edit_metadata = patient, None, []
        with self.lock, torch.inference_mode():
            for index, edit in enumerate(mask_options["edits"]):
                name = str(edit["prompt"])
                if name not in regions_config:
                    raise ValueError(f"Unknown mice region prompt {name!r}; choose from {sorted(regions_config)}")
                regions = parse_regions(edit.get("region", regions_config[name]["region"]))
                mask = self.masker.union(masks, regions, int(edit.get("dilate", mask_options["dilate"])))
                mask_image = Image.fromarray(mask, mode="L").filter(ImageFilter.GaussianBlur(mask_options["feather"]))
                embeds = library.load(f"{layout}/mice/region/{name}", self.args.device, self.dtype)
                generator = torch.Generator(device=self.args.device).manual_seed(seed + index)
                current = pipeline(
                    image=current,
                    mask_image=mask_image,
                    image_reference=reference,
                    strength=mask_options["strength"],
                    prompt_embeds=embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    num_images_per_prompt=1,
                    generator=generator,
                ).images[0]
                if current.size != original_size:
                    current = current.resize(original_size, Image.Resampling.LANCZOS)
                union = mask if union is None else union | mask
                edit_metadata.append({"prompt": name, "regions": regions, "mask_pixels": int((mask > 0).sum())})
        union_mask = Image.fromarray(union, mode="L").filter(ImageFilter.GaussianBlur(mask_options["feather"]))
        generated = Image.composite(current.convert("RGB"), patient, union_mask)
        metadata = {
            "style_name": parameters["style_name"],
            "layout": layout,
            "roles": roles,
            "mix": {"mode": "inpaint_sequential", "edits": [e["prompt"] for e in edit_metadata]},
            "seed": seed,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "input_size": list(original_size),
            "internal_size": [width, height],
            "output_size": list(generated.size),
            "text_encoder_loaded": False,
            "mask": {
                "enabled": True,
                "mode": "inpaint",
                "strength": mask_options["strength"],
                "sequential_edits": edit_metadata,
            },
            "step_warning": step_warning,
        }
        return generated, metadata

    def _mice(
        self,
        library: EmbeddingLibrary,
        config: dict[str, Any],
        layout: str,
        image: Image.Image,
        options: dict[str, Any],
    ) -> tuple[MiceAttention, Image.Image | None, dict[str, Any]]:
        regions_config = config.get("mice", {}).get("regions", {})
        if not regions_config:
            raise ValueError("This style has no mice section; add mice.global / mice.regions and rebuild")
        if not options["text_tokens"] and self.prompt_length is None:
            self.prompt_length = PromptLengthCounter(self.args.text_encoder_model)

        def trimmed(entry_id: str) -> torch.Tensor:
            embeds = library.load(entry_id, self.args.device, self.dtype)
            if options["text_tokens"]:
                return embeds[:, : options["text_tokens"]]
            return embeds[:, : self.prompt_length(library.prompt_for(entry_id), embeds.shape[1])]

        masks, _ = self.masker.region_masks(image)
        edits, union, edit_metadata = [], None, []
        for edit in options["edits"]:
            name = str(edit["prompt"])
            if name not in regions_config:
                raise ValueError(f"Unknown mice region prompt {name!r}; choose from {sorted(regions_config)}")
            regions = parse_regions(edit.get("region", regions_config[name]["region"]))
            mask = self.masker.union(masks, regions, int(edit.get("dilate", options["dilate"])))
            edits.append(MiceEdit(mask, trimmed(f"{layout}/mice/region/{name}")))
            union = mask if union is None else union | mask
            edit_metadata.append({"prompt": name, "regions": regions, "mask_pixels": int((mask > 0).sum())})
        mice = MiceAttention(
            trimmed(f"{layout}/mice/global/{options['global']}"),
            edits,
            kernel_size=options["kernel_size"],
            temperature=options["temperature"],
            steps=options["mice_steps"],
        )
        mask_image = None
        if options["composite"]:
            mask_image = Image.fromarray(union, mode="L").filter(ImageFilter.GaussianBlur(radius=options["feather"]))
        metadata = {
            "enabled": True,
            "mode": "mice",
            "global": options["global"],
            "edits": edit_metadata,
            "kernel_size": options["kernel_size"],
            "temperature": options["temperature"],
            "mice_steps": options["mice_steps"],
            "composite": options["composite"],
            "text_tokens_per_prompt": options["text_tokens"],
            "text_tokens": int(mice.prompt_embeds.shape[1]),
        }
        return mice, mask_image, metadata

    def _embedding(
        self,
        library: EmbeddingLibrary,
        layout: str,
        parameters: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        mode = str(parameters.get("mode", "preset"))
        values = parameters.get("control_values", {})
        if not isinstance(values, dict):
            raise ValueError("control_values must be an object")
        normalized_values = {str(key): float(value) for key, value in values.items()}
        if mode == "preset":
            preset = parameters.get("preset", "natural")
            return library.preset(layout, str(preset), self.args.device, self.dtype), {
                "mode": mode,
                "preset": preset,
            }
        if mode == "anchor":
            attribute = parameters.get("attribute")
            value = parameters.get("value")
            if isinstance(attribute, str) and value is None:
                value = normalized_values.get(attribute)
            if not isinstance(attribute, str) or value is None:
                raise ValueError("anchor mode requires attribute and its value in control_values")
            return library.interpolate_anchor(
                layout, attribute, float(value), self.args.device, self.dtype
            ), {"mode": mode, "attribute": attribute, "value": float(value)}
        if mode == "grid":
            grid = str(parameters.get("grid", ""))
            return library.interpolate_grid(
                layout, grid, normalized_values, self.args.device, self.dtype
            ), {"mode": mode, "grid": grid, "values": normalized_values}
        if mode == "delta":
            strengths = parameters.get("delta_strengths", {})
            if not isinstance(strengths, dict):
                raise ValueError("delta_strengths must be an object")
            normalized_strengths = {str(key): float(value) for key, value in strengths.items()}
            return library.compose_deltas(
                layout,
                normalized_values,
                normalized_strengths,
                self.args.device,
                self.dtype,
            ), {"mode": mode, "values": normalized_values, "strengths": normalized_strengths}
        raise ValueError(f"Unknown inference mode: {mode}")


def internal_size(width: int, height: int, multiple: int, max_area: int) -> tuple[int, int]:
    if width < 1 or height < 1:
        raise ValueError(f"Invalid input size: {width}x{height}")
    scale = 1.0
    if max_area and width * height > max_area:
        scale = math.sqrt(max_area / (width * height))
    scaled_width = max(multiple, int(width * scale))
    scaled_height = max(multiple, int(height * scale))
    adjusted_width = max(multiple, round(scaled_width / multiple) * multiple)
    adjusted_height = max(multiple, round(scaled_height / multiple) * multiple)
    return adjusted_width, adjusted_height


def parse_parameters() -> dict[str, Any]:
    raw = request.form.get("parameters")
    if raw is None:
        raise ValueError("Missing multipart form field: parameters")
    try:
        parameters = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"parameters is not valid JSON: {error}") from error
    if not isinstance(parameters, dict):
        raise ValueError("parameters must contain a JSON object")
    if not isinstance(parameters.get("style_name"), str):
        raise ValueError("parameters.style_name is required")
    return parameters


def load_uploaded_images(max_images: int) -> list[Image.Image]:
    files = request.files.getlist("images")
    if not files:
        raise ValueError("At least one multipart images part is required")
    if len(files) > max_images:
        raise ValueError(f"At most {max_images} images are allowed")
    images = []
    for index, upload in enumerate(files):
        if not upload.filename:
            raise ValueError(f"Image {index} has no filename")
        try:
            image = ImageOps.exif_transpose(Image.open(upload.stream)).convert("RGB")
            image.load()
        except Exception as error:
            raise ValueError(f"Could not decode image {index}: {upload.filename}") from error
        images.append(image)
    return images


def create_app(args: argparse.Namespace | None = None) -> Flask:
    args = args or parse_args([])
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = args.max_request_megabytes * 1024 * 1024
    store = StyleStore(args.styles_dir, args.libraries_dir)
    engine = InferenceEngine(args)
    builds = BuildManager(args, store)
    builds.gpu_lock = engine.lock
    app.extensions["design_tune"] = {"store": store, "engine": engine, "builds": builds, "args": args}

    @app.get("/health")
    def health():
        running = sum(job.status in {"queued", "running"} for job in builds.jobs.values())
        return jsonify(
            {
                "status": "ok",
                "model_loaded": engine.loaded,
                "model": args.model,
                "device": args.device,
                "dtype": args.dtype,
                "running_build_jobs": running,
                "fp8_weights": args.fp8_weights,
                "min_steps_non_distilled": args.min_steps_non_distilled,
                "mask": {
                    "enabled": args.mask,
                    "mode": args.mask_mode,
                    "regions": parse_regions(args.mask_regions),
                    "dilate": args.mask_dilate,
                    "feather": args.mask_feather,
                    "available_regions": list(MASK_REGIONS),
                },
            }
        )

    @app.post("/api/v1/mask")
    def preview_mask():
        # 생성 없이 첫 번째 이미지의 마스크만 확인 (overlay=true면 원본 위에 빨간색으로 겹쳐 반환)
        try:
            raw = request.form.get("parameters", "{}")
            parameters = json.loads(raw)
            if not isinstance(parameters, dict):
                raise ValueError("parameters must contain a JSON object")
            mask_value = parameters.get("mask")
            options = resolve_mask_options(mask_value if isinstance(mask_value, dict) else {}, args)
            image = load_uploaded_images(args.max_images)[0]
            mask, metadata = engine.masker.build(image, options["regions"], options["dilate"], options["feather"])
            if parameters.get("overlay"):
                red = Image.new("RGB", image.size, (255, 0, 0))
                mask = Image.composite(Image.blend(image, red, 0.5), image, mask)
            output = io.BytesIO()
            mask.save(output, format="PNG")
            output.seek(0)
            response = send_file(output, mimetype="image/png", download_name="mask.png")
            response.headers["X-Design-Tune-Mask"] = json.dumps(metadata, ensure_ascii=True)
            return response
        except json.JSONDecodeError as error:
            return json_error(f"parameters is not valid JSON: {error}", 400)
        except FileNotFoundError as error:
            return json_error(str(error), 404)
        except (ValueError, TypeError) as error:
            return json_error(str(error), 400)

    @app.get("/api/v1/styles")
    def list_styles():
        return jsonify({"styles": store.list()})

    @app.get("/api/v1/styles/<name>")
    def get_style(name: str):
        try:
            _, content, _, _ = store.read(name)
        except FileNotFoundError as error:
            return json_error(str(error), 404)
        if "application/json" in request.accept_mimetypes:
            return jsonify({"name": name, "yaml": content})
        return Response(content, content_type="text/yaml; charset=utf-8")

    @app.put("/api/v1/styles/<name>")
    def put_style(name: str):
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("yaml"), str):
            return json_error('Body must be JSON with a string field named "yaml"', 400)
        try:
            path, config, config_hash = store.save(name, payload["yaml"])
        except (ValueError, yaml.YAMLError) as error:
            return json_error(str(error), 400)
        return jsonify(
            {
                "name": name,
                "file": str(path),
                "template_version": config["template_version"],
                "config_hash": config_hash,
                "library_ready": store.library_path(name, path).joinpath("manifest.json").is_file(),
            }
        )

    @app.delete("/api/v1/styles/<name>")
    def delete_style(name: str):
        try:
            path = store.delete(name)
        except FileNotFoundError as error:
            return json_error(str(error), 404)
        return jsonify({"name": name, "deleted": True, "file": str(path), "embedding_cache_retained": True})

    @app.post("/api/v1/styles/<name>/build")
    def build_style(name: str):
        options = request.get_json(silent=True) or {}
        if not isinstance(options, dict):
            return json_error("Build body must be a JSON object", 400)
        try:
            job = builds.start(name, options)
        except FileNotFoundError as error:
            return json_error(str(error), 404)
        except ValueError as error:
            return json_error(str(error), 400)
        return jsonify(job.as_dict()), 200 if job.skipped else 202

    @app.get("/api/v1/jobs/<job_id>")
    def get_job(job_id: str):
        try:
            job = builds.get(job_id)
        except FileNotFoundError as error:
            return json_error(str(error), 404)
        return jsonify(job.as_dict())

    @app.post("/api/v1/infer")
    def infer():
        try:
            parameters = parse_parameters()
            style_name = safe_style_name(parameters["style_name"])
            style_path, _, config, _ = store.read(style_name)
            library_path = store.library_path(style_name, style_path)
            if not (library_path / "manifest.json").is_file():
                return json_error(
                    f"Embedding library is not built for style {style_name}",
                    409,
                    build_endpoint=f"/api/v1/styles/{style_name}/build",
                    expected_library=str(library_path),
                )
            library = EmbeddingLibrary(library_path)
            if library.manifest.get("model") != args.model:
                return json_error(
                    "Embedding library model does not match the server model",
                    409,
                    library_model=library.manifest.get("model"),
                    server_model=args.model,
                )
            images = load_uploaded_images(args.max_images)
            generated, metadata = engine.infer(library, config, parameters, images)
            output = io.BytesIO()
            png_info = PngImagePlugin.PngInfo()
            png_info.add_text("design_tune", json.dumps(metadata, ensure_ascii=False))
            generated.save(output, format="PNG", pnginfo=png_info)
            output.seek(0)
            response = send_file(output, mimetype="image/png", download_name="design_tune.png")
            response.headers["X-Design-Tune-Metadata"] = json.dumps(metadata, ensure_ascii=True)
            return response
        except FileNotFoundError as error:
            return json_error(str(error), 404)
        except (ValueError, TypeError, KeyError) as error:
            return json_error(str(error), 400)
        except torch.OutOfMemoryError as error:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return json_error(f"CUDA out of memory: {error}", 507)
        except Exception as error:
            app.logger.exception("Design Tune inference failed")
            return json_error(f"Inference failed: {error}", 500)

    @app.errorhandler(413)
    def request_too_large(_):
        return json_error(f"Request exceeds {args.max_request_megabytes} MiB", 413)

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--text-encoder-model", default=DEFAULT_TEXT_ENCODER_MODEL)
    parser.add_argument("--styles-dir", type=Path, default=ROOT / "my" / "configs" / "smile_design")
    parser.add_argument(
        "--libraries-dir", type=Path, default=ROOT / "outputs" / "design_tune_embedding_libraries"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--build-device",
        help="GPU for embedding builds (text encoder, about 8GB peak). Default: same as --device, which blocks inference while building",
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--steps", type=parse_positive_int, default=4)
    parser.add_argument(
        "--min-steps-non-distilled",
        type=parse_positive_int,
        default=20,
        help=(
            "Below this num_inference_steps, warn when the loaded model is not step-distilled "
            "(e.g. klein-base-4B at 4 steps produces blurry/incoherent output; use klein-4B for low steps)"
        ),
    )
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--dimension-multiple", type=parse_positive_int, default=16)
    parser.add_argument("--max-generation-area", type=parse_non_negative_int, default=1024 * 1024)
    parser.add_argument("--max-images", type=parse_positive_int, default=5)
    parser.add_argument("--max-request-megabytes", type=parse_positive_int, default=64)
    parser.add_argument("--build-batch-size", type=parse_positive_int, default=1)
    parser.add_argument("--build-timeout", type=parse_positive_int, default=7200)
    parser.add_argument("--job-log-characters", type=parse_positive_int, default=20000)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument(
        "--fp8-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store transformer weights in fp8 and upcast per layer (about 3.8GB less VRAM, about 3%% slower)",
    )
    parser.add_argument(
        "--mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Restrict edits to the masked face regions (per-request parameters.mask overrides)",
    )
    parser.add_argument(
        "--mask-mode",
        choices=("inpaint", "composite", "mice"),
        default="inpaint",
        help="Default mask.mode: true latent-level inpaint (recommended, see final_report.md), "
        "legacy full-regenerate-then-RGB-composite, or the attention-bias MICE path",
    )
    parser.add_argument("--mask-regions", default="mouth", help=f"Comma-separated, from {', '.join(MASK_REGIONS)}")
    parser.add_argument("--mask-dilate", type=parse_non_negative_int, default=8, help="Mask dilation in pixels")
    parser.add_argument("--mask-feather", type=float, default=12.0, help="Mask edge blur radius in pixels")
    parser.add_argument("--face-landmarker-model", type=Path, default=ROOT / "weights" / "face_landmarker.task")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    try:
        parse_regions(args.mask_regions)
    except ValueError as error:
        parser.error(str(error))
    return args


def main() -> None:
    args = parse_args()
    app = create_app(args)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
