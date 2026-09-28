import argparse
import gc
import json
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, broadcast_object_list, set_seed
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from diffusers import FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline, Flux2Transformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

try:
    from .publicdataset import CommonCatalogDataset
except ImportError:
    from publicdataset import CommonCatalogDataset


DEFAULT_TEACHER = "black-forest-labs/FLUX.2-klein-base-4B"
DEFAULT_STUDENT_CONFIG = Path(__file__).with_name("student_configure.yaml")
CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)$")
STUDENT_CONFIG_FIELDS = {
    "architecture": {
        "num_layers": ("student_num_layers", 2),
        "num_single_layers": ("student_num_single_layers", 3),
        "hidden_size": ("student_hidden_size", None),
        "double_layer_mapping": ("student_double_layer_mapping", None),
        "single_layer_mapping": ("student_single_layer_mapping", None),
        "mapping_strategy": ("mapping_strategy", "manual"),
        "mapping_calibration_batches": ("mapping_calibration_batches", 2),
        "random_init": ("random_init", False),
        "teacher_clone_check": ("teacher_clone_check", False),
        "teacher_clone_tolerance": ("teacher_clone_tolerance", 1e-3),
    },
    "distillation": {
        "lambda_gt": ("lambda_gt", 1.0),
        "lambda_kd": ("lambda_kd", 1.0),
        "lambda_flow": ("lambda_flow", None),
        "lambda_hidden": ("lambda_hidden", 0.0),
        "lambda_direction": ("lambda_direction", 0.05),
        "lambda_trajectory": ("lambda_trajectory", 1.0),
        "trajectory_probability": ("trajectory_probability", 0.2),
        "trajectory_rollout_steps": ("trajectory_rollout_steps", 2),
        "trajectory_num_inference_steps": ("trajectory_num_inference_steps", 50),
        "trajectory_warmup_steps": ("trajectory_warmup_steps", 1000),
        "hidden_double_layers": ("hidden_double_layers", None),
        "hidden_single_layers": ("hidden_single_layers", None),
        "unconditional_probability": ("unconditional_probability", 0.0),
        "weighting_scheme": ("weighting_scheme", "none"),
        "logit_mean": ("logit_mean", 0.0),
        "logit_std": ("logit_std", 1.0),
    },
    "optimization": {
        "learning_rate": ("learning_rate", 1e-4),
        "lr_scheduler": ("lr_scheduler", "constant_with_warmup"),
        "lr_warmup_steps": ("lr_warmup_steps", 500),
    },
    "data": {
        "max_train_samples": ("max_train_samples", None),
        "editing_mode": ("editing_mode", "single"),
        "editing_max_reference_images": ("editing_max_reference_images", 1),
    },
    "validation": {
        "num_inference_steps": ("validation_num_inference_steps", 50),
        "guidance_scale": ("validation_guidance_scale", 4.0),
        "height": ("validation_height", None),
        "width": ("validation_width", None),
        "num_images": ("num_validation_images", 4),
        "prompt_file": ("validation_prompt_file", None),
        "encoding_batch_size": ("validation_encoding_batch_size", 1),
        "editing_dataset_root": ("validation_editing_dataset_root", None),
    },
}


def load_student_config(path):
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Student config not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Student config must be a YAML mapping: {path}")
    if config.get("version") != 1:
        raise ValueError(f"Unsupported student config version in {path}: {config.get('version')!r}")

    unknown_sections = set(config) - {"version", *STUDENT_CONFIG_FIELDS}
    if unknown_sections:
        raise ValueError(f"Unknown student config sections: {sorted(unknown_sections)}")
    for section, fields in STUDENT_CONFIG_FIELDS.items():
        values = config.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"Student config section {section!r} must be a mapping")
        unknown_fields = set(values) - set(fields)
        if unknown_fields:
            raise ValueError(f"Unknown fields in student config section {section!r}: {sorted(unknown_fields)}")
    return path.resolve(), config


def apply_student_config(args):
    path, config = load_student_config(args.student_config)
    for section, fields in STUDENT_CONFIG_FIELDS.items():
        values = config.get(section, {})
        for key, (attribute, fallback) in fields.items():
            if getattr(args, attribute) is None:
                setattr(args, attribute, values.get(key, fallback))
    args.student_config = str(path)
    return args


def resolved_student_config(args):
    config = {"version": 1}
    for section, fields in STUDENT_CONFIG_FIELDS.items():
        config[section] = {key: getattr(args, attribute) for key, (attribute, _) in fields.items()}
    return config


class CachedFlux2Dataset(Dataset):
    def __init__(self, cache_dir, max_samples=None):
        self.cache_dir = Path(cache_dir)
        self.samples = []
        for sample_dir in sorted(path for path in self.cache_dir.iterdir() if path.is_dir()):
            latent_path = self._first_existing(sample_dir, ("latent.pt", "image_latent.pt"))
            embedding_path = self._first_existing(sample_dir, ("embedding.pt", "text_embedding.pt"))
            if latent_path is not None and embedding_path is not None:
                self.samples.append((sample_dir, latent_path, embedding_path))
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise ValueError(
                f"No cached samples found in {self.cache_dir}. Each sample directory must contain "
                "latent.pt and embedding.pt."
            )

    @staticmethod
    def _first_existing(directory, names):
        for name in names:
            path = directory / name
            if path.is_file():
                return path
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_dir, latent_path, embedding_path = self.samples[index]
        latent = torch.load(latent_path, map_location="cpu", weights_only=True)
        embedding = torch.load(embedding_path, map_location="cpu", weights_only=True)
        if latent.ndim == 4 and latent.shape[0] == 1:
            latent = latent[0]
        if embedding.ndim == 3 and embedding.shape[0] == 1:
            embedding = embedding[0]
        if latent.ndim != 3:
            raise ValueError(f"{latent_path} must have shape (C,H,W), got {tuple(latent.shape)}")
        if embedding.ndim != 2:
            raise ValueError(f"{embedding_path} must have shape (L,D), got {tuple(embedding.shape)}")
        prompt_path = sample_dir / "prompt.txt"
        return {
            "latent": latent,
            "embedding": embedding,
            "prompt": prompt_path.read_text(encoding="utf-8") if prompt_path.is_file() else "",
        }


class MagicBrushDataset(Dataset):
    def __init__(self, root_dir, mode="single", max_reference_images=1, max_samples=None):
        self.root_dir = Path(root_dir).expanduser()
        metadata_path = self.root_dir / "metadata.jsonl"
        if not metadata_path.is_file():
            raise ValueError(f"MagicBrush metadata not found: {metadata_path}")
        if mode not in ("single", "multi_reference"):
            raise ValueError(f"Unsupported MagicBrush editing mode: {mode}")
        if max_reference_images < 1:
            raise ValueError("editing_max_reference_images must be at least 1")
        if mode == "single" and max_reference_images != 1:
            raise ValueError("MagicBrush single mode requires editing_max_reference_images=1")
        if mode == "multi_reference" and max_reference_images < 2:
            raise ValueError("MagicBrush multi_reference mode requires at least 2 reference images")

        records = []
        for line_number, line in enumerate(metadata_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {metadata_path}:{line_number}: {error}") from error
            missing = {
                key
                for key in ("img_id", "turn_index", "instruction", "source_img", "target_img")
                if record.get(key) is None
            }
            if missing:
                raise ValueError(f"Missing MagicBrush fields at {metadata_path}:{line_number}: {sorted(missing)}")
            record = dict(record)
            record["turn_index"] = int(record["turn_index"])
            record["source_path"] = self.root_dir / record["source_img"]
            record["target_path"] = self.root_dir / record["target_img"]
            if not record["source_path"].is_file() or not record["target_path"].is_file():
                raise ValueError(
                    f"MagicBrush image missing at {metadata_path}:{line_number}: "
                    f"source={record['source_path']}, target={record['target_path']}"
                )
            records.append(record)

        sessions = {}
        for record in records:
            sessions.setdefault(str(record["img_id"]), []).append(record)
        for session in sessions.values():
            session.sort(key=lambda item: item["turn_index"])

        self.samples = []
        for session_id in sorted(sessions):
            session = sessions[session_id]
            for index, record in enumerate(session):
                if mode == "single":
                    reference_paths = [record["source_path"]]
                else:
                    history = [item["source_path"] for item in session[: index + 1]]
                    if len(history) < max_reference_images:
                        continue
                    if max_reference_images == 2:
                        reference_paths = [history[0], history[-1]]
                    else:
                        reference_paths = [history[0], *history[-(max_reference_images - 1) :]]
                self.samples.append(
                    {
                        "session_id": session_id,
                        "turn_index": record["turn_index"],
                        "instruction": str(record["instruction"]).strip(),
                        "reference_paths": reference_paths,
                        "target_path": record["target_path"],
                    }
                )

        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise ValueError(
                f"No usable MagicBrush samples found in {self.root_dir} for mode={mode!r}, "
                f"max_reference_images={max_reference_images}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample["target_path"]) as image:
            target_image = image.convert("RGB")
        reference_images = []
        for path in sample["reference_paths"]:
            with Image.open(path) as image:
                reference_images.append(image.convert("RGB"))
        return {
            "target_image": target_image,
            "reference_images": reference_images,
            "instruction": sample["instruction"],
            "session_id": sample["session_id"],
            "turn_index": sample["turn_index"],
            "reference_files": [str(path) for path in sample["reference_paths"]],
            "target_file": str(sample["target_path"]),
        }


def collate_cached_samples(examples):
    latent_shapes = {tuple(example["latent"].shape) for example in examples}
    embedding_shapes = {tuple(example["embedding"].shape) for example in examples}
    if len(latent_shapes) != 1:
        raise ValueError(f"A batch contains different latent shapes: {sorted(latent_shapes)}")
    if len(embedding_shapes) != 1:
        raise ValueError(f"A batch contains different embedding shapes: {sorted(embedding_shapes)}")
    return {
        "latents": torch.stack([example["latent"] for example in examples]),
        "embeddings": torch.stack([example["embedding"] for example in examples]),
        "prompts": [example["prompt"] for example in examples],
    }


def collate_public_samples(examples):
    return {
        "images": [example["image"] for example in examples],
        "captions": [example["caption"] for example in examples],
        "source_files": [example["source_file"] for example in examples],
    }


def collate_magicbrush_samples(examples):
    reference_counts = {len(example["reference_images"]) for example in examples}
    if len(reference_counts) != 1:
        raise ValueError(f"A MagicBrush batch contains different reference counts: {sorted(reference_counts)}")
    reference_count = reference_counts.pop()
    return {
        "target_images": [example["target_image"] for example in examples],
        "reference_images": [
            [example["reference_images"][reference_index] for example in examples]
            for reference_index in range(reference_count)
        ],
        "captions": [example["instruction"] for example in examples],
        "session_ids": [example["session_id"] for example in examples],
        "turn_indices": [example["turn_index"] for example in examples],
        "reference_files": [example["reference_files"] for example in examples],
        "target_files": [example["target_file"] for example in examples],
    }


def preprocess_public_images(images, resolution):
    tensors = []
    for image in images:
        image = image.convert("RGB")
        scale = resolution / min(image.size)
        resized = (
            max(resolution, round(image.width * scale)),
            max(resolution, round(image.height * scale)),
        )
        image = image.resize(resized, resample=Image.Resampling.LANCZOS)
        left = (image.width - resolution) // 2
        top = (image.height - resolution) // 2
        image = image.crop((left, top, left + resolution, top + resolution))
        array = np.asarray(image, dtype=np.float32).copy()
        tensors.append(torch.from_numpy(array).permute(2, 0, 1) / 127.5 - 1.0)
    return torch.stack(tensors)


@torch.no_grad()
def encode_images_to_latents(pipeline, images, resolution, device, dtype):
    pixel_values = preprocess_public_images(images, resolution).to(device=device, dtype=dtype)
    latents = pipeline.vae.encode(pixel_values).latent_dist.mode()
    latents = Flux2KleinPipeline._patchify_latents(latents)
    mean = pipeline.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    variance = pipeline.vae.bn.running_var.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    std = torch.sqrt(variance + pipeline.vae.bn.eps)
    return ((latents - mean) / std).to(dtype=dtype)


def load_public_encoding_pipeline(args, dtype, device):
    pipeline = Flux2KleinPipeline.from_pretrained(
        args.teacher_model,
        transformer=None,
        scheduler=None,
        dtype=dtype,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    pipeline.vae.eval().requires_grad_(False).to(device=device)
    pipeline.text_encoder.eval().requires_grad_(False).to(device=device)
    return pipeline


@torch.no_grad()
def encode_public_batch(pipeline, batch, resolution, device, dtype):
    latents = encode_images_to_latents(pipeline, batch["images"], resolution, device, dtype)
    prompt_embeds, _ = pipeline.encode_prompt(prompt=batch["captions"], device=device)
    return latents.to(dtype=dtype), prompt_embeds.to(dtype=dtype)


@torch.no_grad()
def encode_magicbrush_batch(pipeline, batch, resolution, device, dtype):
    target_latents = encode_images_to_latents(
        pipeline, batch["target_images"], resolution, device, dtype
    )
    reference_latents = [
        encode_images_to_latents(pipeline, images, resolution, device, dtype)
        for images in batch["reference_images"]
    ]
    prompt_embeds, _ = pipeline.encode_prompt(prompt=batch["captions"], device=device)
    return target_latents, reference_latents, prompt_embeds.to(dtype=dtype)


@torch.no_grad()
def load_public_validation_samples(args, pipeline, dtype, device):
    samples = []
    if args.validation_prompt_file:
        prompt_path = Path(args.validation_prompt_file).expanduser()
        captions = [
            line.strip()
            for line in prompt_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ][: args.num_validation_images]
        source = prompt_path
    else:
        root_dir = args.validation_public_dataset_root or args.public_dataset_root
        dataset = CommonCatalogDataset(
            root_dir=root_dir,
            image_key=args.public_image_key,
            caption_key=args.public_caption_key,
            fallback_caption_key=args.public_fallback_caption_key,
            batch_size=args.public_parquet_batch_size,
            skip_broken_files=True,
        )
        captions = []
        for sample in dataset:
            captions.append(sample["caption"])
            if len(captions) == args.num_validation_images:
                break
        source = root_dir
    if not captions:
        raise ValueError(f"No validation prompts found in {source}")
    for start in range(0, len(captions), args.validation_encoding_batch_size):
        caption_batch = captions[start : start + args.validation_encoding_batch_size]
        embeddings, _ = pipeline.encode_prompt(prompt=caption_batch, device=device)
        embeddings = embeddings.to(device="cpu", dtype=dtype)
        for caption, embedding in zip(caption_batch, embeddings):
            samples.append({"embedding": embedding, "prompt": caption})
        del embeddings
    return samples


@torch.no_grad()
def load_editing_validation_samples(args, pipeline, dtype, device):
    root_dir = args.validation_editing_dataset_root or args.editing_dataset_root
    dataset = MagicBrushDataset(
        root_dir=root_dir,
        mode=args.editing_mode,
        max_reference_images=args.editing_max_reference_images,
        max_samples=args.num_validation_images,
    )
    examples = [dataset[index] for index in range(len(dataset))]
    samples = []
    for start in range(0, len(examples), args.validation_encoding_batch_size):
        example_batch = examples[start : start + args.validation_encoding_batch_size]
        captions = [example["instruction"] for example in example_batch]
        embeddings, _ = pipeline.encode_prompt(prompt=captions, device=device)
        embeddings = embeddings.to(device="cpu", dtype=dtype)
        for example, embedding in zip(example_batch, embeddings):
            samples.append(
                {
                    "embedding": embedding,
                    "prompt": example["instruction"],
                    "reference_images": example["reference_images"],
                    "reference_files": example["reference_files"],
                    "target_image": example["target_image"],
                    "target_file": example["target_file"],
                    "session_id": example["session_id"],
                    "turn_index": example["turn_index"],
                }
            )
        del embeddings
    return samples


def evenly_spaced_indices(teacher_count, student_count):
    if student_count < 1 or student_count > teacher_count:
        raise ValueError(f"student_count must be between 1 and {teacher_count}, got {student_count}")
    if student_count == 1:
        return [0]
    return [round(index * (teacher_count - 1) / (student_count - 1)) for index in range(student_count)]


def reset_parameters(model):
    for module in model.modules():
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()


def expand_segment_indices(indices, source_segment_size, num_segments):
    return [index + segment * source_segment_size for segment in range(num_segments) for index in indices]


def build_width_indices(teacher, student):
    teacher_hidden = teacher.inner_dim
    student_hidden = student.inner_dim
    if student_hidden > teacher_hidden:
        raise ValueError(
            f"Student hidden size cannot exceed teacher hidden size: {student_hidden} > {teacher_hidden}"
        )
    if student.config.attention_head_dim != teacher.config.attention_head_dim:
        raise ValueError("Teacher and student must use the same attention_head_dim for structured initialization")

    head_dim = teacher.config.attention_head_dim
    teacher_heads = teacher.config.num_attention_heads
    student_heads = student.config.num_attention_heads
    selected_heads = evenly_spaced_indices(teacher_heads, student_heads)
    hidden = [head * head_dim + offset for head in selected_heads for offset in range(head_dim)]

    teacher_mlp = int(teacher_hidden * teacher.config.mlp_ratio)
    student_mlp = int(student_hidden * student.config.mlp_ratio)
    mlp = evenly_spaced_indices(teacher_mlp, student_mlp)
    return {
        "hidden": hidden,
        "mlp": mlp,
        "double_mod": expand_segment_indices(hidden, teacher_hidden, 6),
        "single_mod": expand_segment_indices(hidden, teacher_hidden, 3),
        "norm": expand_segment_indices(hidden, teacher_hidden, 2),
        "ff_in": mlp + [index + teacher_mlp for index in mlp],
        "single_qkv_mlp": (
            expand_segment_indices(hidden, teacher_hidden, 3)
            + [index + 3 * teacher_hidden for index in mlp]
            + [index + 3 * teacher_hidden + teacher_mlp for index in mlp]
        ),
        "single_out_input": hidden + [index + teacher_hidden for index in mlp],
    }


def slice_teacher_parameter(name, source, target_shape, width_indices):
    hidden = width_indices["hidden"]
    mlp = width_indices["mlp"]
    if "double_stream_modulation" in name:
        output_indices, input_indices = width_indices["double_mod"], hidden
    elif "single_stream_modulation" in name:
        output_indices, input_indices = width_indices["single_mod"], hidden
    elif ".ff.linear_in." in name or ".ff_context.linear_in." in name:
        output_indices, input_indices = width_indices["ff_in"], hidden
    elif ".ff.linear_out." in name or ".ff_context.linear_out." in name:
        output_indices, input_indices = hidden, mlp
    elif "single_transformer_blocks" in name and ".attn.to_qkv_mlp_proj." in name:
        output_indices, input_indices = width_indices["single_qkv_mlp"], hidden
    elif "single_transformer_blocks" in name and ".attn.to_out." in name:
        output_indices, input_indices = hidden, width_indices["single_out_input"]
    elif "norm_out.linear." in name:
        output_indices, input_indices = width_indices["norm"], hidden
    else:
        output_indices = hidden if source.shape[0] != target_shape[0] else None
        input_indices = hidden if source.ndim == 2 and source.shape[1] != target_shape[1] else None

    result = source
    if output_indices is not None:
        result = result.index_select(0, torch.tensor(output_indices, device=source.device))
    if input_indices is not None:
        result = result.index_select(1, torch.tensor(input_indices, device=source.device))
    if tuple(result.shape) != tuple(target_shape):
        raise ValueError(
            f"Unsupported structured initialization for {name}: {tuple(source.shape)} -> "
            f"{tuple(target_shape)}, sliced={tuple(result.shape)}"
        )
    return result


def teacher_parameter_name(student_name, double_layer_mapping, single_layer_mapping):
    double_match = re.match(r"transformer_blocks\.(\d+)\.(.+)", student_name)
    if double_match:
        student_index = int(double_match.group(1))
        return f"transformer_blocks.{double_layer_mapping[student_index]}.{double_match.group(2)}"
    single_match = re.match(r"single_transformer_blocks\.(\d+)\.(.+)", student_name)
    if single_match:
        student_index = int(single_match.group(1))
        return f"single_transformer_blocks.{single_layer_mapping[student_index]}.{single_match.group(2)}"
    return student_name


def resolve_layer_mapping(mapping, teacher_count, student_count, name):
    if mapping is None:
        return evenly_spaced_indices(teacher_count, student_count)
    if len(mapping) != student_count:
        raise ValueError(f"{name} mapping must contain {student_count} indices, got {len(mapping)}")
    if mapping != sorted(set(mapping)):
        raise ValueError(f"{name} mapping must contain unique indices in ascending order: {mapping}")
    if mapping[0] < 0 or mapping[-1] >= teacher_count:
        raise ValueError(f"{name} mapping indices must be between 0 and {teacher_count - 1}: {mapping}")
    return mapping


def initialize_student_from_teacher(student, teacher, double_layer_mapping=None, single_layer_mapping=None):
    double_indices = resolve_layer_mapping(
        double_layer_mapping,
        len(teacher.transformer_blocks),
        len(student.transformer_blocks),
        "double-stream",
    )
    single_indices = resolve_layer_mapping(
        single_layer_mapping,
        len(teacher.single_transformer_blocks),
        len(student.single_transformer_blocks),
        "single-stream",
    )
    teacher_parameters = dict(teacher.named_parameters())
    width_indices = build_width_indices(teacher, student)
    with torch.no_grad():
        for student_name, student_parameter in student.named_parameters():
            teacher_name = teacher_parameter_name(student_name, double_indices, single_indices)
            teacher_parameter = teacher_parameters[teacher_name]
            if teacher_parameter.shape == student_parameter.shape:
                initialized = teacher_parameter
            else:
                initialized = slice_teacher_parameter(
                    student_name,
                    teacher_parameter,
                    student_parameter.shape,
                    width_indices,
                )
            student_parameter.copy_(initialized.to(student_parameter.dtype))
    return double_indices, single_indices


def build_student(
    teacher,
    num_layers,
    num_single_layers,
    hidden_size,
    random_init,
    device,
    double_layer_mapping=None,
    single_layer_mapping=None,
):
    config = dict(teacher.config)
    config["num_layers"] = num_layers
    config["num_single_layers"] = num_single_layers
    if hidden_size is not None:
        if hidden_size % teacher.config.attention_head_dim:
            raise ValueError(
                f"student hidden_size must be divisible by attention_head_dim "
                f"{teacher.config.attention_head_dim}, got {hidden_size}"
            )
        config["num_attention_heads"] = hidden_size // teacher.config.attention_head_dim
    with torch.device("meta"):
        student = Flux2Transformer2DModel.from_config(config)
    student.to_empty(device=device)
    reset_parameters(student)
    mapping = (
        resolve_layer_mapping(
            double_layer_mapping,
            len(teacher.transformer_blocks),
            len(student.transformer_blocks),
            "double-stream",
        ),
        resolve_layer_mapping(
            single_layer_mapping,
            len(teacher.single_transformer_blocks),
            len(student.single_transformer_blocks),
            "single-stream",
        ),
    )
    if not random_init:
        mapping = initialize_student_from_teacher(
            student,
            teacher,
            double_layer_mapping=double_layer_mapping,
            single_layer_mapping=single_layer_mapping,
        )
    return student, mapping


class HiddenProjectionModel(nn.Module):
    def __init__(self, student_hidden_size, teacher_hidden_size, double_layers, single_layers, teacher_hidden_indices):
        super().__init__()
        self.double_projections = nn.ModuleDict(
            {str(index): nn.Linear(student_hidden_size, teacher_hidden_size, bias=False) for index in double_layers}
        )
        self.single_projections = nn.ModuleDict(
            {str(index): nn.Linear(student_hidden_size, teacher_hidden_size, bias=False) for index in single_layers}
        )
        with torch.no_grad():
            for projection in [*self.double_projections.values(), *self.single_projections.values()]:
                projection.weight.zero_()
                projection.weight[teacher_hidden_indices, torch.arange(student_hidden_size)] = 1

    def forward(self, double_features, single_features):
        projected_double = {
            index: (
                self.double_projections[str(index)](text),
                self.double_projections[str(index)](image),
            )
            for index, (text, image) in double_features.items()
        }
        projected_single = {
            index: self.single_projections[str(index)](feature) for index, feature in single_features.items()
        }
        return projected_double, projected_single


@contextmanager
def capture_hidden_features(model, double_layers, single_layers, detach=False):
    unwrapped = model.module if hasattr(model, "module") else model
    double_features = {}
    single_features = {}

    def capture_double(index):
        def hook(module, inputs, output):
            text, image = output
            if detach:
                text, image = text.detach(), image.detach()
            double_features[index] = (text, image)

        return hook

    def capture_single(index):
        def hook(module, inputs, output):
            single_features[index] = output.detach() if detach else output

        return hook

    handles = [
        unwrapped.transformer_blocks[index].register_forward_hook(capture_double(index))
        for index in double_layers
    ]
    handles.extend(
        unwrapped.single_transformer_blocks[index].register_forward_hook(capture_single(index))
        for index in single_layers
    )
    try:
        yield double_features, single_features
    finally:
        for handle in handles:
            handle.remove()


def hidden_distillation_loss(projected_features, teacher_features, double_mapping, single_mapping):
    projected_double, projected_single = projected_features
    teacher_double, teacher_single = teacher_features
    losses = []
    for student_index, (student_text, student_image) in projected_double.items():
        teacher_text, teacher_image = teacher_double[double_mapping[student_index]]
        losses.append(relative_mse(student_text, teacher_text))
        losses.append(relative_mse(student_image, teacher_image))
    for student_index, student_feature in projected_single.items():
        teacher_feature = teacher_single[single_mapping[student_index]]
        losses.append(relative_mse(student_feature, teacher_feature))
    if not losses:
        raise ValueError("Hidden distillation requires at least one selected feature")
    return torch.stack(losses).mean()


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def resolve_latest_checkpoint(output_dir):
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if path.is_dir() and match:
            checkpoints.append((int(match.group(1)), path))
    return max(checkpoints, default=(None, None))[1]


def register_accelerator_hooks(accelerator):
    def save_model_hook(models, weights, output_dir):
        for model in models:
            unwrapped = accelerator.unwrap_model(model)
            if isinstance(unwrapped, Flux2Transformer2DModel):
                if accelerator.is_main_process:
                    state_dict = accelerator.get_state_dict(model)
                    unwrapped.save_pretrained(
                        Path(output_dir) / "transformer",
                        state_dict=state_dict,
                        safe_serialization=True,
                    )
            elif isinstance(unwrapped, HiddenProjectionModel):
                if accelerator.is_main_process:
                    torch.save(accelerator.get_state_dict(model), Path(output_dir) / "hidden_projections.pt")
            else:
                raise ValueError(f"Unexpected model in checkpoint: {type(unwrapped).__name__}")
            if weights:
                weights.pop()

    def load_model_hook(models, input_dir):
        while models:
            model = models.pop()
            unwrapped = accelerator.unwrap_model(model)
            if isinstance(unwrapped, Flux2Transformer2DModel):
                loaded = Flux2Transformer2DModel.from_pretrained(
                    input_dir,
                    subfolder="transformer",
                    dtype=torch.float32,
                    local_files_only=True,
                )
                unwrapped.register_to_config(**dict(loaded.config))
                unwrapped.load_state_dict(loaded.state_dict())
                del loaded
            elif isinstance(unwrapped, HiddenProjectionModel):
                projection_path = Path(input_dir) / "hidden_projections.pt"
                if not projection_path.is_file():
                    raise ValueError(f"Checkpoint hidden projections not found: {projection_path}")
                unwrapped.load_state_dict(torch.load(projection_path, map_location="cpu", weights_only=True))
            else:
                raise ValueError(f"Unexpected model in checkpoint: {type(unwrapped).__name__}")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)


def validate_cache_shapes(dataset, teacher):
    sample = dataset[0]
    latent = sample["latent"]
    embedding = sample["embedding"]
    if latent.shape[0] != teacher.config.in_channels:
        raise ValueError(
            f"Cached latent has {latent.shape[0]} channels, but the transformer expects "
            f"{teacher.config.in_channels}. Cache the normalized, patchified FLUX.2 VAE latent."
        )
    if embedding.shape[-1] != teacher.config.joint_attention_dim:
        raise ValueError(
            f"Cached embedding has dimension {embedding.shape[-1]}, but the transformer expects "
            f"{teacher.config.joint_attention_dim}."
        )


def get_sigmas(scheduler, indices, latent_ndim, device, dtype):
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)[indices]
    while sigmas.ndim < latent_ndim:
        sigmas = sigmas.unsqueeze(-1)
    return sigmas


def transformer_prediction(
    model,
    noisy_latents,
    prompt_embeds,
    timesteps,
    guidance_scale,
    reference_latents=None,
):
    packed_latents = Flux2KleinPipeline._pack_latents(noisy_latents)
    image_ids = Flux2KleinPipeline._prepare_latent_ids(noisy_latents).to(noisy_latents.device)
    model_latents = packed_latents
    if reference_latents:
        packed_references = [Flux2KleinPipeline._pack_latents(latent) for latent in reference_latents]
        reference_ids = Flux2KleinPipeline._prepare_image_ids(
            [latent[:1] for latent in reference_latents]
        ).to(noisy_latents.device)
        reference_ids = reference_ids.expand(noisy_latents.shape[0], -1, -1)
        model_latents = torch.cat([packed_latents, *packed_references], dim=1)
        image_ids = torch.cat([image_ids, reference_ids], dim=1)
    text_ids = Flux2KleinPipeline._prepare_text_ids(prompt_embeds).to(noisy_latents.device)
    guidance = None
    model_config = model.module.config if hasattr(model, "module") else model.config
    if model_config.guidance_embeds:
        guidance = torch.full(
            (noisy_latents.shape[0],), guidance_scale, device=noisy_latents.device, dtype=noisy_latents.dtype
        )
    prediction = model(
        hidden_states=model_latents,
        encoder_hidden_states=prompt_embeds,
        timestep=timesteps / 1000,
        img_ids=image_ids,
        txt_ids=text_ids,
        guidance=guidance,
        return_dict=False,
    )[0]
    return prediction[:, : packed_latents.shape[1]]


def weighted_mse(prediction, target, weights):
    per_sample = (prediction.float() - target.float()).pow(2).flatten(1).mean(1)
    return (per_sample * weights.float().flatten()).mean()


def weighted_direction_loss(prediction, target, weights):
    per_token = 1 - F.cosine_similarity(prediction.float(), target.float(), dim=-1, eps=1e-8)
    per_sample = per_token.mean(1)
    return (per_sample * weights.float().flatten()).mean()


def unpack_latent_prediction(prediction, latent_shape):
    batch_size, num_channels, height, width = latent_shape
    expected_shape = (batch_size, height * width, num_channels)
    if prediction.shape != expected_shape:
        raise ValueError(
            f"Packed prediction has shape {tuple(prediction.shape)}, expected {expected_shape}"
        )
    return prediction.permute(0, 2, 1).reshape(latent_shape)


def prepare_trajectory_schedule(noise_scheduler, model_input, num_inference_steps):
    trajectory_scheduler = FlowMatchEulerDiscreteScheduler.from_config(noise_scheduler.config)
    image_seq_len = Flux2KleinPipeline._pack_latents(model_input[:1]).shape[1]
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    trajectory_scheduler.set_timesteps(
        num_inference_steps,
        device=model_input.device,
        sigmas=sigmas,
        mu=mu,
    )
    return (
        trajectory_scheduler.timesteps,
        trajectory_scheduler.sigmas.to(device=model_input.device, dtype=model_input.dtype),
    )


@torch.no_grad()
def rollout_student_trajectory(
    student,
    clean_latents,
    noise,
    prompt_embeds,
    timestep_density,
    trajectory_timesteps,
    trajectory_sigmas,
    rollout_steps,
    guidance_scale,
    reference_latents=None,
):
    num_inference_steps = len(trajectory_timesteps)
    max_start_index = num_inference_steps - rollout_steps - 1
    start_indices = (timestep_density * (max_start_index + 1)).long().clamp(max=max_start_index)
    start_sigmas = trajectory_sigmas[start_indices]
    while start_sigmas.ndim < clean_latents.ndim:
        start_sigmas = start_sigmas.unsqueeze(-1)
    trajectory_latents = (1 - start_sigmas) * clean_latents + start_sigmas * noise

    for offset in range(rollout_steps):
        current_indices = start_indices + offset
        current_timesteps = trajectory_timesteps[current_indices]
        prediction = transformer_prediction(
            student,
            trajectory_latents,
            prompt_embeds,
            current_timesteps,
            guidance_scale,
            reference_latents=reference_latents,
        )
        velocity = unpack_latent_prediction(prediction, trajectory_latents.shape)
        delta_sigmas = trajectory_sigmas[current_indices + 1] - trajectory_sigmas[current_indices]
        while delta_sigmas.ndim < trajectory_latents.ndim:
            delta_sigmas = delta_sigmas.unsqueeze(-1)
        trajectory_latents = trajectory_latents.float().add(
            delta_sigmas.float() * velocity.float()
        ).to(dtype=clean_latents.dtype)

    final_indices = start_indices + rollout_steps
    final_sigmas = trajectory_sigmas[final_indices]
    while final_sigmas.ndim < clean_latents.ndim:
        final_sigmas = final_sigmas.unsqueeze(-1)
    return trajectory_latents.detach(), trajectory_timesteps[final_indices], final_sigmas


def deterministic_probability_sample(seed, step):
    value = (int(seed) + int(step) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 31
    return value / 2**64


def relative_mse(value, reference):
    value = value.float()
    reference = reference.float()
    denominator = reference.square().mean().clamp_min(torch.finfo(torch.float32).eps)
    return (value - reference).square().mean() / denominator


def select_sensitive_indices(activation_scores, loss_sensitivity_scores, count):
    activation_order = sorted(
        range(len(activation_scores)), key=lambda index: (activation_scores[index], -index), reverse=True
    )
    sensitivity_order = sorted(
        range(len(loss_sensitivity_scores)),
        key=lambda index: (loss_sensitivity_scores[index], -index),
        reverse=True,
    )
    activation_ranks = {index: len(activation_order) - rank for rank, index in enumerate(activation_order)}
    sensitivity_ranks = {index: len(sensitivity_order) - rank for rank, index in enumerate(sensitivity_order)}
    combined_order = sorted(
        range(len(activation_scores)),
        key=lambda index: (
            activation_ranks[index] + sensitivity_ranks[index],
            loss_sensitivity_scores[index],
            activation_scores[index],
            -index,
        ),
        reverse=True,
    )
    return sorted(combined_order[:count])


@torch.no_grad()
def calibrate_layer_mapping(
    teacher,
    args,
    dataset,
    noise_scheduler,
    dtype,
    device,
    public_encoding_pipeline=None,
):
    if args.public_dataset_root:
        calibration_dataset = CommonCatalogDataset(
            root_dir=args.public_dataset_root,
            image_key=args.public_image_key,
            caption_key=args.public_caption_key,
            fallback_caption_key=args.public_fallback_caption_key,
            batch_size=args.public_parquet_batch_size,
            skip_broken_files=True,
        )
        calibration_loader = DataLoader(
            calibration_dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=collate_public_samples,
        )
    elif args.editing_dataset_root:
        calibration_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_magicbrush_samples,
        )
    else:
        calibration_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_cached_samples,
        )

    double_activation = [0.0] * len(teacher.transformer_blocks)
    single_activation = [0.0] * len(teacher.single_transformer_blocks)
    double_sensitivity = [0.0] * len(teacher.transformer_blocks)
    single_sensitivity = [0.0] * len(teacher.single_transformer_blocks)
    timestep_indices = []
    generator = torch.Generator(device=device).manual_seed(args.seed)
    completed_batches = 0

    for batch in calibration_loader:
        if completed_batches == args.mapping_calibration_batches:
            break
        reference_latents = None
        if args.public_dataset_root:
            model_input, prompt_embeds = encode_public_batch(
                public_encoding_pipeline,
                batch,
                args.public_resolution,
                device,
                dtype,
            )
        elif args.editing_dataset_root:
            model_input, reference_latents, prompt_embeds = encode_magicbrush_batch(
                public_encoding_pipeline,
                batch,
                args.public_resolution,
                device,
                dtype,
            )
        else:
            model_input = batch["latents"].to(device=device, dtype=dtype)
            prompt_embeds = batch["embeddings"].to(device=device, dtype=dtype)

        noise = torch.randn(model_input.shape, generator=generator, device=device, dtype=dtype)
        timestep_index = round(
            (completed_batches + 1)
            * (noise_scheduler.config.num_train_timesteps - 1)
            / (args.mapping_calibration_batches + 1)
        )
        indices = torch.full((model_input.shape[0],), timestep_index, device=device, dtype=torch.long)
        timesteps = noise_scheduler.timesteps.to(device)[indices]
        sigmas = get_sigmas(noise_scheduler, indices, model_input.ndim, device, model_input.dtype)
        noisy_model_input = (1 - sigmas) * model_input + sigmas * noise

        double_values = [None] * len(teacher.transformer_blocks)
        single_values = [None] * len(teacher.single_transformer_blocks)

        def capture_double(index):
            def hook(module, hook_args, hook_kwargs, output):
                image_score = relative_mse(output[1], hook_kwargs["hidden_states"])
                text_score = relative_mse(output[0], hook_kwargs["encoder_hidden_states"])
                double_values[index] = (image_score + text_score) / 2

            return hook

        def capture_single(index):
            def hook(module, hook_args, hook_kwargs, output):
                single_values[index] = relative_mse(output, hook_kwargs["hidden_states"])

            return hook

        handles = [
            block.register_forward_hook(capture_double(index), with_kwargs=True)
            for index, block in enumerate(teacher.transformer_blocks)
        ]
        handles.extend(
            block.register_forward_hook(capture_single(index), with_kwargs=True)
            for index, block in enumerate(teacher.single_transformer_blocks)
        )
        try:
            baseline_prediction = transformer_prediction(
                teacher,
                noisy_model_input,
                prompt_embeds,
                timesteps,
                args.guidance_scale,
                reference_latents=reference_latents,
            )
        finally:
            for handle in handles:
                handle.remove()

        for index, score in enumerate(double_values):
            double_activation[index] += score.item()
        for index, score in enumerate(single_values):
            single_activation[index] += score.item()

        def bypass_double(module, hook_args, hook_kwargs, output):
            return hook_kwargs["encoder_hidden_states"], hook_kwargs["hidden_states"]

        def bypass_single(module, hook_args, hook_kwargs, output):
            return hook_kwargs["hidden_states"]

        for index, block in enumerate(teacher.transformer_blocks):
            handle = block.register_forward_hook(bypass_double, with_kwargs=True)
            try:
                ablated_prediction = transformer_prediction(
                    teacher,
                    noisy_model_input,
                    prompt_embeds,
                    timesteps,
                    args.guidance_scale,
                    reference_latents=reference_latents,
                )
            finally:
                handle.remove()
            double_sensitivity[index] += relative_mse(ablated_prediction, baseline_prediction).item()

        for index, block in enumerate(teacher.single_transformer_blocks):
            handle = block.register_forward_hook(bypass_single, with_kwargs=True)
            try:
                ablated_prediction = transformer_prediction(
                    teacher,
                    noisy_model_input,
                    prompt_embeds,
                    timesteps,
                    args.guidance_scale,
                    reference_latents=reference_latents,
                )
            finally:
                handle.remove()
            single_sensitivity[index] += relative_mse(ablated_prediction, baseline_prediction).item()

        timestep_indices.append(timestep_index)
        completed_batches += 1
        print(f"Layer mapping calibration: {completed_batches}/{args.mapping_calibration_batches}")

    if completed_batches != args.mapping_calibration_batches:
        raise RuntimeError(
            f"Layer mapping calibration found {completed_batches} batches, "
            f"but {args.mapping_calibration_batches} were requested"
        )

    double_activation = [score / completed_batches for score in double_activation]
    single_activation = [score / completed_batches for score in single_activation]
    double_sensitivity = [score / completed_batches for score in double_sensitivity]
    single_sensitivity = [score / completed_batches for score in single_sensitivity]
    double_mapping = select_sensitive_indices(
        double_activation, double_sensitivity, args.student_num_layers
    )
    single_mapping = select_sensitive_indices(
        single_activation, single_sensitivity, args.student_num_single_layers
    )
    report = {
        "strategy": "sensitivity",
        "calibration_batches": completed_batches,
        "seed": args.seed,
        "timestep_indices": timestep_indices,
        "selected_mapping": {"double": double_mapping, "single": single_mapping},
        "double_stream": [
            {
                "layer": index,
                "activation_score": double_activation[index],
                "loss_sensitivity": double_sensitivity[index],
                "selected": index in double_mapping,
            }
            for index in range(len(double_activation))
        ],
        "single_stream": [
            {
                "layer": index,
                "activation_score": single_activation[index],
                "loss_sensitivity": single_sensitivity[index],
                "selected": index in single_mapping,
            }
            for index in range(len(single_activation))
        ],
    }
    return double_mapping, single_mapping, report


@torch.no_grad()
def load_empty_prompt_embedding(args, dtype, device, text_pipeline=None):
    owns_pipeline = text_pipeline is None
    if owns_pipeline:
        text_pipeline = Flux2KleinPipeline.from_pretrained(
            args.teacher_model,
            transformer=None,
            vae=None,
            scheduler=None,
            dtype=dtype,
            revision=args.revision,
            local_files_only=args.local_files_only,
        )
        text_pipeline.text_encoder.to(device=device)
    empty_prompt_embedding, _ = text_pipeline.encode_prompt(prompt="", device=device)
    empty_prompt_embedding = empty_prompt_embedding.to(device=device, dtype=dtype)
    if owns_pipeline:
        text_pipeline.text_encoder.to(device="cpu")
        del text_pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return empty_prompt_embedding


def apply_conditioning_dropout(prompt_embeds, empty_prompt_embedding, probability, generator=None):
    if probability <= 0:
        return prompt_embeds, torch.zeros((), device=prompt_embeds.device)
    if empty_prompt_embedding is None:
        raise ValueError("empty_prompt_embedding is required when conditioning dropout is enabled")
    if empty_prompt_embedding.ndim != prompt_embeds.ndim or empty_prompt_embedding.shape[0] != 1:
        raise ValueError(
            "Empty prompt embedding must have shape (1,L,D), got "
            f"{tuple(empty_prompt_embedding.shape)}"
        )
    if empty_prompt_embedding.shape[1:] != prompt_embeds.shape[1:]:
        raise ValueError(
            "Empty and conditional prompt embedding shapes do not match: "
            f"{tuple(empty_prompt_embedding.shape)} vs {tuple(prompt_embeds.shape)}"
        )
    mask = torch.rand(prompt_embeds.shape[0], device=prompt_embeds.device, generator=generator) < probability
    if mask.any():
        prompt_embeds = prompt_embeds.clone()
        prompt_embeds[mask] = empty_prompt_embedding.expand(prompt_embeds.shape[0], -1, -1)[mask]
    return prompt_embeds, mask.float().mean()


def validate_resume_student_architecture(resume_path, args):
    config_path = Path(resume_path) / "transformer" / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Checkpoint transformer config not found: {config_path}")
    checkpoint_config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "num_layers": args.student_num_layers,
        "num_single_layers": args.student_num_single_layers,
        "num_attention_heads": args.student_num_attention_heads,
    }
    actual = {key: checkpoint_config.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            "Cannot resume a checkpoint with a different student architecture: "
            f"checkpoint={actual}, requested={expected}. Start a new output directory without "
            "--resume_from_checkpoint when changing the student structure."
        )


def resolve_student_transformer_path(value):
    path = Path(value).expanduser()
    candidates = (path / "student", path / "transformer", path)
    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return candidate
    raise ValueError(
        f"Student transformer not found under {path}. Expected config.json in student/, transformer/, or the path."
    )


def load_student_initialization(student, value, args):
    model_path = resolve_student_transformer_path(value)
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    expected = {
        "num_layers": args.student_num_layers,
        "num_single_layers": args.student_num_single_layers,
        "num_attention_heads": args.student_num_attention_heads,
    }
    actual = {key: config.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"Student initialization architecture mismatch: weights={actual}, requested={expected}"
        )
    loaded = Flux2Transformer2DModel.from_pretrained(
        model_path,
        dtype=torch.float32,
        local_files_only=True,
    )
    student.load_state_dict(loaded.state_dict())
    del loaded
    return model_path


def load_validation_pipeline(args, transformer, dtype, device):
    pipeline = Flux2KleinPipeline.from_pretrained(
        args.teacher_model,
        transformer=transformer,
        text_encoder=None,
        tokenizer=None,
        dtype=dtype,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    pipeline.vae.to(device=device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


@torch.no_grad()
def load_validation_negative_prompt_embeds(args, dtype, device, text_pipeline=None):
    if args.validation_guidance_scale <= 1:
        return None
    if args.validation_negative_prompt_embeds:
        path = Path(args.validation_negative_prompt_embeds)
        embeds = torch.load(path, map_location="cpu", weights_only=True)
        if embeds.ndim == 2:
            embeds = embeds.unsqueeze(0)
        if embeds.ndim != 3:
            raise ValueError(f"{path} must have shape (L,D) or (1,L,D), got {tuple(embeds.shape)}")
        return embeds.to(dtype=dtype)

    return load_empty_prompt_embedding(args, dtype, device, text_pipeline=text_pipeline).to(device="cpu")


@torch.no_grad()
def generate_validation_images(
    accelerator,
    pipeline,
    transformer,
    samples,
    negative_prompt_embeds,
    args,
    dtype,
):
    if args.validation_guidance_scale > 1 and negative_prompt_embeds is None:
        raise ValueError("CFG validation requires an empty negative-prompt embedding")
    previous_transformer = pipeline.transformer
    pipeline.transformer = transformer
    was_training = transformer.training
    transformer.eval()
    images = []
    for index, sample in enumerate(samples):
        prompt_embeds = sample["embedding"].unsqueeze(0).to(accelerator.device, dtype=dtype)
        negative_embeds = None
        if negative_prompt_embeds is not None:
            negative_embeds = negative_prompt_embeds.to(accelerator.device, dtype=dtype)
        generator = torch.Generator(device=accelerator.device).manual_seed(args.validation_seed + index)
        pipeline_args = {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_embeds,
            "guidance_scale": args.validation_guidance_scale,
            "num_inference_steps": args.validation_num_inference_steps,
            "generator": generator,
            "output_type": "pil",
        }
        if args.validation_height is not None:
            pipeline_args["height"] = args.validation_height
            pipeline_args["width"] = args.validation_width
        if sample.get("reference_images"):
            pipeline_args["image"] = sample["reference_images"]
        with accelerator.autocast():
            image = pipeline(**pipeline_args).images[0]
        images.append(image)
    transformer.train(was_training)
    pipeline.transformer = previous_transformer
    return images


def log_validation_images(accelerator, tag, images, samples, args, step):
    tracker = next((tracker for tracker in accelerator.trackers if tracker.name == "tensorboard"), None)
    if tracker is not None:
        tracker.writer.add_images(
            tag,
            np.stack([np.asarray(image) for image in images]),
            global_step=step,
            dataformats="NHWC",
        )

    tag_name = tag.replace("/", "-")
    validation_dir = Path(args.output_dir) / "validation" / f"step-{step:08d}"
    validation_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    for index, (image, sample) in enumerate(zip(images, samples)):
        seed = args.validation_seed + index
        filename = f"{tag_name}-{index:03d}-seed-{seed}.png"
        image.save(validation_dir / filename)
        record = {
            "file": filename,
            "prompt": sample["prompt"],
            "seed": seed,
            "guidance_scale": args.validation_guidance_scale,
            "num_inference_steps": args.validation_num_inference_steps,
            "width": image.width,
            "height": image.height,
        }
        if sample.get("reference_images"):
            reference_files = []
            for reference_index, reference_image in enumerate(sample["reference_images"]):
                reference_name = f"reference-{index:03d}-{reference_index:02d}.png"
                reference_image.save(validation_dir / reference_name)
                reference_files.append(reference_name)
            target_name = f"target-{index:03d}.png"
            sample["target_image"].save(validation_dir / target_name)
            record.update(
                {
                    "reference_files": reference_files,
                    "source_reference_files": sample.get("reference_files", []),
                    "target_file": target_name,
                    "source_target_file": sample.get("target_file"),
                    "session_id": sample.get("session_id"),
                    "turn_index": sample.get("turn_index"),
                }
            )
        metadata.append(record)
    (validation_dir / f"{tag_name}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    accelerator.print(f"Saved {tag} validation images: {validation_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    data_source = parser.add_mutually_exclusive_group(required=True)
    data_source.add_argument("--cache_dir")
    data_source.add_argument("--public_dataset_root")
    data_source.add_argument("--editing_dataset_root")
    parser.add_argument("--validation_cache_dir")
    parser.add_argument("--validation_public_dataset_root")
    parser.add_argument("--validation_editing_dataset_root")
    parser.add_argument("--public_image_key", default="jpg")
    parser.add_argument("--public_caption_key", default="blip2_caption")
    parser.add_argument("--public_fallback_caption_key", default="caption")
    parser.add_argument("--public_parquet_batch_size", type=int, default=64)
    parser.add_argument("--public_resolution", type=int, default=512)
    parser.add_argument("--teacher_model", default=DEFAULT_TEACHER)
    parser.add_argument("--revision")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--student_config", type=Path, default=DEFAULT_STUDENT_CONFIG)
    parser.add_argument("--student_init")
    parser.add_argument("--student_num_layers", type=int)
    parser.add_argument("--student_num_single_layers", type=int)
    parser.add_argument("--student_hidden_size", type=int)
    parser.add_argument("--student_double_layer_mapping", type=int, nargs="+")
    parser.add_argument("--student_single_layer_mapping", type=int, nargs="+")
    parser.add_argument("--mapping_strategy", choices=("manual", "sensitivity"))
    parser.add_argument("--mapping_calibration_batches", type=int)
    parser.add_argument("--random_init", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--teacher_clone_check", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--teacher_clone_tolerance", type=float)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--editing_mode", choices=("single", "multi_reference"))
    parser.add_argument("--editing_max_reference_images", type=int)
    parser.add_argument("--max_train_steps", type=int, default=50000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--lr_scheduler")
    parser.add_argument("--lr_warmup_steps", type=int)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lambda_gt", type=float)
    parser.add_argument("--lambda_kd", type=float)
    parser.add_argument("--lambda_flow", type=float)
    parser.add_argument("--lambda_hidden", type=float)
    parser.add_argument("--lambda_direction", type=float)
    parser.add_argument("--lambda_trajectory", type=float)
    parser.add_argument("--trajectory_probability", type=float)
    parser.add_argument("--trajectory_rollout_steps", type=int)
    parser.add_argument("--trajectory_num_inference_steps", type=int)
    parser.add_argument("--trajectory_warmup_steps", type=int)
    parser.add_argument("--hidden_double_layers", type=int, nargs="+")
    parser.add_argument("--hidden_single_layers", type=int, nargs="+")
    parser.add_argument("--unconditional_probability", type=float)
    parser.add_argument(
        "--weighting_scheme",
        choices=("none", "logit_normal", "mode", "sigma_sqrt", "cosmap"),
        default=None,
    )
    parser.add_argument("--logit_mean", type=float)
    parser.add_argument("--logit_std", type=float)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--resume_from_checkpoint")
    parser.add_argument("--report_to", choices=("tensorboard", "none"), default="tensorboard")
    parser.add_argument("--logging_dir", default="logs")
    parser.add_argument("--validation_epochs", type=int, default=0)
    parser.add_argument("--validation_steps", type=int, default=0)
    parser.add_argument("--num_validation_images", type=int)
    parser.add_argument("--validation_num_inference_steps", type=int)
    parser.add_argument("--validation_guidance_scale", type=float)
    parser.add_argument("--validation_seed", type=int, default=0)
    parser.add_argument("--validation_height", type=int)
    parser.add_argument("--validation_width", type=int)
    parser.add_argument("--validation_prompt_file")
    parser.add_argument("--validation_encoding_batch_size", type=int)
    parser.add_argument("--validation_negative_prompt_embeds")
    model_source = parser.add_mutually_exclusive_group()
    model_source.add_argument("--local_files_only", dest="local_files_only", action="store_true")
    model_source.add_argument("--allow_download", dest="local_files_only", action="store_false")
    parser.set_defaults(local_files_only=True)
    parser.add_argument("--skip_final_save", action="store_true")
    args = parser.parse_args()
    try:
        args = apply_student_config(args)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    if args.lambda_flow is None:
        args.lambda_flow = args.lambda_kd
    if args.max_train_steps < 1:
        parser.error("--max_train_steps must be at least 1")
    if args.public_resolution < 16 or args.public_resolution % 16:
        parser.error("--public_resolution must be a positive multiple of 16")
    if args.public_dataset_root and args.validation_epochs > 0 and args.validation_steps == 0:
        parser.error("CommonCatalog streaming mode uses --validation_steps instead of --validation_epochs")
    if args.student_num_layers < 1 or args.student_num_single_layers < 1:
        parser.error("Student double-stream and single-stream layer counts must both be positive")
    if args.student_hidden_size is not None and args.student_hidden_size < 1:
        parser.error("--student_hidden_size must be positive")
    if args.mapping_strategy not in ("manual", "sensitivity"):
        parser.error("--mapping_strategy must be manual or sensitivity")
    if args.mapping_calibration_batches < 1:
        parser.error("--mapping_calibration_batches must be at least 1")
    if args.mapping_strategy == "sensitivity":
        if args.random_init:
            parser.error("Sensitivity mapping cannot be combined with --random_init")
        if args.student_double_layer_mapping is not None or args.student_single_layer_mapping is not None:
            parser.error(
                "Sensitivity mapping selects both layer mappings automatically; "
                "remove double_layer_mapping and single_layer_mapping from the student config"
            )
    if any(
        weight < 0
        for weight in (
            args.lambda_gt,
            args.lambda_flow,
            args.lambda_hidden,
            args.lambda_direction,
            args.lambda_trajectory,
        )
    ):
        parser.error("Loss weights must be non-negative")
    if (
        args.lambda_gt
        + args.lambda_flow
        + args.lambda_hidden
        + args.lambda_direction
        + args.lambda_trajectory * args.trajectory_probability
        == 0
    ):
        parser.error("At least one non-negative loss weight must be greater than zero")
    if not 0 <= args.trajectory_probability <= 1:
        parser.error("--trajectory_probability must be between 0 and 1")
    if args.trajectory_rollout_steps < 1:
        parser.error("--trajectory_rollout_steps must be at least 1")
    if args.trajectory_num_inference_steps < 2:
        parser.error("--trajectory_num_inference_steps must be at least 2")
    if args.trajectory_rollout_steps >= args.trajectory_num_inference_steps:
        parser.error("--trajectory_rollout_steps must be smaller than --trajectory_num_inference_steps")
    if args.trajectory_warmup_steps < 0:
        parser.error("--trajectory_warmup_steps must be non-negative")
    hidden_double_layers = args.hidden_double_layers or []
    hidden_single_layers = args.hidden_single_layers or []
    if hidden_double_layers != sorted(set(hidden_double_layers)) or any(
        index < 0 or index >= args.student_num_layers for index in hidden_double_layers
    ):
        parser.error("--hidden_double_layers must be unique ascending student layer indices")
    if hidden_single_layers != sorted(set(hidden_single_layers)) or any(
        index < 0 or index >= args.student_num_single_layers for index in hidden_single_layers
    ):
        parser.error("--hidden_single_layers must be unique ascending student layer indices")
    if args.lambda_hidden > 0 and not (hidden_double_layers or hidden_single_layers):
        parser.error("Positive --lambda_hidden requires hidden_double_layers or hidden_single_layers")
    if args.max_train_samples is not None and args.max_train_samples < 1:
        parser.error("--max_train_samples must be at least 1")
    if args.editing_max_reference_images < 1:
        parser.error("--editing_max_reference_images must be at least 1")
    if args.editing_mode == "single" and args.editing_max_reference_images != 1:
        parser.error("--editing_mode single requires --editing_max_reference_images 1")
    if args.editing_mode == "multi_reference" and args.editing_max_reference_images < 2:
        parser.error("--editing_mode multi_reference requires at least 2 reference images")
    if args.num_validation_images < 1:
        parser.error("--num_validation_images must be at least 1")
    if args.validation_encoding_batch_size < 1:
        parser.error("--validation_encoding_batch_size must be at least 1")
    if args.teacher_clone_tolerance <= 0:
        parser.error("--teacher_clone_tolerance must be positive")
    if args.validation_prompt_file and not Path(args.validation_prompt_file).expanduser().is_file():
        parser.error(f"Validation prompt file not found: {args.validation_prompt_file}")
    if args.validation_editing_dataset_root and not args.editing_dataset_root:
        parser.error("--validation_editing_dataset_root requires --editing_dataset_root")
    if args.student_init and args.resume_from_checkpoint:
        parser.error("--student_init and --resume_from_checkpoint cannot be used together")
    if not 0 <= args.unconditional_probability <= 1:
        parser.error("--unconditional_probability must be between 0 and 1")
    if args.learning_rate <= 0:
        parser.error("--learning_rate must be positive")
    if (args.validation_height is None) != (args.validation_width is None):
        parser.error("--validation_height and --validation_width must be specified together")
    if args.validation_height is not None and (
        args.validation_height < 16
        or args.validation_width < 16
        or args.validation_height % 16
        or args.validation_width % 16
    ):
        parser.error("Validation height and width must be positive multiples of 16")
    return args


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging_dir = output_dir / args.logging_dir
    project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
        project_config=project_config,
    )
    if args.max_train_samples is not None and args.max_train_samples % accelerator.num_processes:
        raise ValueError(
            "max_train_samples must be divisible by the number of training processes so every rank "
            "runs the same number of batches"
        )
    if accelerator.is_main_process:
        (output_dir / "student_configure.resolved.yaml").write_text(
            yaml.safe_dump(resolved_student_config(args), sort_keys=False),
            encoding="utf-8",
        )
    set_seed(args.seed, device_specific=True)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    is_public_dataset = args.public_dataset_root is not None
    is_editing_dataset = args.editing_dataset_root is not None
    if is_public_dataset:
        dataset = CommonCatalogDataset(
            root_dir=args.public_dataset_root,
            image_key=args.public_image_key,
            caption_key=args.public_caption_key,
            fallback_caption_key=args.public_fallback_caption_key,
            batch_size=args.public_parquet_batch_size,
            skip_broken_files=True,
            process_index=accelerator.process_index,
            num_processes=accelerator.num_processes,
            max_samples=args.max_train_samples,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=collate_public_samples,
            persistent_workers=args.dataloader_num_workers > 0,
        )
    elif is_editing_dataset:
        dataset = MagicBrushDataset(
            root_dir=args.editing_dataset_root,
            mode=args.editing_mode,
            max_reference_images=args.editing_max_reference_images,
            max_samples=args.max_train_samples,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_magicbrush_samples,
            persistent_workers=args.dataloader_num_workers > 0,
        )
    else:
        dataset = CachedFlux2Dataset(args.cache_dir, max_samples=args.max_train_samples)
        dataloader = DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_cached_samples,
            persistent_workers=args.dataloader_num_workers > 0,
        )
    if not is_public_dataset and len(dataloader) == 0:
        raise ValueError(
            f"The training dataloader is empty: dataset_size={len(dataset)}, "
            f"train_batch_size={args.train_batch_size}. Reduce --train_batch_size or add samples."
        )

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.teacher_model,
        subfolder="scheduler",
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    teacher = Flux2Transformer2DModel.from_pretrained(
        args.teacher_model,
        subfolder="transformer",
        revision=args.revision,
        dtype=weight_dtype,
        local_files_only=args.local_files_only,
    )
    teacher.to(accelerator.device)
    teacher.eval().requires_grad_(False)
    pipeline_config = Flux2KleinPipeline.load_config(
        args.teacher_model,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    if pipeline_config.get("is_distilled", False):
        raise ValueError(
            "Capacity distillation requires FLUX.2-klein-base-4B, not the step-distilled FLUX.2-klein-4B"
        )
    if teacher.config.guidance_embeds:
        raise ValueError("FLUX.2 Klein base teacher must have guidance_embeds=False")
    if args.student_hidden_size is None:
        args.student_hidden_size = teacher.inner_dim
    if args.student_hidden_size > teacher.inner_dim:
        raise ValueError(
            f"Student hidden size cannot exceed teacher hidden size: "
            f"{args.student_hidden_size} > {teacher.inner_dim}"
        )
    if args.student_hidden_size % teacher.config.attention_head_dim:
        raise ValueError(
            f"Student hidden size must be divisible by attention_head_dim "
            f"{teacher.config.attention_head_dim}: {args.student_hidden_size}"
        )
    args.student_num_attention_heads = args.student_hidden_size // teacher.config.attention_head_dim
    if not is_public_dataset and not is_editing_dataset:
        validate_cache_shapes(dataset, teacher)

    public_encoding_pipeline = None
    if args.mapping_strategy == "sensitivity":
        calibration_path = output_dir / "layer_mapping_calibration.json"
        mapping_payload = [None]
        if accelerator.is_main_process:
            if args.resume_from_checkpoint and calibration_path.is_file():
                calibration_report = json.loads(calibration_path.read_text(encoding="utf-8"))
                selected_mapping = calibration_report["selected_mapping"]
                double_mapping = selected_mapping["double"]
                single_mapping = selected_mapping["single"]
                print(f"Reusing calibrated layer mapping: double={double_mapping}, single={single_mapping}")
            else:
                if is_public_dataset or is_editing_dataset:
                    public_encoding_pipeline = load_public_encoding_pipeline(
                        args, weight_dtype, accelerator.device
                    )
                double_mapping, single_mapping, calibration_report = calibrate_layer_mapping(
                    teacher,
                    args,
                    dataset,
                    noise_scheduler,
                    weight_dtype,
                    accelerator.device,
                    public_encoding_pipeline=public_encoding_pipeline,
                )
                calibration_path.write_text(
                    json.dumps(calibration_report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(f"Calibrated layer mapping: double={double_mapping}, single={single_mapping}")
            mapping_payload[0] = (double_mapping, single_mapping)
        broadcast_object_list(mapping_payload)
        args.student_double_layer_mapping, args.student_single_layer_mapping = mapping_payload[0]
        if accelerator.is_main_process:
            (output_dir / "student_configure.resolved.yaml").write_text(
                yaml.safe_dump(resolved_student_config(args), sort_keys=False),
                encoding="utf-8",
            )

    student, mapping = build_student(
        teacher,
        args.student_num_layers,
        args.student_num_single_layers,
        args.student_hidden_size,
        args.random_init,
        accelerator.device,
        double_layer_mapping=args.student_double_layer_mapping,
        single_layer_mapping=args.student_single_layer_mapping,
    )
    student_init_path = None
    if args.student_init:
        student_init_path = load_student_initialization(student, args.student_init, args)
    if args.gradient_checkpointing:
        student.enable_gradient_checkpointing()

    if args.teacher_clone_check and (
        student.inner_dim != teacher.inner_dim
        or len(student.transformer_blocks) != len(teacher.transformer_blocks)
        or len(student.single_transformer_blocks) != len(teacher.single_transformer_blocks)
        or args.random_init
    ):
        raise ValueError("Teacher clone check requires the exact teacher width/depth and teacher initialization")
    if args.teacher_clone_check:
        student.to(dtype=weight_dtype)

    hidden_double_layers = args.hidden_double_layers or []
    hidden_single_layers = args.hidden_single_layers or []
    hidden_projections = None
    if args.lambda_hidden > 0:
        teacher_hidden_indices = build_width_indices(teacher, student)["hidden"]
        hidden_projections = HiddenProjectionModel(
            student.inner_dim,
            teacher.inner_dim,
            hidden_double_layers,
            hidden_single_layers,
            teacher_hidden_indices,
        ).to(accelerator.device)
    teacher_hidden_double_layers = [mapping[0][index] for index in hidden_double_layers]
    teacher_hidden_single_layers = [mapping[1][index] for index in hidden_single_layers]

    if accelerator.is_main_process:
        (output_dir / "student_configure.resolved.yaml").write_text(
            yaml.safe_dump(resolved_student_config(args), sort_keys=False),
            encoding="utf-8",
        )

    if (
        is_public_dataset
        or is_editing_dataset
        or args.validation_prompt_file
    ) and public_encoding_pipeline is None:
        public_encoding_pipeline = load_public_encoding_pipeline(
            args, weight_dtype, accelerator.device
        )

    empty_prompt_embedding = None
    if args.unconditional_probability > 0:
        empty_prompt_embedding = load_empty_prompt_embedding(
            args,
            weight_dtype,
            accelerator.device,
            text_pipeline=public_encoding_pipeline,
        )

    trainable_parameters = list(student.parameters())
    if hidden_projections is not None:
        trainable_parameters.extend(hidden_projections.parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    if hidden_projections is None:
        student, optimizer, lr_scheduler = accelerator.prepare(student, optimizer, lr_scheduler)
    else:
        student, hidden_projections, optimizer, lr_scheduler = accelerator.prepare(
            student, hidden_projections, optimizer, lr_scheduler
        )
    if not is_public_dataset:
        dataloader = accelerator.prepare(dataloader)
    register_accelerator_hooks(accelerator)

    if accelerator.is_main_process:
        tracker_config = {
            key: value
            if isinstance(value, (int, float, str, bool, torch.Tensor))
            else json.dumps(value, default=str)
            for key, value in vars(args).items()
        }
        accelerator.init_trackers("flux2-capacity-distillation", config=tracker_config)
        teacher_parameters = count_parameters(teacher)
        student_parameters = count_parameters(accelerator.unwrap_model(student))
        print(f"Teacher parameters: {teacher_parameters:,} ({teacher_parameters / 1e9:.3f}B)")
        print(f"Student parameters: {student_parameters:,} ({student_parameters / 1e9:.3f}B)")
        if hidden_projections is not None:
            projection_parameters = count_parameters(accelerator.unwrap_model(hidden_projections))
            print(f"Training-only hidden projection parameters: {projection_parameters:,}")
        if mapping is not None:
            print(f"Teacher layer mapping: double={mapping[0]}, single={mapping[1]}")
        if student_init_path is not None:
            print(f"Initialized student transformer from: {student_init_path}")
        if is_public_dataset:
            data_mode = f"CommonCatalog streaming: {args.public_dataset_root}"
        elif is_editing_dataset:
            data_mode = (
                f"MagicBrush {args.editing_mode}: {args.editing_dataset_root} "
                f"({len(dataset):,} usable samples, {args.editing_max_reference_images} references)"
            )
        else:
            data_mode = f"cache: {args.cache_dir}"
        print(f"Training data mode: {data_mode}")
        if args.max_train_samples is not None:
            print(f"Training sample subset: {args.max_train_samples:,} deterministic samples")
        global_batch_size = (
            args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        )
        print(f"Global batch size: {global_batch_size}")
        if args.lambda_trajectory > 0 and args.trajectory_probability > 0:
            print(
                "Trajectory KD: "
                f"probability={args.trajectory_probability}, rollout_steps={args.trajectory_rollout_steps}, "
                f"inference_steps={args.trajectory_num_inference_steps}, "
                f"warmup_steps={args.trajectory_warmup_steps}, weight={args.lambda_trajectory}"
            )

    global_step = 0
    resume_path = None
    set_seed(args.seed, device_specific=True)
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            resume_path = resolve_latest_checkpoint(output_dir)
        else:
            resume_path = Path(args.resume_from_checkpoint)
        if resume_path is None or not resume_path.is_dir():
            raise ValueError(f"Checkpoint not found: {args.resume_from_checkpoint}")
        validate_resume_student_architecture(resume_path, args)
        accelerator.load_state(resume_path)
        match = CHECKPOINT_PATTERN.fullmatch(resume_path.name)
        if match is None:
            raise ValueError(f"Checkpoint directory must be named checkpoint-<step>: {resume_path}")
        global_step = int(match.group(1))
        accelerator.print(f"Resumed complete training state from {resume_path} at step {global_step}")

    validation_samples = None
    validation_pipeline = None
    validation_negative_prompt_embeds = None
    validation_enabled = args.validation_epochs > 0 or args.validation_steps > 0
    if validation_enabled:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if is_editing_dataset:
                validation_samples = load_editing_validation_samples(
                    args, public_encoding_pipeline, weight_dtype, accelerator.device
                )
            elif args.validation_prompt_file:
                validation_samples = load_public_validation_samples(
                    args, public_encoding_pipeline, weight_dtype, accelerator.device
                )
            elif args.validation_cache_dir or not is_public_dataset:
                validation_dataset = CachedFlux2Dataset(args.validation_cache_dir or args.cache_dir)
                validation_samples = [
                    validation_dataset[index]
                    for index in range(min(args.num_validation_images, len(validation_dataset)))
                ]
            else:
                validation_samples = load_public_validation_samples(
                    args, public_encoding_pipeline, weight_dtype, accelerator.device
                )
            validation_pipeline = load_validation_pipeline(
                args, accelerator.unwrap_model(student), weight_dtype, accelerator.device
            )
            validation_negative_prompt_embeds = load_validation_negative_prompt_embeds(
                args,
                weight_dtype,
                accelerator.device,
                text_pipeline=public_encoding_pipeline,
            )
            teacher_images = generate_validation_images(
                accelerator,
                validation_pipeline,
                teacher,
                validation_samples,
                validation_negative_prompt_embeds,
                args,
                weight_dtype,
            )
            log_validation_images(
                accelerator, "validation/teacher", teacher_images, validation_samples, args, 0
            )
        accelerator.wait_for_everyone()

    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Steps",
    )
    clone_check_pending = args.teacher_clone_check
    trajectory_schedule_cache = {}
    training_micro_step = global_step * args.gradient_accumulation_steps

    epoch = 0
    while global_step < args.max_train_steps:
        student.train()
        batches_this_pass = 0
        for batch in dataloader:
            batches_this_pass += 1
            accumulated_models = (student,) if hidden_projections is None else (student, hidden_projections)
            with accelerator.accumulate(*accumulated_models):
                reference_latents = None
                if is_public_dataset:
                    model_input, prompt_embeds = encode_public_batch(
                        public_encoding_pipeline,
                        batch,
                        args.public_resolution,
                        accelerator.device,
                        weight_dtype,
                    )
                elif is_editing_dataset:
                    model_input, reference_latents, prompt_embeds = encode_magicbrush_batch(
                        public_encoding_pipeline,
                        batch,
                        args.public_resolution,
                        accelerator.device,
                        weight_dtype,
                    )
                else:
                    model_input = batch["latents"].to(
                        accelerator.device, dtype=weight_dtype, non_blocking=True
                    )
                    prompt_embeds = batch["embeddings"].to(
                        accelerator.device, dtype=weight_dtype, non_blocking=True
                    )
                prompt_embeds, unconditional_fraction = apply_conditioning_dropout(
                    prompt_embeds,
                    empty_prompt_embedding,
                    args.unconditional_probability,
                )
                noise = torch.randn_like(model_input)
                batch_size = model_input.shape[0]
                timestep_density = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=batch_size,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                    device=accelerator.device,
                )
                timestep_indices = (
                    timestep_density * noise_scheduler.config.num_train_timesteps
                ).long().clamp(max=noise_scheduler.config.num_train_timesteps - 1)
                timesteps = noise_scheduler.timesteps.to(accelerator.device)[timestep_indices]
                sigmas = get_sigmas(
                    noise_scheduler,
                    timestep_indices,
                    model_input.ndim,
                    accelerator.device,
                    model_input.dtype,
                )
                noisy_model_input = (1 - sigmas) * model_input + sigmas * noise
                packed_target = Flux2KleinPipeline._pack_latents(noise - model_input)
                trajectory_active = (
                    not clone_check_pending
                    and global_step >= args.trajectory_warmup_steps
                    and args.lambda_trajectory > 0
                    and deterministic_probability_sample(args.seed, training_micro_step)
                    < args.trajectory_probability
                )
                if trajectory_active:
                    schedule_key = (*model_input.shape[1:], args.trajectory_num_inference_steps)
                    if schedule_key not in trajectory_schedule_cache:
                        trajectory_schedule_cache[schedule_key] = prepare_trajectory_schedule(
                            noise_scheduler,
                            model_input,
                            args.trajectory_num_inference_steps,
                        )
                    trajectory_timesteps, trajectory_sigmas = trajectory_schedule_cache[schedule_key]
                    with accelerator.autocast():
                        noisy_model_input, timesteps, sigmas = rollout_student_trajectory(
                            student,
                            model_input,
                            noise,
                            prompt_embeds,
                            timestep_density,
                            trajectory_timesteps,
                            trajectory_sigmas,
                            args.trajectory_rollout_steps,
                            args.guidance_scale,
                            reference_latents=reference_latents,
                        )

                with capture_hidden_features(
                    teacher,
                    teacher_hidden_double_layers,
                    teacher_hidden_single_layers,
                    detach=True,
                ) as teacher_features:
                    with torch.no_grad(), accelerator.autocast():
                        teacher_prediction = transformer_prediction(
                            teacher,
                            noisy_model_input,
                            prompt_embeds,
                            timesteps,
                            args.guidance_scale,
                            reference_latents=reference_latents,
                        )

                with capture_hidden_features(
                    student,
                    hidden_double_layers,
                    hidden_single_layers,
                ) as student_features:
                    if clone_check_pending:
                        with torch.no_grad(), accelerator.autocast():
                            student_prediction = transformer_prediction(
                                student,
                                noisy_model_input,
                                prompt_embeds,
                                timesteps,
                                args.guidance_scale,
                                reference_latents=reference_latents,
                            )
                    else:
                        student_prediction = transformer_prediction(
                            student,
                            noisy_model_input,
                            prompt_embeds,
                            timesteps,
                            args.guidance_scale,
                            reference_latents=reference_latents,
                        )
                loss_weights = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme,
                    sigmas=sigmas,
                )
                loss_flow = weighted_mse(student_prediction, teacher_prediction, loss_weights)
                loss_direction = weighted_direction_loss(
                    student_prediction, teacher_prediction, loss_weights
                )
                zero_loss = torch.zeros((), device=student_prediction.device, dtype=torch.float32)
                if trajectory_active:
                    loss_gt = zero_loss
                    teacher_gt = zero_loss
                    loss_trajectory = loss_flow
                    flow_weight = args.lambda_trajectory
                else:
                    loss_gt = weighted_mse(student_prediction, packed_target, loss_weights)
                    teacher_gt = weighted_mse(teacher_prediction, packed_target, loss_weights)
                    loss_trajectory = zero_loss
                    flow_weight = args.lambda_flow
                loss_hidden = torch.zeros((), device=student_prediction.device, dtype=torch.float32)
                if hidden_projections is not None:
                    projected_features = hidden_projections(*student_features)
                    loss_hidden = hidden_distillation_loss(
                        projected_features,
                        teacher_features,
                        mapping[0],
                        mapping[1],
                    )
                loss = (
                    args.lambda_gt * loss_gt
                    + flow_weight * loss_flow
                    + args.lambda_hidden * loss_hidden
                    + args.lambda_direction * loss_direction
                )

                if clone_check_pending:
                    clone_max_diff = (student_prediction.float() - teacher_prediction.float()).abs().max()
                    if clone_max_diff.item() > args.teacher_clone_tolerance:
                        raise RuntimeError(
                            f"Teacher clone check failed: max_diff={clone_max_diff.item():.6g}, "
                            f"tolerance={args.teacher_clone_tolerance:.6g}"
                        )
                    accelerator.print(f"Teacher clone check passed: max_diff={clone_max_diff.item():.6g}")
                    accelerator.wait_for_everyone()
                    accelerator.end_training()
                    return

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss at step {global_step}: total={loss.item()}, "
                        f"gt={loss_gt.item()}, flow={loss_flow.item()}, hidden={loss_hidden.item()}, "
                        f"direction={loss_direction.item()}, trajectory={loss_trajectory.item()}"
                    )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            training_micro_step += 1

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                logs = {
                    "train/loss": loss.detach().item(),
                    "train/loss_gt": loss_gt.detach().item(),
                    "train/loss_kd": loss_flow.detach().item(),
                    "train/loss_flow": loss_flow.detach().item(),
                    "train/loss_hidden": loss_hidden.detach().item(),
                    "train/loss_direction": loss_direction.detach().item(),
                    "train/loss_trajectory": loss_trajectory.detach().item(),
                    "train/teacher_gt": teacher_gt.detach().item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/unconditional_fraction": unconditional_fraction.detach().item(),
                    "train/trajectory_fraction": float(trajectory_active),
                    "train/epoch": epoch + 1,
                }
                progress_bar.set_postfix(
                    loss=f"{logs['train/loss']:.4f}",
                    flow=f"{logs['train/loss_flow']:.4f}",
                    hidden=f"{logs['train/loss_hidden']:.4f}",
                    trajectory=int(trajectory_active),
                )
                accelerator.log(logs, step=global_step)

                if args.checkpointing_steps and global_step % args.checkpointing_steps == 0:
                    save_path = output_dir / f"checkpoint-{global_step}"
                    accelerator.save_state(save_path)
                    accelerator.print(f"Saved checkpoint: {save_path}")

                if args.validation_steps > 0 and global_step % args.validation_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        student_images = generate_validation_images(
                            accelerator,
                            validation_pipeline,
                            accelerator.unwrap_model(student),
                            validation_samples,
                            validation_negative_prompt_embeds,
                            args,
                            weight_dtype,
                        )
                        log_validation_images(
                            accelerator,
                            "validation/student",
                            student_images,
                            validation_samples,
                            args,
                            global_step,
                        )
                    accelerator.wait_for_everyone()

            if global_step >= args.max_train_steps:
                break

        if batches_this_pass == 0:
            raise RuntimeError("The training dataset produced no valid batches; check the dataset root and inputs")

        if not is_public_dataset and args.validation_epochs > 0 and (epoch + 1) % args.validation_epochs == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                student_images = generate_validation_images(
                    accelerator,
                    validation_pipeline,
                    accelerator.unwrap_model(student),
                    validation_samples,
                    validation_negative_prompt_embeds,
                    args,
                    weight_dtype,
                )
                log_validation_images(
                    accelerator,
                    "validation/student",
                    student_images,
                    validation_samples,
                    args,
                    global_step,
                )
            accelerator.wait_for_everyone()

        epoch += 1

    accelerator.wait_for_everyone()
    if accelerator.is_main_process and not args.skip_final_save:
        final_dir = output_dir / "student"
        state_dict = accelerator.get_state_dict(student)
        accelerator.unwrap_model(student).save_pretrained(
            final_dir,
            state_dict=state_dict,
            safe_serialization=True,
        )
        (output_dir / "training_args.json").write_text(
            json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Saved final Diffusers transformer: {final_dir}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
