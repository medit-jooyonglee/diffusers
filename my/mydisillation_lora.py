import argparse
import json
import math
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from diffusers import (
    DPMSolverMultistepScheduler,
    FlowMatchEulerDiscreteScheduler,
    Flux2KleinPipeline,
    Flux2Transformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import convert_unet_state_dict_to_peft

from mydisillation import (
    CHECKPOINT_PATTERN,
    DEFAULT_TEACHER,
    CachedFlux2Dataset,
    collate_cached_samples,
    collate_public_samples,
    encode_public_batch,
    generate_validation_images,
    get_sigmas,
    load_public_encoding_pipeline,
    load_validation_negative_prompt_embeds,
    load_validation_pipeline,
    log_validation_images,
    resolve_latest_checkpoint,
    transformer_prediction,
    validate_cache_shapes,
    weighted_mse,
)


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class ImageCaptionDataset(Dataset):
    def __init__(self, image_dirs):
        self.image_dirs = [Path(directory) for directory in image_dirs]
        if not self.image_dirs:
            raise ValueError("--image_dir must contain at least one directory")

        image_paths = {}
        for image_dir in self.image_dirs:
            if not image_dir.is_dir():
                raise ValueError(f"Image directory does not exist or is not a directory: {image_dir}")
            for image_path in image_dir.rglob("*"):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    image_paths.setdefault(image_path.resolve(), image_path)

        self.found_images = len(image_paths)
        self.missing_prompts = 0
        self.empty_prompts = 0
        self.unreadable_prompts = 0
        self.broken_images = 0
        self.samples = []
        for image_path in sorted(image_paths.values()):
            prompt_path = image_path.with_suffix(".txt")
            if not prompt_path.is_file():
                self.missing_prompts += 1
                continue
            try:
                prompt = prompt_path.read_text(encoding="utf-8-sig").strip()
            except (OSError, UnicodeError):
                self.unreadable_prompts += 1
                continue
            if not prompt:
                self.empty_prompts += 1
                continue
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception:
                self.broken_images += 1
                continue
            self.samples.append((image_path, prompt_path, prompt))

        if not self.samples:
            raise ValueError(
                "No valid image/text pairs found. Each supported image must have a non-empty "
                ".txt file with the same basename."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, _, prompt = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB").copy()
        return {"image": image, "caption": prompt, "source_file": str(image_path)}

    def prompt(self, index):
        return self.samples[index][2]


def encode_image_validation_samples(dataset, pipeline, count, device, dtype):
    prompts = [dataset.prompt(index) for index in range(min(count, len(dataset)))]
    embeddings, _ = pipeline.encode_prompt(prompt=prompts, device=device)
    return [
        {"embedding": embedding.to(device="cpu", dtype=dtype), "prompt": prompt}
        for prompt, embedding in zip(prompts, embeddings)
    ]


def configure_validation_scheduler(pipeline, scheduler_name):
    if scheduler_name == "flow_match_euler":
        return
    source_config = pipeline.scheduler.config
    pipeline.scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=source_config.num_train_timesteps,
        algorithm_type="dpmsolver++",
        solver_order=2,
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        flow_shift=getattr(source_config, "shift", 1.0),
        use_dynamic_shifting=True,
        time_shift_type=getattr(source_config, "time_shift_type", "exponential"),
    )


def unwrap_model(accelerator, model):
    model = accelerator.unwrap_model(model)
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def default_target_modules(transformer):
    targets = ["to_k", "to_q", "to_v", "to_out.0", "to_qkv_mlp_proj"]
    targets.extend(
        f"single_transformer_blocks.{index}.attn.to_out"
        for index in range(len(transformer.single_transformer_blocks))
    )
    return targets


def register_lora_hooks(accelerator, transformer, mixed_precision):
    transformer_type = type(unwrap_model(accelerator, transformer))

    def save_model_hook(models, weights, output_dir):
        model_to_save = None
        for model in models:
            if not isinstance(unwrap_model(accelerator, model), transformer_type):
                raise ValueError(f"Unexpected model in checkpoint: {type(model).__name__}")
            model_to_save = model
        if model_to_save is None:
            raise ValueError("No transformer model found in checkpoint state")
        if accelerator.is_main_process:
            lora_state = get_peft_model_state_dict(unwrap_model(accelerator, model_to_save))
            lora_state = {
                key: value.detach().cpu().contiguous() if isinstance(value, torch.Tensor) else value
                for key, value in lora_state.items()
            }
            Flux2KleinPipeline.save_lora_weights(
                save_directory=output_dir,
                transformer_lora_layers=lora_state,
            )
        if weights:
            weights.pop()

    def load_model_hook(models, input_dir):
        model_to_load = None
        while models:
            model = models.pop()
            if not isinstance(unwrap_model(accelerator, model), transformer_type):
                raise ValueError(f"Unexpected model in checkpoint: {type(model).__name__}")
            model_to_load = unwrap_model(accelerator, model)
        if model_to_load is None:
            raise ValueError("No transformer model found in checkpoint state")
        lora_state = Flux2KleinPipeline.lora_state_dict(input_dir)
        transformer_state = {
            key.removeprefix("transformer."): value
            for key, value in lora_state.items()
            if key.startswith("transformer.")
        }
        transformer_state = convert_unet_state_dict_to_peft(transformer_state)
        incompatible = set_peft_model_state_dict(model_to_load, transformer_state, adapter_name="default")
        unexpected = getattr(incompatible, "unexpected_keys", None)
        if unexpected:
            raise ValueError(f"Unexpected LoRA keys while loading {input_dir}: {unexpected}")
        if mixed_precision == "fp16":
            cast_training_params([model_to_load], dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)


def load_transformer(args, dtype):
    if args.transformer_model:
        return Flux2Transformer2DModel.from_pretrained(
            args.transformer_model,
            revision=args.revision,
            dtype=dtype,
            local_files_only=args.local_files_only,
        )
    return Flux2Transformer2DModel.from_pretrained(
        args.teacher_model,
        subfolder="transformer",
        revision=args.revision,
        dtype=dtype,
        local_files_only=args.local_files_only,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    data_source = parser.add_mutually_exclusive_group(required=True)
    data_source.add_argument("--cache_dir")
    data_source.add_argument("--image_dir", nargs="+", metavar="DIR")
    validation_source = parser.add_mutually_exclusive_group()
    validation_source.add_argument("--validation_cache_dir")
    validation_source.add_argument("--validation_image_dir", nargs="+", metavar="DIR")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--teacher_model", default=DEFAULT_TEACHER)
    parser.add_argument("--transformer_model")
    parser.add_argument("--revision")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_layers")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=10000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--weighting_scheme",
        choices=("none", "logit_normal", "mode", "sigma_sqrt", "cosmap"),
        default="none",
    )
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
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
    parser.add_argument("--num_validation_images", type=int, default=4)
    parser.add_argument("--validation_num_inference_steps", type=int, default=50)
    parser.add_argument(
        "--validation_scheduler",
        choices=("dpm_solver", "flow_match_euler"),
        default="dpm_solver",
    )
    parser.add_argument("--validation_guidance_scale", type=float, default=4.0)
    parser.add_argument("--validation_seed", type=int, default=0)
    parser.add_argument("--validation_height", type=int)
    parser.add_argument("--validation_width", type=int)
    parser.add_argument("--validation_negative_prompt_embeds")
    model_source = parser.add_mutually_exclusive_group()
    model_source.add_argument("--local_files_only", dest="local_files_only", action="store_true")
    model_source.add_argument("--allow_download", dest="local_files_only", action="store_false")
    parser.set_defaults(local_files_only=True)
    args = parser.parse_args()
    if args.max_train_steps < 1:
        parser.error("--max_train_steps must be at least 1")
    if args.rank < 1:
        parser.error("--rank must be at least 1")
    if args.resolution < 16 or args.resolution % 16:
        parser.error("--resolution must be a positive multiple of 16")
    if (args.validation_height is None) != (args.validation_width is None):
        parser.error("--validation_height and --validation_width must be specified together")
    if args.validation_height is not None and (args.validation_height < 16 or args.validation_width < 16):
        parser.error("Validation height and width must be at least 16")
    return args


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    project_config = ProjectConfiguration(
        project_dir=output_dir,
        logging_dir=output_dir / args.logging_dir,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
        project_config=project_config,
    )
    set_seed(args.seed, device_specific=True)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    is_image_dataset = args.image_dir is not None
    if is_image_dataset:
        dataset = ImageCaptionDataset(args.image_dir)
        collate_fn = collate_public_samples
    else:
        dataset = CachedFlux2Dataset(args.cache_dir)
        collate_fn = collate_cached_samples
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=not is_image_dataset,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=args.dataloader_num_workers > 0,
    )
    if len(dataloader) == 0:
        raise ValueError(
            f"The training dataloader is empty: dataset_size={len(dataset)}, "
            f"train_batch_size={args.train_batch_size}. Reduce --train_batch_size or add samples."
        )
    if accelerator.is_main_process:
        if is_image_dataset:
            print(f"Image roots: {len(args.image_dir):,}")
            print(f"Images found: {dataset.found_images:,}")
            print(f"Usable image/text pairs: {len(dataset):,}")
            print(f"Skipped without .txt: {dataset.missing_prompts:,}")
            print(f"Skipped empty prompts: {dataset.empty_prompts:,}")
            print(f"Skipped unreadable prompts: {dataset.unreadable_prompts:,}")
            print(f"Skipped broken images: {dataset.broken_images:,}")
        else:
            print(f"Cached training samples: {len(dataset):,}")

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.teacher_model,
        subfolder="scheduler",
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    transformer = load_transformer(args, weight_dtype)
    if not is_image_dataset:
        validate_cache_shapes(dataset, transformer)
    transformer.requires_grad_(False)

    uses_image_validation = args.validation_image_dir is not None or (
        is_image_dataset and args.validation_cache_dir is None
    )
    encoding_pipeline = None
    if is_image_dataset or (args.validation_epochs > 0 and uses_image_validation):
        encoding_pipeline = load_public_encoding_pipeline(
            args, weight_dtype, accelerator.device
        )

    target_modules = (
        [name.strip() for name in args.lora_layers.split(",") if name.strip()]
        if args.lora_layers
        else default_target_modules(transformer)
    )
    transformer.add_adapter(
        LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
    )
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    if accelerator.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)

    trainable_parameters = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError(f"No trainable LoRA parameters matched target modules: {target_modules}")

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

    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )
    register_lora_hooks(accelerator, transformer, accelerator.mixed_precision)

    if accelerator.is_main_process:
        tracker_config = {
            key: json.dumps(value) if isinstance(value, list) else value
            for key, value in vars(args).items()
        }
        accelerator.init_trackers("flux2-klein-lora", config=tracker_config)
        total = sum(parameter.numel() for parameter in unwrap_model(accelerator, transformer).parameters())
        trainable = sum(
            parameter.numel()
            for parameter in unwrap_model(accelerator, transformer).parameters()
            if parameter.requires_grad
        )
        print(f"Transformer parameters: {total:,} ({total / 1e9:.3f}B)")
        print(f"Trainable LoRA parameters: {trainable:,} ({100 * trainable / total:.4f}%)")
        print(f"LoRA target modules: {target_modules}")
        print(f"Hugging Face downloads enabled: {not args.local_files_only}")

    global_step = 0
    if args.resume_from_checkpoint:
        resume_path = (
            resolve_latest_checkpoint(output_dir)
            if args.resume_from_checkpoint == "latest"
            else Path(args.resume_from_checkpoint)
        )
        if resume_path is None or not resume_path.is_dir():
            raise ValueError(f"Checkpoint not found: {args.resume_from_checkpoint}")
        accelerator.load_state(resume_path)
        match = CHECKPOINT_PATTERN.fullmatch(resume_path.name)
        if match is None:
            raise ValueError(f"Checkpoint directory must be named checkpoint-<step>: {resume_path}")
        global_step = int(match.group(1))
        accelerator.print(f"Resumed complete training state from {resume_path} at step {global_step}")

    validation_samples = None
    validation_pipeline = None
    validation_negative_prompt_embeds = None
    if args.validation_epochs > 0:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if uses_image_validation:
                validation_dataset = ImageCaptionDataset(
                    args.validation_image_dir or args.image_dir
                )
                validation_samples = encode_image_validation_samples(
                    validation_dataset,
                    encoding_pipeline,
                    args.num_validation_images,
                    accelerator.device,
                    weight_dtype,
                )
                print(f"Usable validation image/text pairs: {len(validation_dataset):,}")
            else:
                validation_dataset = CachedFlux2Dataset(
                    args.validation_cache_dir or args.cache_dir
                )
                validation_samples = [
                    validation_dataset[index]
                    for index in range(min(args.num_validation_images, len(validation_dataset)))
                ]
            validation_pipeline = load_validation_pipeline(
                args,
                unwrap_model(accelerator, transformer),
                weight_dtype,
                accelerator.device,
            )
            configure_validation_scheduler(
                validation_pipeline, args.validation_scheduler
            )
            print(
                f"Validation scheduler: {validation_pipeline.scheduler.__class__.__name__}"
            )
            validation_negative_prompt_embeds = load_validation_negative_prompt_embeds(
                args,
                weight_dtype,
                accelerator.device,
                text_pipeline=encoding_pipeline,
            )
            base_images = generate_validation_images(
                accelerator,
                validation_pipeline,
                unwrap_model(accelerator, transformer),
                validation_samples,
                validation_negative_prompt_embeds,
                args,
                weight_dtype,
            )
            log_validation_images(
                accelerator, "validation/base", base_images, validation_samples, args, global_step
            )
        accelerator.wait_for_everyone()

    updates_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / updates_per_epoch)
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Steps",
    )

    for epoch in range(num_train_epochs):
        transformer.train()
        for batch in dataloader:
            with accelerator.accumulate(transformer):
                if is_image_dataset:
                    model_input, prompt_embeds = encode_public_batch(
                        encoding_pipeline,
                        batch,
                        args.resolution,
                        accelerator.device,
                        weight_dtype,
                    )
                else:
                    model_input = batch["latents"].to(
                        accelerator.device,
                        dtype=weight_dtype,
                        non_blocking=True,
                    )
                    prompt_embeds = batch["embeddings"].to(
                        accelerator.device,
                        dtype=weight_dtype,
                        non_blocking=True,
                    )
                noise = torch.randn_like(model_input)
                density = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=model_input.shape[0],
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                    device=accelerator.device,
                )
                timestep_indices = (
                    density * noise_scheduler.config.num_train_timesteps
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
                prediction = transformer_prediction(
                    transformer,
                    noisy_model_input,
                    prompt_embeds,
                    timesteps,
                    args.guidance_scale,
                )
                loss_weights = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme,
                    sigmas=sigmas,
                )
                loss = weighted_mse(prediction, packed_target, loss_weights)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {global_step}: {loss.item()}")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                logs = {
                    "train/loss": loss.detach().item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/epoch": epoch + 1,
                }
                progress_bar.set_postfix(loss=f"{logs['train/loss']:.4f}")
                accelerator.log(logs, step=global_step)

                if args.checkpointing_steps and global_step % args.checkpointing_steps == 0:
                    save_path = output_dir / f"checkpoint-{global_step}"
                    accelerator.save_state(save_path)
                    accelerator.print(f"Saved checkpoint: {save_path}")

            if global_step >= args.max_train_steps:
                break

        if args.validation_epochs > 0 and (epoch + 1) % args.validation_epochs == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                images = generate_validation_images(
                    accelerator,
                    validation_pipeline,
                    unwrap_model(accelerator, transformer),
                    validation_samples,
                    validation_negative_prompt_embeds,
                    args,
                    weight_dtype,
                )
                log_validation_images(
                    accelerator, "validation/lora", images, validation_samples, args, global_step
                )
            accelerator.wait_for_everyone()

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model = unwrap_model(accelerator, transformer)
        lora_state = get_peft_model_state_dict(model)
        lora_state = {
            key: value.detach().cpu().contiguous() if isinstance(value, torch.Tensor) else value
            for key, value in lora_state.items()
        }
        Flux2KleinPipeline.save_lora_weights(
            save_directory=output_dir,
            transformer_lora_layers=lora_state,
        )
        (output_dir / "training_args.json").write_text(
            json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Saved final Diffusers LoRA: {output_dir}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
