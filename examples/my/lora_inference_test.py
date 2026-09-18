"""Quick manual test script for running inference with a trained SDXL LoRA.

See lora_train.md section 21 ("LoRA Inference") for the walkthrough this follows.

Usage:
    python lora_inference_test.py \
        --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
        --lora_path ./smoke_test \
        --prompt "woman, test pc, human" \
        --output_path output/results \
        --num_generate 4

Each run of the same prompt is saved as its own timestamped (ms-precision)
file under --output_path, so repeated runs never overwrite each other.

Add --fuse_lora to merge the LoRA into the base weights before inference
(see lora_train.md section 22, "LoRA Merge").
"""
import random
import argparse
import os
import time

import torch

from diffusers import StableDiffusionXLPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run SDXL + LoRA inference.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Path to the base SDXL model on the Hub or locally.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="",
        help="Directory containing the trained LoRA weights (see mytrain.sh --output_dir).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="woman, test pc, human",
        help="prompt",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
    )
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_path", type=str, default="output/results")
    parser.add_argument("--num_generate", type=int, default=1)
    parser.add_argument(
        "--fuse_lora",
        action="store_true",
        help="Merge the LoRA weights into the base model before inference (see lora_train.md section 22).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda:5") if torch.cuda.is_available() else torch.device("cpu")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
    ).to(device)

    if args.lora_path:
        pipe.load_lora_weights(args.lora_path,
                               weight_name="pytorch_lora_weights.safetensors",)
        
        
        # pipe.set_adapters(
            # "avengers",
            # adapter_weights=args.lora_scale,
        # )

        print("Available LoRA:", pipe.get_list_adapters())
        print("Active LoRA:", pipe.get_active_adapters())

        if args.fuse_lora:
            pipe.fuse_lora(lora_scale=args.lora_scale)

    cross_attention_kwargs = None if args.fuse_lora else {"scale": args.lora_scale}

    os.makedirs(args.output_path, exist_ok=True)

    for i in range(args.num_generate):
        generator = None
        if args.seed is not None:
            generator = torch.Generator(device=device).manual_seed(args.seed + i)

        # ran_choice_prompt = random.choice(args.prompt)
        random_token = False
        if random_token:
            clean_tokens = [token.strip() for token in args.prompt.split(",")]
            random_sel_prompt = ', '.join(random.choices(clean_tokens, k=int(len(clean_tokens) * 0.5)))
            prompt = random_sel_prompt
        else:
            prompt = args.prompt
        image = pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            cross_attention_kwargs=cross_attention_kwargs,
            generator=generator,
        ).images[0]

        timestamp_ms = time.strftime("%Y%m%d_%H%M%S", time.localtime()) + f"_{int(time.time() * 1000) % 1000:03d}"
        output_file = os.path.join(args.output_path, f"{timestamp_ms}.png")
        image.save(output_file)
        print(f"Saved image to {output_file}")


if __name__ == "__main__":
    main()
