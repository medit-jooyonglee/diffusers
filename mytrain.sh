accelerate launch --config_file accelerate_config.yaml examples/text_to_image/train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"    \
  --train_data_dir="/workspace/dataset/images/avengers/" \
  --resolution=1024   --train_batch_size=1   \
  --learning_rate=1e-4   --rank=8   \
  --checkpointing_steps=500 \
  --max_train_steps=10000   \
  --mixed_precision="fp16" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --output_dir="./output/lora_train_avengers_fp16" \
  --validation_prompt "avngrsstyle superhero, superhero, power armor, " \
  --num_validation_images 2 \
  --validation_epochs 1 \
  --resume_from_checkpoint="latest"

