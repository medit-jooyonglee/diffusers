python my/general_inference.py \
  --prompt "portrait cute girl" \
  --num-images 10


# Illustrious
python my/general_inference.py \
  --weights /path/to/illustrious.safetensors \
  --model "/data1/jooyonglee/pretrained/diffusions/illustrious/waiIllustriousSDXL_v170.safetensors" \
  --steps 50 \
  --model-type illustrious \
  --prompt-file "/workspace/dataset/prompts/test/even-more-random-prompts1.txt" \
  --negative-prompt-file "/workspace/dataset/prompts/test/negative.txt" \
  --num-images 300 \
  --lora "/data1/jooyonglee/pretrained/diffusions/illustrious/lora/add-detail-xl.safetensors" \
  --lora "/data1/jooyonglee/pretrained/diffusions/illustrious/lora/ponyv3_ill01_2_adamW-000017.safetensors" \
  --lora-adapter-name detail \
  --lora-adapter-name phony \
  --output-dir outputs/mytest \
  --lora-scale 0.7 \
  --lora-scale 0.5


  python my/general_inference.py \
  --weights /path/to/illustrious.safetensors \
  --model "/data1/jooyonglee/pretrained/diffusions/sd15/chikmix_V3.safetensors" \
  --steps 50 \
  --model-type sd15 \
  --prompt-file "/workspace/dataset/prompts/test/even-more-random-prompts1.txt" \
  --negative-prompt-file "/workspace/dataset/prompts/test/negative.txt" \
  --num-images 500 \
  --output-dir outputs/mytest00 \
  --device cuda:3


    python my/general_inference.py \
  --weights /path/to/illustrious.safetensors \
  --model "/data1/jooyonglee/pretrained/diffusions/sd15/chikmix_V3.safetensors" \
  --steps 50 \
  --model-type sd15 \
  --prompt-file "/data1/jooyonglee/pretrained/diffusions/sd15/lora/prompt.txt" \
  --negative-prompt-file "/data1/jooyonglee/pretrained/diffusions/sd15/lora/negative_prompt.txt" \
  --num-images 300 \
  --lora "/data1/jooyonglee/pretrained/diffusions/sd15/lora/more_details.safetensors" \
  --lora "/data1/jooyonglee/pretrained/diffusions/sd15/lora/ClothingAdjuster2.safetensors" \
  --lora-adapter-name detail \
  --lora-adapter-name phony \
  --output-dir outputs/mytest01 \
  --lora-scale 1.0 \
  --lora-scale 1.0 \
  --device cuda:3



CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 \
python my/general_inference.py \
  --model black-forest-labs/FLUX.2-klein-base-4B \
  --model-type auto \
  --local-files-only \
  --lora ./output/flux2-4b-lora_mytest/checkpoint-10000 \
  --lora-adapter-name trained \
  --lora-scale 1.0 \
  --prompt "8k,ultra detailed,perfect anatomy,beautiful lighting,uncensored,laboratory,pretty,beautiful,photorealistic,32mm camera photo,textured skin,pores,,(doakasumi, brown hair, ponytail, hair ribbon, choker),spread legs,close up,sex machine,dildo in pussy,squirting,orgasm,moaning in pleasure" \
  --negative-prompt-file "/workspace/dataset/prompts/test/negative.txt" \
  --num-images 100 \
  --batch-size 1 \
  --steps 4 \
  --guidance-scale 4.0 \
  --width 512 \
  --height 512 \
  --scheduler model \
  --dtype bf16 \
  --device cuda:6 \
  --clip-skip 1 \
  --seed 42 \
  --output-dir ./outputs/flux2_lora_test

  # --prompt-file "/workspace/dataset/prompts/test/even-more-random-prompts1.txt" \


  python my/predfict_distillation.py \
  --student /data1/jooyonglee/training/diffuers/distillation/flux2/checkpoint-100000/ \
  --prompt "an old photo of a large house in the middle of a field" \
  --prompt="two people walking with bikes at a triathlon" \
  --prompt="a view of a road with a telephone pole in the middle" \
  --prompt="a black and white photo of a tree in the sky" \
  --prompt="a red telephone booth with a bicycle parked next to it" \
  --num-images 10 \
  --steps 50 \
  --guidance-scale 4.0 \
  --height 512 \
  --width 512 \
  --seed 42 \
  --scheduler model \
  --dtype bf16 \
  --device cuda:0 \
  --output-dir ./outputs/distilled-studentpt


  # CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 \
python my/predict_distillation.py \
  --student /data1/jooyonglee/training/diffuers/distillation/flux2/checkpoint-270000/ \
  --prompt-file "my/samples/commoncatalog/prompts.txt" \
  --num-images 300 \
  --steps 50 \
  --guidance-scale 4.0 \
  --height 512 \
  --width 512 \
  --seed 42 \
  --scheduler model \
  --dtype bf16 \
  --device cuda:0 \
  --output-dir ./outputs/distilled-studentpt0921

