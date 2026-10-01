### 학습 distillation 데이터 구축


* CommonCatalog

[4]- 13290 Running                 nohup hf download common-canvas/commoncatalog-cc-by --repo-type dataset --include "*/least_dim_range=512-768/*.parquet" --local-dir /data1/jooyonglee/commoncatalog-cc-by > logs/commondata.l                        og 2>&1 &
[5]+ 13533 Running                 nohup hf download common-canvas/commoncatalog-cc-by --repo-type dataset --include "*/least_dim_range=768-1024/*.parquet" --local-dir /data1/jooyonglee/commoncatalog-cc-by > logs/com_768_102                        4.log 2>&1 &

## distillation 추론, 학습 주의사항

general_inference.py 기본:
scheduler       = dpmpp-2m-karras
guidance_scale  = 7.0
dtype           = auto → CUDA에서 fp16
seed            = random

predict_distillation.py 기본:
scheduler       = model
guidance_scale  = 4.0
dtype           = bf16
seed            = 0


학습 seed와 추론 seed는 같은 의미가 아닙니다.
- 학습 --seed 42: 초기화, timestep sampling, noise, dataloader 재현성
- 추론 --seed 42: 최초 latent noise 결정

--guidance-scale 4.0은 실제 CFG를 수행하므로 결과에 큰



## distillation training


batch 1 × GPU 2 × accumulation 4 = 8

1 epoch = 5000 / 8 = 약 625 steps
4000 steps = 약 6.4 epochs

~~~bash
nohup env \
CUDA_VISIBLE_DEVICES=4,5 \
HF_HUB_OFFLINE=1 \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
  --num_processes 2 \
  --main_process_port 29535 \
  --mixed_precision bf16 \
  my/mydisillation.py \
  --student_config my/student_configure_t2i_b.yaml \
  --public_dataset_root /data1/jooyonglee/commoncatalog-cc-by \
  --public_resolution 512 \
  --output_dir /data1/jooyonglee/training/diffuers/distillation/flux2-b1-t2i-b \
  --resume_from_checkpoint latest \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 400000 \
  --checkpointing_steps 6000 \
  --validation_steps 1000 \
  --learning_rate 2e-5 \
  --cfg_aware_probability 0.5 \
  --cfg_aware_guidance_scale 4.0 \
  --cfg_aware_warmup_steps 0 \
  --use_ema \
  --ema_device cpu \
  --ema_update_every 10 \
  --gradient_checkpointing \
  --allow_tf32 \
  --dataloader_num_workers 2 \
  --local_files_only \
  > logs/t2i_b_cfg_ema.log 2>&1 &



  
  --output_dir /data1/.../flux2-b1-t2i-b \
  --resume_from_checkpoint latest

# tensorboard 여러 디렉토리 설정
 nohup tensorboard \
  --logdir_spec="runB:/data1/jooyonglee/training/diffuers/distillation/flux2-b1-t2i-b/logs/,runB-scrach:/data1/jooyonglee/training/diffuers/distillation/flux2-b1-t2i-b-scratch/logs,runB-shift:/data1/jooyonglee/training/diffuers/distillation/flux2-b1-t2i-b-shift/logs/" \
  --port 8605 \
  --host 0.0.0.0 \
  --purge_orphaned_data false \
  --reload_interval 30 \
  --max_reload_threads 4 \
  > logs/board.log 2>&1 &

 nohup tensorboard \
  --logdir_spec="personStudent:/data1/jooyonglee/training/diffuers/distillation/flux2-b1-t2i-b-person-scratch-v2/logs/" \
  --port 8605 \
  --host 0.0.0.0 \
  --purge_orphaned_data false \
  --reload_interval 30 \
  --max_reload_threads 4 \
  > logs/board.log 2>&1 &

 56450 

nohup tensorboard \
  --logdir_spec="runC:/data1/jooyonglee/training/diffuers/distillation/flux2-b1-t2i-c" \
  --port 8605 \
  --host 0.0.0.0 \
  > logs/tensor.log 2>&1 &
# --resume_from_checkpoint latest \
# flux2-v2-1p18b-sensitivity

  # --prompt "a dog is running in the grass" \
  # --prompt "an old photo of a large house in the middle of a field" \
  # --prompt "the cover of the album santana by the band" \
# export CUDA_VISIBLE_DEVICES=4,5 HF_HUB_OFFLINE=1 
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 
python my/predict_distillation.py \
  --student /data1/jooyonglee/training/diffuers/distillation/flux2-b1-t2i-b/checkpoint-222000/ \
  --prompt-file "my/samples/commoncatalog" \
  --num-images 50 \
  --steps 50 \
  --guidance-scale 4.0 \
  --height 512 \
  --width 512 \
  --seed 0 \
  --output-dir outputs/t2i_b/checkpoint-guide-embed-common \
  --filename-prefix c-cfg4.0 \
  --dtype bf16 \
  --local-files-only

~~~
## editting 이미지 distilation 학습

~~~bash
nohup env \
CUDA_VISIBLE_DEVICES=2,3 \
HF_HUB_OFFLINE=1 \
PYTHONUNBUFFERED=1 \
accelerate launch \
  --num_processes 2 \
  --main_process_port 29520 \
  --mixed_precision bf16 \
  my/mydisillation.py \
  --student_config my/configs/editing/phase2_magicbrush_single_b.yaml \
  --student_init /data1/jooyonglee/training/diffuers/distillation/flux2-b1-t2i-b/checkpoint-216000/ \
  --editing_dataset_root /data1/jooyonglee/public/magicbrush_subset-20000/train \
  --public_resolution 512 \
  --output_dir /data1/jooyonglee/training/diffuers/distillation/flux2-edit-phase2-b \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 800000 \
  --checkpointing_steps 6000 \
  --validation_steps 1000 \
  --gradient_checkpointing \
  --allow_tf32 \
  --dataloader_num_workers 2 \
  --local_files_only > logs/edit_b.log 2>&1 &

# validate 데이터 잇을때만
  --validation_editing_dataset_root /data1/jooyonglee/public/magicbrush_subset-20000/dev \
echo "PID=$!"
~~~


~~~bash

export CUDA_VISIBLE_DEVICES=2,3
export HF_HUB_OFFLINE=1

accelerate launch \
  --num_processes 2 \
  --mixed_precision bf16 \
  my/mydisillation.py \
  --student_config my/student_configure.yaml \
  --public_dataset_root /data1/jooyonglee/commoncatalog-cc-by \
  --public_resolution 512 \
  --output_dir /data1/jooyonglee/training/diffuers/distillation/flux2-v2-1b \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_train_steps 10000000 \
  --gradient_checkpointing \
  --allow_tf32 \
  --dataloader_num_workers 2 \
  --checkpointing_steps 10000 \
  --validation_steps 5000 \
  --num_validation_images 2 \
  --local_files_only


accelerate launch \
  --num_processes 2 \
  --mixed_precision bf16 \
  my/mydisillation.py \
  --student_config my/student_configure_1b_1d6s.yaml \
  --public_dataset_root /data1/jooyonglee/commoncatalog-cc-by \
  --public_resolution 512 \
  --output_dir /data1/jooyonglee/training/diffuers/distillation/flux2-v2-1p18b-sensitivity \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_train_steps 10000000 \
  --gradient_checkpointing \
  --allow_tf32 \
  --dataloader_num_workers 2 \
  --checkpointing_steps 10000 \
  --validation_steps 5000 \
  --num_validation_images 2 \
  --local_files_only
~~~

## Lora 4B 학습


~~~bash

export CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 
python my/mydisillation_lora.py \
  --image_dir \
    ./outputs/mytest \
    ./outputs/mytest00 \
    ./outputs/mytest00 \
    ./outputs/mytest4 \
    ./outputs/mytest5 \
  --validation_image_dir ./outputs/mytest5 \
  --resolution 512 \
  --output_dir ./output/flux2-4b-lora_mytest \
  --rank 16 \
  --lora_alpha 16 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --max_train_steps 10000 \
  --learning_rate 1e-4 \
  --gradient_checkpointing \
  --allow_tf32 \
  --validation_epochs 1

~~~





flux-4b  

~~~
#  ~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294/
snapshot/
├── model_index.json
├── scheduler/
│   └── scheduler_config.json
├── tokenizer/
│   └── ...
├── text_encoder/                       # Qwen3-4B
│   ├── config.json
│   ├── model.safetensors.index.json
│   ├── model-00001-of-00002.safetensors
│   └── model-00002-of-00002.safetensors
├── transformer/                        # FLUX.2 DiT
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── vae/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
└── flux-2-klein-base-4b.safetensors    # 원본 형식 DiT만
~~~

# teacher & student 실행

## teacher local weights 실행

~~~bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 

  python my/general_inference.py \
  --weights /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-base-4B/snapshots/a3b4f4849157f664bdbc776fd7453c2783562f4d \
  --model-type auto \
  --local-files-only \
  --prompt-file my/samples/test/sample.txt \
  --num-images -1 \
  --steps 50 \
  --guidance-scale 4.0 \
  --height 512 \
  --width 512 \
  --seed 42 \
  --scheduler model \
  --clip-skip 1 \
  --dtype bf16 \
  --device cuda:0 \
  --output-dir ./outputs/teacher22_50
~~~


~~~ 
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 \
python my/predict_distillation.py \
  --student /data1/jooyonglee/training/diffuers/distillation/flux2/checkpoint-270000/ \
  --base-model /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-base-4B/snapshots/a3b4f4849157f664bdbc776fd7453c2783562f4d \
  --local-files-only \
  --prompt-file my/samples/commoncatalog/prompts.txt \
  --num-images -1 \
  --steps 50 \
  --guidance-scale 4.0 \
  --height 512 \
  --width 512 \
  --seed 42 \
  --scheduler model \
  --dtype bf16 \
  --device cuda:0 \
  --output-dir ./outputs/student
  ~~~


process group ID(PGID)인지 종료

kill -- -id
example) kill -- -1212



# teacher를 이용한 이미지 생성

~~~bash
nohup env \
CUDA_VISIBLE_DEVICES=1 \
HF_HUB_OFFLINE=1 \
PYTHONUNBUFFERED=1 \
/opt/conda/envs/diffuser_lora/bin/python \
my/build_flux2_teacher_cache.py \
  --prompt-jsonl my/samples/flux2_distillation_person_prompts_20000.jsonl \
  --output-dir /data1/jooyonglee/public/flux2_teacher_cache/person_20000_base4b \
  --teacher-model /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-base-4B/snapshots/a3b4f4849157f664bdbc776fd7453c2783562f4d \
  --device cuda:0 \
  --dtype bf16 \
  --batch-size 1 \
  --steps 50 \
  --guidance-scale 4.0 \
  --height 512 \
  --width 512 \
  --seed 0 \
  --preview-every 50 \
  --save-embeddings \
  --local-files-only \
  > logs/teacher_person_base4b.log 2>&1 &




for shard in 0 1 2 3; do
  gpu=$((shard + 2))

  nohup env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    /opt/conda/envs/diffuser_lora/bin/python \
    my/build_flux2_teacher_cache.py \
      --prompt-jsonl my/samples/flux2_distillation_person_prompts_20000.jsonl \
      --output-dir /data1/jooyonglee/public/flux2_teacher_cache/person_20000_base4b \
      --teacher-model /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-base-4B/snapshots/a3b4f4849157f664bdbc776fd7453c2783562f4d \
      --device cuda:0 \
      --dtype bf16 \
      --batch-size 1 \
      --steps 50 \
      --guidance-scale 4.0 \
      --height 512 \
      --width 512 \
      --seed 0 \
      --max-samples 5000 \
      --preview-every 50 \
      --log-every 10 \
      --num-shards 4 \
      --shard-index "$shard" \
      --save-embeddings \
      --local-files-only \
      > "logs/teacher_person_base4b_shard${shard}.log" 2>&1 &
done
~~~


~~~
person_20000_base4b/
├── person-xxxxxxxxxxxxxxxx/
│   ├── prompt.txt
│   ├── embedding.pt
│   ├── latent.pt
│   ├── metadata.json
│   ├── preview.png       # preview-every에 해당할 때만
│   └── complete.json
└── manifest-shard-00000.jsonl
~~~

~~~
latent.pt:
  shape = (128, 32, 32)
  dtype = bfloat16
  format = normalized_patchified_chw

embedding.pt:
  shape = (512, 7680)
  dtype = bfloat16
~~~


## tearcher embedding학습하기


~~~bash
nohup env \
  CUDA_VISIBLE_DEVICES=4,5 \
  HF_HUB_OFFLINE=1 \
  PYTHONUNBUFFERED=1 \
  /opt/conda/envs/diffuser_lora/bin/accelerate launch \
    --num_processes 2 \
    --main_process_port 29530 \
    --mixed_precision bf16 \
    my/mydisillation.py \
    --student_config my/student_configure_t2i_b.yaml \
    --cache_dir /data1/jooyonglee/public/flux2_teacher_cache/person_20000_base4b \
    --teacher_model /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-base-4B/snapshots/a3b4f4849157f664bdbc776fd7453c2783562f4d \
    --output_dir /data1/jooyonglee/training/diffuers/distillation/flux2-b1-t2i-b-person-scratch-v2 \
    --max_train_samples 4998 \
    --resume_from_checkpoint latest \
    --train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --max_train_steps 100000 \
    --learning_rate 2e-5 \
    --checkpointing_steps 5000 \
    --validation_steps 1000 \
    --num_validation_images 6 \
    --validation_prompt_file '' \
    --validation_cache_dir /data1/jooyonglee/public/flux2_teacher_cache/person_20000_base4b \
    --cfg_aware_probability 1.0 \
    --cfg_aware_guidance_scale 4.0 \
    --use_ema \
    --ema_device gpu \
    --ema_update_every 10 \
    --gradient_checkpointing \
    --allow_tf32 \
    --dataloader_num_workers 2 \
    --local_files_only \
    > logs/t2i_b_person_scratch_v2_resume7gpu.log 2>&1 &
  ~~~