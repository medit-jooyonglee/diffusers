### 학습 distillation 데이터 구축


* CommonCatalog

[4]- 13290 Running                 nohup hf download common-canvas/commoncatalog-cc-by --repo-type dataset --include "*/least_dim_range=512-768/*.parquet" --local-dir /data1/jooyonglee/commoncatalog-cc-by > logs/commondata.l                        og 2>&1 &
[5]+ 13533 Running                 nohup hf download common-canvas/commoncatalog-cc-by --repo-type dataset --include "*/least_dim_range=768-1024/*.parquet" --local-dir /data1/jooyonglee/commoncatalog-cc-by > logs/com_768_102                        4.log 2>&1 &


## distillation training


~~~bash
export CUDA_VISIBLE_DEVICES=2,3 HF_HUB_OFFLINE=1 
accelerate launch \
  --num_processes 2 \
  --mixed_precision bf16 \
  my/mydisillation.py \
  --public_dataset_root /data1/jooyonglee/commoncatalog-cc-by \
  --public_resolution 512 \
  --output_dir /data1/jooyonglee/training/diffuers/distillation/flux2 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_train_steps 100000 \
  --learning_rate 1e-4 \
  --gradient_checkpointing \
  --allow_tf32 \
  --dataloader_num_workers 2 \
  --checkpointing_steps 1000 \
  --validation_steps 5000 \
  --num_validation_images 2
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