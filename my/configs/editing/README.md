# MagicBrush editing distillation

이 문서는 `my/mydisillation.py`로 FLUX.2 Klein capacity-distilled student를 MagicBrush editing에 학습하는 실행 요약이다.

## 데이터 구조

`--editing_dataset_root`는 `metadata.jsonl`, `source/`, `target/`, `mask/`가 있는 split 디렉터리를 받는다.

MagicBrush 공식 train split은 8,807 edit turn이다. crawler에 `--num-samples 20000`을 지정해도 중복 없이 저장되는 최대치는 split 전체 크기이며, 완료 시 실제 저장 수를 출력한다.

```text
/data1/jooyonglee/public/magicbrush_subset-20000/
├── train/
│   ├── metadata.jsonl
│   ├── source/
│   ├── target/
│   └── mask/
└── dev/
    ├── metadata.jsonl
    ├── source/
    ├── target/
    └── mask/
```

dev split은 별도로 받는다.

```bash
python my/dataset_collect/magic_brush_crawl.py \
  --output-dir /data1/jooyonglee/public/magicbrush_subset-20000 \
  --split dev \
  --num-samples 528 \
  --no-shuffle
```

## 학습 모드

- Phase 2 `single`: 현재 turn의 source 한 장과 instruction으로 target을 학습한다.
- Phase 3 `multi_reference`: 동일 session의 최초 source와 현재 source 두 장으로 현재 target을 학습한다. 두 번째 turn 이상만 사용한다.
- MagicBrush mask는 현재 학습 conditioning에 사용하지 않는다.
- `--student_init`은 T2I student transformer만 불러오며 optimizer와 global step은 새로 시작한다.
- `--resume_from_checkpoint latest`는 동일 editing output directory의 중단된 학습을 이어갈 때만 사용한다.
- Phase 2/3 YAML은 forward-noised 배치 80%와 2-step student trajectory 배치 20%를 기본으로 혼합한다.
- Trajectory rollout은 reference latent를 그대로 유지하므로 single/multi-reference editing 양쪽에 적용된다.
- 추가 gradient graph를 유지하지 않으므로 peak activation memory 증가는 작지만 기본값 기준 forward 계산량은 약 20% 증가한다.

기본 trajectory 설정은 YAML의 `distillation` section에서 조절한다.

```yaml
lambda_direction: 0.05
lambda_trajectory: 1.0
trajectory_probability: 0.20
trajectory_rollout_steps: 2
trajectory_num_inference_steps: 50
trajectory_warmup_steps: 1000
```

## EMA

T2I A/B/C와 Phase 2/3 A/B/C 설정은 student transformer의 FP32 EMA를 기본으로 사용한다. EMA는 epoch가 아니라 매 optimizer step마다 갱신한다. Teacher와 학습 전용 hidden projection에는 EMA를 적용하지 않는다.

```yaml
optimization:
  use_ema: true
  ema_device: gpu
  ema_decay: 0.9999
  ema_update_every: 1
  ema_update_after_step: 0
  ema_use_warmup: true
  ema_inv_gamma: 1.0
  ema_power: 0.75
```

GPU EMA는 B/C 기준 rank 0 GPU에 약 4.3~4.4GiB를 추가 사용한다. OOM이면 다른 설정은 유지한 채 CLI에 다음 override를 추가한다.

```bash
--ema_device cpu
```

CPU EMA는 rank 0의 pinned host memory에 FP32 shadow와 staging buffer를 둔다. 현재 장비에서 실제 B 모델(1.158B)을 측정한 결과 optimizer step당 GPU EMA는 약 0.021초, CPU EMA는 약 0.262초였다. GPU 메모리에는 full student copy를 추가하지 않는다. Checkpoint에는 raw `transformer/`와 EMA `transformer_ema/`가 함께 저장되고, validation은 `validation/student`와 `validation/student_ema`를 모두 기록한다.

## Phase 2: single-image editing

B형의 T2I checkpoint-48000에서 시작하는 기본 명령이다. 현재 고정 seed validation에서는 B가 C보다 안정적인 경향을 보여 새 editing 실험의 우선 기준으로 둔다.

```bash
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1

accelerate launch \
  --num_processes 2 \
  --main_process_port 29510 \
  --mixed_precision bf16 \
  my/mydisillation.py \
  --student_config my/configs/editing/phase2_magicbrush_single_b.yaml \
  --student_init /data1/jooyonglee/training/diffuers/distillation/flux2-b1-t2i-b/checkpoint-48000 \
  --editing_dataset_root /data1/jooyonglee/public/magicbrush_subset-20000/train \
  --validation_editing_dataset_root /data1/jooyonglee/public/magicbrush_subset-20000/dev \
  --public_resolution 512 \
  --output_dir /data1/jooyonglee/training/diffuers/distillation/flux2-edit-phase2-b \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 10000 \
  --checkpointing_steps 1000 \
  --validation_steps 1000 \
  --gradient_checkpointing \
  --allow_tf32 \
  --dataloader_num_workers 2 \
  --local_files_only
```

A/C는 config, T2I checkpoint, output directory의 `b`를 각각 `a`, `c`로 바꾼다.

## Phase 3: session-history multi-reference editing

먼저 Phase 2 B의 확정 checkpoint를 초기값으로 사용한다. 복수 reference는 token 수와 attention memory가 늘어나지만 B형, batch 1, gradient checkpointing 조합은 A6000 48GB의 512 해상도 trajectory 1-step smoke test를 통과했다.

```bash
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1

accelerate launch \
  --num_processes 2 \
  --main_process_port 29511 \
  --mixed_precision bf16 \
  my/mydisillation.py \
  --student_config my/configs/editing/phase3_magicbrush_multiref_b.yaml \
  --student_init /data1/jooyonglee/training/diffuers/distillation/flux2-edit-phase2-b/checkpoint-10000 \
  --editing_dataset_root /data1/jooyonglee/public/magicbrush_subset-20000/train \
  --validation_editing_dataset_root /data1/jooyonglee/public/magicbrush_subset-20000/dev \
  --public_resolution 512 \
  --validation_height 512 \
  --validation_width 512 \
  --output_dir /data1/jooyonglee/training/diffuers/distillation/flux2-edit-phase3-b \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 10000 \
  --checkpointing_steps 1000 \
  --validation_steps 1000 \
  --gradient_checkpointing \
  --allow_tf32 \
  --dataloader_num_workers 2 \
  --local_files_only
```

장시간 실행에서 OOM이 발생하면 resolution과 validation 크기를 384로 낮춘다.

## 재개

동일 output directory에 checkpoint가 있을 때만 `latest`를 사용한다.

```bash
accelerate launch \
  --num_processes 2 \
  --mixed_precision bf16 \
  my/mydisillation.py \
  --student_config my/configs/editing/phase2_magicbrush_single_b.yaml \
  --editing_dataset_root /data1/jooyonglee/public/magicbrush_subset-20000/train \
  --validation_editing_dataset_root /data1/jooyonglee/public/magicbrush_subset-20000/dev \
  --output_dir /data1/jooyonglee/training/diffuers/distillation/flux2-edit-phase2-b \
  --resume_from_checkpoint latest \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 20000 \
  --checkpointing_steps 1000 \
  --validation_steps 1000 \
  --gradient_checkpointing \
  --allow_tf32 \
  --dataloader_num_workers 2 \
  --local_files_only
```

validation 결과는 다음 위치에 source reference, target, teacher/student 생성 이미지와 JSON metadata로 저장된다.

```text
OUTPUT_DIR/validation/step-XXXXXXXX/
```
