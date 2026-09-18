# FLUX.2 Distillation 및 LoRA 학습

## 개요

이 폴더에는 Diffusers의 FLUX.2 Klein 구현을 사용하는 두 개의 학습 스크립트가 있다.

| 파일 | 용도 |
| --- | --- |
| `mydisillation.py` | FLUX.2 Klein base 4B teacher를 축소된 student로 capacity distillation |
| `mydisillation_lora.py` | FLUX.2 Klein transformer 또는 distillation student에 LoRA 학습 |

구현 기준 문서는 `/workspace/repo/flux2/docs/flux2_distillation_05b_1b.md`이다. 두 스크립트 모두 이미지 VAE latent와 텍스트 embedding을 미리 계산한 캐시를 사용하므로 학습 중 VAE와 text encoder를 올리지 않는다.

## 모델과 Hugging Face 캐시

기본 모델 ID는 다음과 같다.

```text
black-forest-labs/FLUX.2-klein-base-4B
```

현재 확인된 로컬 캐시는 다음 위치에 있다.

```text
/root/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-base-4B
/root/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B
```

두 스크립트는 기본적으로 `local_files_only=True`로 실행된다. 모델 ID를 넘겨도 Hugging Face가 위 로컬 캐시에서 snapshot을 찾아 사용하며 네트워크 다운로드를 시도하지 않는다.

- 완전 오프라인 강제: `HF_HUB_OFFLINE=1`
- 다운로드 허용: `--allow_download`
- 특정 snapshot 직접 사용: `--teacher_model /path/to/snapshot`

`HF_HUB_OFFLINE=1`은 Hugging Face Hub 전체를 오프라인으로 만드는 환경변수다. 스크립트 자체의 `local_files_only=True`와 함께 사용하면 다운로드를 확실히 차단할 수 있다.

## 데이터 캐시 포맷

학습 데이터는 샘플별 하위 디렉터리 구조를 사용한다.

```text
cache_dir/
├── sample_000001/
│   ├── latent.pt
│   ├── embedding.pt
│   └── prompt.txt
├── sample_000002/
│   ├── latent.pt
│   ├── embedding.pt
│   └── prompt.txt
└── ...
```

지원되는 파일 이름은 다음과 같다.

| 데이터 | 기본 이름 | 호환 이름 | 예상 shape |
| --- | --- | --- | --- |
| FLUX.2 latent | `latent.pt` | `image_latent.pt` | `(128, H, W)` |
| Qwen prompt embedding | `embedding.pt` | `text_embedding.pt` | `(L, 7680)` |
| 원문 prompt | `prompt.txt` | 없음 | 선택 사항 |

`latent.pt`는 단순한 VAE 출력이 아니라 FLUX.2 VAE normalization과 patchify까지 끝난 transformer 입력이어야 한다. 배치 안의 latent와 embedding shape은 모두 같아야 한다.

## CommonCatalog 스트리밍 입력

`mydisillation.py`는 기존 캐시와 별도로 `CommonCatalogDataset`을 직접 연결하는 스트리밍 모드를 지원한다. 두 입력은 상호 배타적이다.

```text
--cache_dir             기존 latent/embedding 캐시 사용
--public_dataset_root   parquet 이미지/caption을 학습 중 직접 인코딩
```

스트리밍 모드는 현재 디스크에 존재하는 parquet 파일만 시작 시점에 수집한다. 다운로드가 끝난 새 파일을 추가하려면 학습을 재시작하거나 checkpoint에서 resume해야 한다. 읽을 수 없는 parquet, 깨진 이미지, caption이 없는 row는 건너뛴다.

현재 `/data1/jooyonglee/commoncatalog-cc-by`에서 확인된 규모는 다음과 같다.

```text
parquet files: 287
raw rows: 1,488,995
file size total: 약 353 GB
```

Raw row 하나가 일반적으로 이미지 하나지만 깨진 이미지, 빈 caption과 중복 가능성이 있으므로 실제 유효 학습 샘플 수는 이보다 적다.

각 batch에서 다음 전처리를 실행한다.

1. PIL 이미지를 RGB로 변환한다.
2. 짧은 변을 `--public_resolution`에 맞추고 중앙 crop한다.
3. 픽셀을 `[-1, 1]`로 변환한다.
4. base 4B VAE로 encode하고 patchify 및 BN normalization한다.
5. base 4B Qwen text encoder로 caption embedding을 계산한다.
6. 같은 latent와 embedding을 teacher와 student에 전달한다.

기본 해상도는 512이며 16의 배수만 허용된다. VAE와 Qwen은 각 GPU에 하나씩 올라가므로 캐시 모드보다 VRAM과 연산량이 증가한다.

### GPU 2·3에서 스트리밍 학습

```bash
CUDA_VISIBLE_DEVICES=2,3 HF_HUB_OFFLINE=1 \
accelerate launch --num_processes 2 --mixed_precision bf16 \
  my/mydisillation.py \
  --public_dataset_root /data1/jooyonglee/commoncatalog-cc-by \
  --public_resolution 512 \
  --output_dir ./output/flux2-public-distill-1b \
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
```

`CUDA_VISIBLE_DEVICES=2,3`을 사용하면 프로세스 내부에서는 각각 `cuda:0`, `cuda:1`로 보인다. Dataset은 Accelerate rank와 DataLoader worker별로 parquet 파일을 분할하므로 동일 파일을 중복 순회하지 않는다.

```text
global batch = train_batch_size × GPU 수 × gradient_accumulation_steps
```

위 명령의 global batch는 2다. Raw rows 전체를 한 번 보는 데 필요한 optimizer step은 유효 샘플 수를 global batch로 나눈 값이며, 현재 metadata 기준으로는 약 744,498 step이다. 데이터 규모가 확정되지 않았으므로 스트리밍 모드에서는 epoch 대신 `--max_train_steps`로 종료하고 `--validation_steps`로 검증 주기를 지정한다.

별도 `--validation_public_dataset_root`를 지정하지 않으면 training root의 첫 유효 샘플들을 validation에 사용한다. 실제 품질 평가에는 별도의 validation parquet 디렉터리를 권장한다.

## Distillation 구현

`mydisillation.py`는 4B teacher의 transformer layer 수를 줄여 약 1B student를 만든다.

기본 student 구성은 다음과 같다.

```text
student_num_layers=2
student_num_single_layers=3
parameter count ≈ 1.054B
```

초기화 시 공통 입력·출력·modulation 모듈을 teacher에서 복사하고, teacher layer를 균등 간격으로 선택한다.

```text
double-stream mapping: [0, 4]
single-stream mapping: [0, 10, 19]
```

`--random_init`을 지정하면 teacher layer 복사를 생략한다.

학습 입력과 target은 FLUX.2 flow-matching 방식이다.

```text
x_t = (1 - sigma) * x_0 + sigma * noise
flow_target = noise - x_0
```

최종 loss는 ground-truth flow loss와 teacher prediction distillation loss의 가중합이다.

```text
loss = lambda_gt * loss_gt + lambda_kd * loss_kd
```

기본값은 `lambda_gt=1.0`, `lambda_kd=1.0`이다. 로그에는 다음 값이 기록된다.

- `train/loss`
- `train/loss_gt`
- `train/loss_kd`
- `train/teacher_gt`
- `train/lr`
- `train/epoch`

### Distillation 실행

물리 GPU 3만 노출하면 프로세스 내부에서는 해당 GPU가 `cuda:0`으로 보인다. 별도의 `--device cuda:3` 옵션은 사용하지 않는다.

```bash
CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 \
python my/mydisillation.py \
  --cache_dir /data1/jooyonglee/smiledesign_cache \
  --validation_cache_dir /data1/jooyonglee/smiledesign_validation_cache \
  --output_dir ./output/flux2-distill-1b \
  --student_num_layers 2 \
  --student_num_single_layers 3 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --max_train_steps 50000 \
  --learning_rate 1e-4 \
  --gradient_checkpointing \
  --allow_tf32 \
  --checkpointing_steps 1000 \
  --validation_epochs 1
```

학습 재개:

```bash
CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 \
python my/mydisillation.py \
  --cache_dir /data1/jooyonglee/smiledesign_cache \
  --output_dir ./output/flux2-distill-1b \
  --resume_from_checkpoint latest \
  --gradient_checkpointing
```

최종 student는 다음 위치에 Diffusers transformer 형식으로 저장된다.

```text
output/flux2-distill-1b/student/
├── config.json
└── diffusion_pytorch_model.safetensors
```

## LoRA 구현

`mydisillation_lora.py`는 transformer 원본 가중치를 고정하고 PEFT adapter만 학습한다. 기본 LoRA target은 다음 계층이다.

- `to_q`, `to_k`, `to_v`
- `to_out.0`
- `to_qkv_mlp_proj`
- 각 single transformer block의 `attn.to_out`

기본 rank와 alpha는 모두 32다. 다른 계층만 학습하려면 쉼표로 구분한 `--lora_layers`를 지정할 수 있다. rank는 데이터 개수에 비례해 정하지 않는다. 단순 스타일이나 한 인물은 보통 8–16부터 시작하고, 여러 개념이나 복잡한 변화는 16–32를 비교한다. 64 이상은 표현력과 함께 optimizer 메모리 및 과적합 위험도 증가한다.

새 이미지 학습은 이미지와 같은 디렉터리에 같은 basename의 `.txt` prompt를 둔다. 하위 디렉터리는 재귀적으로 검색한다.

```text
train-a/000001.jpg
train-a/000001.txt
train-a/nested/000002.png
train-a/nested/000002.txt
train-b/example.webp
train-b/example.txt
```

여러 데이터 디렉터리는 `--image_dir DIR_A DIR_B ...`처럼 한 옵션 뒤에 나열한다. 겹치는 상위/하위 디렉터리를 함께 지정해도 동일한 실제 이미지 경로는 한 번만 사용한다. 시작 로그에는 발견 이미지, 사용 가능한 image/text pair, `.txt` 누락, 빈/읽을 수 없는 prompt, 손상 이미지 수가 각각 출력된다. 이미지는 `--resolution` 크기로 short-side resize 후 center crop하며 기본값은 512다.

이 모드는 VAE와 Qwen text encoder를 학습 중 온라인 실행하므로 `cache_dir` 매칭은 필요하지 않지만, 기존 `latent.pt + embedding.pt` 캐시 모드보다 느리고 GPU 메모리를 더 사용한다. 기존 캐시가 있을 때는 `--image_dir` 대신 `--cache_dir`도 계속 사용할 수 있다.

### 4B 모델 LoRA 학습

```bash
CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 \
python my/mydisillation_lora.py \
  --image_dir /data1/jooyonglee/train-a /data1/jooyonglee/train-b \
  --validation_image_dir /data1/jooyonglee/validation \
  --resolution 512 \
  --output_dir ./output/flux2-4b-lora \
  --rank 32 \
  --lora_alpha 32 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --max_train_steps 10000 \
  --learning_rate 1e-4 \
  --gradient_checkpointing \
  --allow_tf32 \
  --checkpointing_steps 500 \
  --validation_epochs 1
```

### Distillation student LoRA 학습

`--transformer_model`에 저장된 student 디렉터리를 지정한다. scheduler와 validation VAE는 `--teacher_model`의 4B pipeline 구성을 사용한다.

```bash
CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 \
python my/mydisillation_lora.py \
  --image_dir /data1/jooyonglee/train-a /data1/jooyonglee/train-b \
  --transformer_model ./output/flux2-distill-1b/student \
  --output_dir ./output/flux2-distill-1b-lora \
  --rank 32 \
  --lora_alpha 32 \
  --train_batch_size 1 \
  --gradient_checkpointing \
  --max_train_steps 10000
```

학습 재개:

```bash
--resume_from_checkpoint latest
```

최종 LoRA는 Diffusers에서 바로 읽을 수 있는 형식으로 저장된다.

```text
output/flux2-4b-lora/
├── pytorch_lora_weights.safetensors
├── training_args.json
└── checkpoint-*/
```

추론 시에는 LoRA를 학습한 것과 동일한 구조의 transformer에 로드해야 한다.

```python
pipe.load_lora_weights("./output/flux2-4b-lora")
```

4B에서 학습한 LoRA를 축소된 1B student에 로드하거나, 반대로 1B student LoRA를 4B에 로드할 수는 없다.

## Checkpoint와 validation

두 스크립트 모두 Accelerate state를 저장하므로 checkpoint에는 모델 또는 LoRA뿐 아니라 optimizer, LR scheduler, RNG 상태가 포함된다.

```text
checkpoint-1000/
├── optimizer.bin
├── scheduler.bin
├── random_states_0.pkl
└── transformer/ 또는 pytorch_lora_weights.safetensors
```

`--validation_epochs`가 0보다 크면 고정 seed validation을 실행한다. raw-image LoRA는 학습 데이터와 분리된 `--validation_image_dir` 사용을 권장한다. 캐시 모드에서는 `--validation_cache_dir`를 사용한다.

Validation은 실제 `Flux2KleinPipeline` 추론과 같은 설정을 사용한다.

- 기본 inference steps: 50
- 기본 CFG guidance scale: 4.0
- distillation scheduler: 모델의 `FlowMatchEulerDiscreteScheduler`
- LoRA validation 기본 scheduler: `DPMSolverMultistepScheduler` (`flow_prediction`, flow sigmas, dynamic shifting, solver order 2)
- positive prompt embedding: validation 이미지의 `.txt`를 Qwen으로 인코딩하거나 validation cache의 `embedding.pt` 사용
- negative prompt embedding: 로컬 Qwen text encoder에서 empty prompt를 한 번 인코딩
- 기본 해상도: pipeline 기본값인 1024×1024

LoRA validation을 원래 scheduler와 비교하려면 `--validation_scheduler flow_match_euler`를 지정한다. DPM-Solver는 validation/추론에만 사용하며 학습의 flow-matching noise scheduler는 `FlowMatchEulerDiscreteScheduler`를 유지한다. optimizer learning-rate scheduler는 별개의 `--lr_scheduler` 옵션이다.

Qwen text encoder 로딩을 생략하려면 미리 저장한 empty prompt embedding을 `--validation_negative_prompt_embeds`로 전달할 수 있다. 해상도는 `--validation_height`와 `--validation_width`를 함께 지정해 변경한다.

결과는 TensorBoard와 PNG 파일에 동시에 저장된다.

```text
output/flux2-distill-1b/validation/
├── step-00000000/
│   ├── validation-teacher-000-seed-0.png
│   └── validation-teacher.json
└── step-00005000/
    ├── validation-student-000-seed-0.png
    └── validation-student.json
```

JSON에는 파일 이름, prompt, seed, guidance scale, inference steps와 이미지 크기가 기록된다.

```bash
tensorboard --logdir ./output/flux2-distill-1b/logs
```

## 메모리와 실행 방식

- 기본 dtype은 `bf16`이다.
- 한 번에 생성하거나 학습하는 실제 batch는 `--train_batch_size`로 결정된다.
- 메모리가 부족하면 batch size 1과 `--gradient_checkpointing`을 사용하고, effective batch는 `--gradient_accumulation_steps`로 늘린다.
- `CUDA_VISIBLE_DEVICES=3`을 사용한 경우 스크립트나 Accelerate 내부 GPU 번호는 `cuda:0`이다.
- 다중 GPU 실행은 `accelerate launch`를 사용할 수 있다.

```bash
CUDA_VISIBLE_DEVICES=2,3 HF_HUB_OFFLINE=1 \
accelerate launch --num_processes 2 my/mydisillation.py \
  --cache_dir /data1/jooyonglee/smiledesign_cache \
  --output_dir ./output/flux2-distill-1b \
  --gradient_checkpointing
```

## 검증 결과

현재 환경에서 다음 항목을 확인했다.

- Distillation teacher 3.876B에서 student 1.054B 생성
- 실제 FLUX.2 4B teacher와 캐시 데이터로 optimizer step 수행
- GPU 2·3 DDP에서 서로 다른 CommonCatalog parquet shard 사용 확인
- CommonCatalog 이미지→VAE latent, caption→Qwen embedding 온라인 변환 후 2-GPU optimizer step 수행
- CommonCatalog step 기반 teacher/student CFG validation 및 PNG 저장
- Distillation teacher/student validation을 CFG=4 실제 pipeline 경로로 실행하고 TensorBoard·PNG·JSON 저장
- LoRA 전체 transformer 3.880B 중 4,177,920개 파라미터만 학습
- LoRA 1-step 학습, checkpoint 저장, `latest` 재개 후 다음 step 수행
- LoRA가 적용된 `Flux2KleinPipeline` validation 이미지 생성
- 최종 LoRA 파일 크기 약 8.4 MB
- 모든 테스트에서 Hugging Face 다운로드 비활성 상태 확인

## 주의사항

- 캐시 latent normalization 또는 patchify가 틀리면 loss가 내려가더라도 결과 이미지가 깨질 수 있다.
- embedding 차원은 transformer의 `joint_attention_dim`과 일치해야 한다.
- validation은 반드시 고정 prompt, 고정 seed, 별도 validation cache를 사용해 checkpoint 간 품질을 비교하는 것이 좋다.
- `teacher_gt`가 계속 높고 student 결과도 동시에 무너지면 학습 횟수보다 캐시 포맷과 target 계산을 먼저 확인해야 한다.
- 모델 구조가 다른 LoRA끼리는 호환되지 않는다.
