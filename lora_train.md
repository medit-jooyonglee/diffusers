# SDXL LoRA Training Study Guide

## 0. 목표

이 문서는 SDXL 기반 LoRA 학습 파이프라인을 처음부터 직접 실행하고,
이후 LoRA merge 및 distillation까지 확장하기 위한 step-by-step 스터디 문서이다.

전체 흐름:

```text
격리 환경 구성
    ↓
Diffusers 공식 repository 설치
    ↓
SDXL 구조 확인
    ↓
학습 데이터 10장 준비
    ↓
Smoke Testc:\Users\medit\Downloads\lora_train.md
    ↓
SDXL LoRA 학습
    ↓
PEFT / LoRA 구조 확인
    ↓
LoRA safetensors 저장
    ↓
Inference
    ↓
LoRA Merge
    ↓
LCM / Distillation 학습
    ↓
FLUX.2 Klein 4B Base → LoRA → Distillation로 확장
```

## 1. 격리 환경 만들기

기존 FLUX 환경과 SDXL 환경은 분리하는 것을 권장한다.

```bash
conda create -n sdxl_lora python=3.11 -y
conda activate sdxl_lora
python -m pip install --upgrade pip
```

이유는 `transformers`, `huggingface-hub`, `safetensors`, `diffusers`, `peft` 등의 버전 충돌을 피하기 위해서다.

## 2. PyTorch 설치

```bash
pip install torch torchvision
```

GPU 확인:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 3. Diffusers 공식 Repository

공식 repo:

```text
https://github.com/huggingface/diffusers
```

설치:

```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e .
```

SDXL 예제 의존성:

```bash
cd examples/text_to_image
pip install -r requirements_sdxl.txt
pip install peft accelerate transformers datasets safetensors
```

선택:

```bash
pip install xformers
pip install bitsandbytes
```

설치 상태 확인:

```bash
pip check
```

## 4. 주요 라이브러리 역할

```text
PyTorch
 └─ tensor / autograd / optimizer / GPU 학습

Diffusers
 ├─ SDXL Pipeline
 ├─ UNet
 ├─ VAE
 ├─ Scheduler
 └─ diffusion training logic

Transformers
 ├─ tokenizer
 └─ CLIP text encoder

PEFT
 └─ LoRA adapter

Accelerate
 ├─ mixed precision
 ├─ multi-GPU
 └─ distributed training

Datasets
 └─ ImageFolder / metadata.jsonl

Safetensors
 └─ 모델/LoRA tensor weight 저장
```

## 5. 공식 SDXL 학습 코드 위치

```text
diffusers/
└── examples/
    ├── text_to_image/
    │   ├── train_text_to_image_sdxl.py
    │   └── train_text_to_image_lora_sdxl.py
    ├── dreambooth/
    │   └── train_dreambooth_lora_sdxl.py
    └── consistency_distillation/
        ├── train_lcm_distill_sdxl_wds.py
        └── train_lcm_distill_lora_sdxl_wds.py
```

가장 먼저 볼 파일:

```text
examples/text_to_image/train_text_to_image_lora_sdxl.py
```

확인:

```bash
cd diffusers/examples/text_to_image
ls train_text_to_image_lora_sdxl.py
python train_text_to_image_lora_sdxl.py --help
```

## 6. SDXL 구조

```text
Prompt
 │
 ├─ Tokenizer 1 → Text Encoder 1
 │
 └─ Tokenizer 2 → Text Encoder 2
                  ↓
            Prompt Embedding

Image
 ↓
VAE
 ↓
Latent

Latent + Noise + Prompt Embedding
 ↓
UNet
 ↓
Noise Prediction
```

주요 구성:

```text
VAE
2개의 text encoder
UNet (~2.6B parameter 규모)
Scheduler
```

## 7. LoRA 개념

기존 weight:

```text
W
```

LoRA:

```text
W' = W + ΔW
ΔW = B @ A
```

원본 weight는 freeze하고 low-rank A/B만 학습한다.

## 8. PEFT

PEFT = Parameter-Efficient Fine-Tuning.

```python
from peft import LoraConfig
```

예:

```python
unet_lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
)

unet.add_adapter(unet_lora_config)
```

## 9. Trainable Parameter 확인

```python
for name, param in unet.named_parameters():
    if param.requires_grad:
        print(name, param.shape)
```

전체 대비 비율:

```python
trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
total = sum(p.numel() for p in unet.parameters())

print("trainable:", trainable)
print("total:", total)
print("ratio:", trainable / total)
```

## 10. safetensors란?

`safetensors`는 tensor weight 저장 포맷이다.

예:

```text
pytorch_lora_weights.safetensors
```

내부 개념:

```text
{
  "tensor_name_1": Tensor,
  "tensor_name_2": Tensor
}
```

모델 아키텍처가 아니라 실제 weight tensor를 저장한다.

## 11. Smoke Test란?

Smoke test는 전체 pipeline이 최소한 정상 실행되는지 확인하는 테스트다.

```text
이미지 10장
 ↓
Dataset load
 ↓
SDXL load
 ↓
LoRA adapter 생성
 ↓
Forward
 ↓
Loss
 ↓
Backward
 ↓
20 steps
 ↓
LoRA safetensors 저장
```

품질 평가는 하지 않는다.

## 12. 최소 Dataset 준비

```text
test_dataset/
├── 0001.jpg
├── 0002.jpg
├── ...
├── 0010.jpg
└── metadata.jsonl
```

`metadata.jsonl` 예:

```json
{"file_name":"0001.jpg","text":"jy_test_style, woman, black dress, standing indoors"}
{"file_name":"0002.jpg","text":"jy_test_style, woman, white dress, sitting on sofa"}
{"file_name":"0003.jpg","text":"jy_test_style, portrait, long hair, studio lighting"}
{"file_name":"0004.jpg","text":"jy_test_style, full body portrait, standing indoors"}
{"file_name":"0005.jpg","text":"jy_test_style, woman, red dress, soft lighting"}
{"file_name":"0006.jpg","text":"jy_test_style, portrait, looking at viewer"}
{"file_name":"0007.jpg","text":"jy_test_style, woman, sitting indoors"}
{"file_name":"0008.jpg","text":"jy_test_style, fashion portrait, dark background"}
{"file_name":"0009.jpg","text":"jy_test_style, woman, standing in studio"}
{"file_name":"0010.jpg","text":"jy_test_style, portrait, soft directional light"}
```

## 13. Dataset 확인

```python
from datasets import load_dataset

dataset = load_dataset(
    "imagefolder",
    data_dir="./test_dataset"
)

print(dataset)
print(dataset["train"][0])
```

정상 예:

```text
DatasetDict({
    train: Dataset({
        features: ['image', 'text'],
        num_rows: 10
    })
})
```

## 14. Accelerate 초기 설정

```bash
accelerate config default
```

## 15. 최소 Smoke Test

```bash
accelerate launch train_text_to_image_lora_sdxl.py   --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"   --train_data_dir="./test_dataset"   --resolution=512   --train_batch_size=1   --learning_rate=1e-4   --rank=8   --max_train_steps=20   --mixed_precision="fp16"   --output_dir="./smoke_test"
```

항상 먼저:

```bash
python train_text_to_image_lora_sdxl.py --help
```

로 현재 버전의 CLI를 확인한다.

## 16. 2차 동작 테스트

```bash
accelerate launch train_text_to_image_lora_sdxl.py   --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"   --train_data_dir="./test_dataset"   --image_column="image"   --caption_column="text"   --resolution=1024   --train_batch_size=1   --gradient_accumulation_steps=1   --learning_rate=1e-4   --rank=16   --max_train_steps=200   --checkpointing_steps=100   --mixed_precision="fp16"   --output_dir="./sdxl_lora_test"   --seed=42
```

## 17. A6000 초기 권장 설정

```text
resolution = 1024
rank = 16
lora_alpha = 16
learning_rate = 1e-4
batch_size = 2~4
mixed_precision = fp16 or bf16
gradient_checkpointing = OFF부터 시작
steps = 500 / 1000 / 2000
checkpoint = 500 step 간격
```

## 18. Caption 형식

SDXL LoRA는 tag caption으로도 잘 학습된다.

예:

```text
jy_style, woman, long hair, black dress,
standing, looking at viewer, bedroom, full body
```

따라서 WD-EVA02 / WD Tagger 출력도 사용 가능하다.

## 19. Training Loop 이해

공식 코드에서 다음 순서로 읽는다.

```text
1. LoraConfig
2. add_adapter
3. params_to_optimize
4. optimizer
5. encode_prompt
6. noise_scheduler
7. vae.encode
8. unet(...)
9. loss
10. accelerator.backward
11. save_lora_weights
```

핵심:

```text
Image → VAE.encode → Latent
Latent + Noise → Noisy Latent
Caption → Text Encoder → Embedding
Noisy Latent + Embedding → UNet
UNet Output vs Target Noise → Loss
Loss → Backward
LoRA A/B만 update
```

## 20. LoRA 학습 결과

예:

```text
sdxl_lora_test/
├── checkpoint-100/
├── checkpoint-200/
└── pytorch_lora_weights.safetensors
```

## 21. LoRA Inference

```python
import torch
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

pipe.load_lora_weights("./sdxl_lora_test")

image = pipe(
    "jy_test_style, a woman standing in a studio",
    num_inference_steps=25
).images[0]

image.save("test.png")
```

## 22. LoRA Merge

개념:

```text
Wfinal = Wbase + α × BA
```

예:

```python
pipe.load_lora_weights("./sdxl_lora_test")
pipe.fuse_lora()
```

중요:

```text
LoRA Merge != Distillation
```

## 23. Distillation 공식 코드

위치:

```text
examples/consistency_distillation/
```

주요 파일:

```text
train_lcm_distill_sdxl_wds.py
train_lcm_distill_lora_sdxl_wds.py
```

먼저 추천:

```text
train_lcm_distill_lora_sdxl_wds.py
```

## 24. LCM Distillation 개념

일반 diffusion:

```text
x_t → x_(t-1) → x_(t-2) → ... → image
```

Distilled student:

```text
x_t ─────────→ x_(t-k)
```

즉 더 적은 inference step으로 비슷한 결과를 만드는 것이 목적이다.

## 25. Full Distillation vs LCM-LoRA

Full:

```text
SDXL Teacher
 ↓
Consistency Distillation
 ↓
Student Model
 ↓
Few-step inference
```

LCM-LoRA:

```text
SDXL Base
 +
LCM-LoRA
 ↓
Few-step inference
```

## 26. SDXL → FLUX.2 연결

```text
SDXL
 ↓
LoRA pipeline 검증
 ↓
FLUX.2 Klein 4B Base
 ↓
Domain LoRA
 ↓
Fine-tuned Teacher
 ↓
Distillation
 ↓
Fast Production Model
```

## 27. 전체 스터디 순서

```text
1. sdxl_lora Conda 환경 생성
2. Diffusers clone/install
3. train_text_to_image_lora_sdxl.py --help
4. 10장 dataset + metadata.jsonl
5. load_dataset() 확인
6. 512px / 20-step smoke test
7. 1024px / 200-step 테스트
8. safetensors 생성 확인
9. LoRA inference
10. Base vs LoRA 비교
11. LoraConfig / add_adapter 분석
12. trainable parameter 출력
13. 500~2000장으로 확장
14. 500 / 1000 / 2000 step 비교
15. LoRA merge
16. LCM-LoRA distillation 코드 분석
17. SDXL distillation 실행
18. FLUX.2 Klein 4B Base로 개념 이전
```

## 28. 핵심 용어 요약

### LoRA

전체 모델이 아니라 low-rank adapter만 학습.

### PEFT

LoRA를 포함한 Parameter-Efficient Fine-Tuning 라이브러리.

### safetensors

모델 구조가 아니라 tensor weight 저장 포맷.

### Smoke Test

품질이 아니라 pipeline이 안 깨지고 실행되는지 확인.

### Distillation

Teacher의 동작을 Student가 다시 학습하는 과정.

## 29. 다음 스터디

`train_text_to_image_lora_sdxl.py`를 다음 10개 블록으로 나눠 읽는다.

```text
1. Model Loading
2. Freeze
3. LoraConfig
4. Dataset
5. Prompt Encoding
6. VAE / Latent
7. Noise Scheduler
8. UNet Forward
9. Loss / Backward
10. LoRA Save
```

이 구조를 이해하면 이후 LCM distillation과 FLUX.2 LoRA 코드도 훨씬 쉽게 읽을 수 있다.
