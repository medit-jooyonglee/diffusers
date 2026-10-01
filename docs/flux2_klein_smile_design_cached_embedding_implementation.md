# FLUX.2 Klein Smile Design Cached Embedding 구현 및 사용 가이드

이 문서는 `docs/flux2_klein_smile_design_cached_embedding_product_plan.md`를 기반으로 구현한 기능과 실제 실행 방법을 정리한다.

## 1. 구현 목표

제품 추론 시 Qwen text encoder를 로드하지 않고, 사전에 계산한 `prompt_embeds`를 사용하여 FLUX.2 Klein으로 Smile Design 이미지를 생성한다.

```text
사전 준비
고정 prompt template + reference layout + style anchor
    -> Qwen text encoder
    -> prompt embedding 저장

제품 추론
환자/reference 이미지 + preset/slider 값
    -> 저장된 embedding 선택 및 보간
    -> FLUX.2 Klein에 prompt_embeds 전달
    -> 편집 이미지 생성
```

슬라이더가 변경하는 것은 FLUX 모델의 학습 가중치가 아니다. 슬라이더 값에 대응하는 cached prompt embedding을 선택하거나 보간하여 inference condition으로 전달한다.

## 2. 구현 파일

| 파일 | 역할 |
| --- | --- |
| `my/configs/smile_design/smile_v1.yaml` | prompt template, reference layout, slider anchor, preset, 2D grid 정의 |
| `my/smile_design/prompt_templates.py` | 설정 검증, 전체 prompt 생성, embedding entry 생성 |
| `my/smile_design/build_embedding_library.py` | Qwen을 이용한 embedding library 생성 및 증분 저장 |
| `my/smile_design/embedding_library.py` | embedding 로드, anchor 보간, grid 보간, delta 합성 |
| `my/smile_design/inference.py` | text encoder 없는 제품 추론 및 결과 기록 |
| `my/smile_design/verify_cached_embedding.py` | runtime encoding과 cached embedding의 parity 검증 |
| `my/build_smile_embedding_library.py` | embedding 생성 CLI 진입점 |
| `my/smile_design_inference.py` | 제품 추론 CLI 진입점 |
| `my/verify_smile_cached_embedding.py` | parity 검증 CLI 진입점 |
| `tests/my/test_smile_design.py` | prompt 생성과 embedding 보간 단위 테스트 |

## 3. Reference layout

reference 이미지는 역할과 순서를 고정한다. 전달된 이미지 조합으로 layout을 자동 판별할 수 있으며 `--layout`으로 명시할 수도 있다.

| Layout | 이미지 순서 |
| --- | --- |
| `patient_only` | patient |
| `patient_tooth` | patient, tooth |
| `patient_mouth` | patient, mouth |
| `patient_tooth_mouth` | patient, tooth, mouth |
| `patient_tooth_mouth_pose` | patient, tooth, mouth, pose |
| `patient_tooth_mouth_pose_extra` | patient, tooth, mouth, pose, extra |

각 layout은 독립된 전체 prompt를 사용한다. 예를 들어 `patient_tooth_mouth`는 image 1을 환자, image 2를 치아 형태, image 3을 입과 미소 스타일 reference로 사용하라는 지시를 embedding에 포함한다.

## 4. Slider와 preset

현재 정의된 slider는 다음과 같다.

| Slider | 범위 | 기본값 |
| --- | ---: | ---: |
| `whitening` | 0~100 | 0 |
| `smile_intensity` | 0~100 | 50 |
| `tooth_roundness` | 0~100 | 50 |
| `gum_exposure` | 0~100 | 25 |
| `mouth_opening` | 0~100 | 25 |

각 slider에는 `0, 25, 50, 75, 100` anchor가 있으며, anchor마다 숫자만 다른 것이 아니라 의미가 다른 자연어 문장이 정의되어 있다.

현재 preset은 다음과 같다.

- `natural`
- `mild_whitening`
- `strong_whitening`
- `subtle_smile`
- `broad_smile`
- `cosmetic_rounded`

## 5. Embedding 선택 및 보간 방식

### 5.1 Preset

사전에 계산한 preset embedding 하나를 그대로 사용한다.

```text
patient_tooth/preset/cosmetic_rounded
```

### 5.2 단일 anchor 보간

slider 값이 anchor 사이에 있으면 두 embedding을 선형 보간한다. 예를 들어 whitening이 63이면 50과 75 anchor를 사용한다.

```text
t = (63 - 50) / (75 - 50) = 0.52
E(63) = lerp(E(50), E(75), 0.52)
```

보간은 동일한 shape의 `[1, sequence_length, embedding_dim]` tensor에 대해 수행한다.

### 5.3 Whitening + Smile grid

제품용 권장 방식이다. `whitening_smile` grid는 다음 25개 전체 prompt embedding을 사전 계산한다.

```text
whitening       = 0, 25, 50, 75, 100
smile_intensity = 0, 25, 50, 75, 100
```

예를 들어 whitening 63, smile 82라면 다음 네 모서리를 선택한다.

```text
E(50, 75)  E(75, 75)
E(50,100)  E(75,100)
```

두 축에 대해 bilinear interpolation하여 `E(63,82)`를 만든다. 각 모서리는 Qwen이 whitening과 smile 문맥을 함께 본 전체 prompt이므로 두 속성의 상호작용을 delta 방식보다 안정적으로 반영할 수 있다.

### 5.4 Delta composition

각 속성의 embedding 방향을 base embedding에 더한다.

```text
delta_white = E(whitening) - E_base
delta_smile = E(smile) - E_base

E_final = E_base
        + alpha * delta_white
        + beta  * delta_smile
```

`--delta-strength whitening=0.8`과 같이 속성별 강도를 별도로 설정할 수 있다. 조합 수가 증가하지 않는 장점이 있지만 whitening과 smile 등 속성 간 간섭이 발생할 수 있으므로 현재는 실험 모드로 취급한다.

단순히 별도로 인코딩한 prompt sequence들을 concat하는 방식은 구현하지 않았다. 각 sequence가 다른 속성의 문맥을 보지 못하고 position/context 의미도 전체 prompt encoding과 달라지기 때문이다.

## 6. Embedding library 저장 형식

현재 생성한 smoke library는 다음 위치에 있다.

```text
/workspace/repo/mydiffusers/outputs/smile_design_embedding_smoke
```

```text
smile_design_embedding_smoke/
├── manifest.json
├── version.json
├── negative.pt
└── embeddings/
    ├── <prompt-sha256>.pt
    └── ...
```

- `manifest.json`: entry ID, prompt, layout, slider 값, tensor 파일, 모델 및 encoder metadata
- `version.json`: library schema/template version 정보
- `negative.pt`: empty/negative prompt embedding
- `embeddings/*.pt`: 실제 `prompt_embeds` tensor

embedding 파일명은 전체 prompt의 SHA256이다. 여러 entry가 동일한 prompt를 사용하면 동일 파일을 참조하므로 tensor를 중복 저장하지 않는다. 기존 파일은 기본적으로 다시 계산하지 않으며 `--overwrite`를 지정할 때만 덮어쓴다.

현재 smoke library 상태:

- logical entry 57개
- 중복 제거된 embedding tensor 38개
- tensor shape `[1, 512, 7680]`
- dtype `bf16`
- 전체 크기 약 533 MB

제품용 권장 저장 위치는 다음과 같다.

```text
/data1/jooyonglee/smiledesign/embedding_library/smile-v1
```

## 7. Embedding library 생성

로컬 distilled FLUX.2 Klein snapshot에는 완전한 Qwen weight가 없으므로, runtime 모델과 text encoder 모델을 분리하여 지정한다. 확인된 base와 distilled snapshot의 tokenizer 및 text encoder config는 동일하다.

```bash
env \
  CUDA_VISIBLE_DEVICES=1 \
  HF_HUB_OFFLINE=1 \
  PYTHONUNBUFFERED=1 \
  /opt/conda/envs/diffuser_lora/bin/python \
  my/build_smile_embedding_library.py \
    --model /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294 \
    --text-encoder-model /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-base-4B/snapshots/a3b4f4849157f664bdbc776fd7453c2783562f4d \
    --config my/configs/smile_design/smile_v1.yaml \
    --output-dir /data1/jooyonglee/smiledesign/embedding_library/smile-v1 \
    --batch-size 2 \
    --dtype bf16 \
    --device cuda:0 \
    --local-files-only
```

`--layout`을 생략하면 여섯 layout을 모두 생성한다. 일부만 생성하려면 옵션을 반복해서 지정한다.

```bash
--layout patient_only \
--layout patient_tooth
```

entry 종류도 필요한 것만 선택할 수 있다.

```bash
--kind base \
--kind anchor \
--kind grid
```

`--layout`과 `--kind`을 모두 생략하면 설정에 포함된 전체 library를 생성한다.

## 8. 제품 추론

### 8.1 Whitening + Smile 동시 제어

```bash
env \
  CUDA_VISIBLE_DEVICES=1 \
  HF_HUB_OFFLINE=1 \
  PYTHONUNBUFFERED=1 \
  /opt/conda/envs/diffuser_lora/bin/python \
  my/smile_design_inference.py \
    --model /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294 \
    --library outputs/smile_design_embedding_test \
    --patient my/samples/faces/samples/sample15.png \
    --tooth-reference my/samples/dentals/teeeth1.png \
    --mode delta \ \
    --grid whitening_smile \
    --whitening 63 \
    --smile-intensity 82 \
    --steps 4 \
    --guidance-scale 1.0 \
    --height 512 \
    --width 512 \
    --seed 42 \
    --dtype bf16 \
    --device cuda:0 \
    --local-files-only \
    --output-dir outputs/smile_design_gridmy

## base/anchors 생성 명령

env CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 /opt/conda/envs/diffuser_lora/bin/python my/build_smile_embedding_library.py --model /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294 --text-encoder-model /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-base-4B/snapshots/a3b4f4849157f664bdbc776fd7453c2783562f4d --config my/configs/smile_design/smile_v1.yaml --output-dir outputs/smile_design_embedding_basenachors --layout patient_tooth --kind base --kind anchor --batch-size 1 --dtype bf16 --device cuda:0 --local-files-only


env \
  CUDA_VISIBLE_DEVICES=1 \
  HF_HUB_OFFLINE=1 \
  PYTHONUNBUFFERED=1 \
  /opt/conda/envs/diffuser_lora/bin/python \
  my/smile_design_inference.py \
    --model /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294 \
    --library outputs/smile_design_embedding_smoke \
    --patient my/samples/faces/samples/sample15.png \
    --tooth-reference my/samples/dentals/teeeth1.png \
    --mode grid \
    --grid whitening_smile \
    --whitening 63 \
    --smile-intensity 100 \
    --steps 4 \
    --guidance-scale 1.0 \
    --height 736 \
    --width 512 \
    --seed 42 \
    --dtype bf16 \
    --device cuda:0 \
    --local-files-only \
    --output-dir outputs/smile_design_gridmy


env \
  CUDA_VISIBLE_DEVICES=1 \
  HF_HUB_OFFLINE=1 \
  PYTHONUNBUFFERED=1 \
  /opt/conda/envs/diffuser_lora/bin/python \
  my/smile_design_inference.py \
    --model /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294 \
    --library outputs/smile_design_embedding_basenachors \
    --patient my/samples/faces/samples/sample15.png \
    --tooth-reference my/samples/dentals/teeeth1.png \
    --mode delta \
    --whitening 63 \
    --smile-intensity 100 \
    --gum-exposure 5 \
    --mouth-opening 10 \
    --steps 4 \
    --guidance-scale 1.0 \
    --height 736 \
    --width 512 \
    --seed 42 \
    --dtype bf16 \
    --device cuda:0 \
    --local-files-only \
    --output-dir outputs/smile_design_deltamy2
```

`--layout`을 생략하면 전달한 reference 조합으로 `patient_tooth`를 자동 선택한다.

### 8.2 Preset

```bash
/opt/conda/envs/diffuser_lora/bin/python \
  my/smile_design_inference.py \
    --model /path/to/flux2-klein-4b \
    --library /data1/jooyonglee/smiledesign/embedding_library/smile-v1 \
    --patient /path/to/patient.png \
    --mode preset \
    --preset cosmetic_rounded \
    --seed 42 \
    --device cuda:0 \
    --local-files-only
```

### 8.3 단일 slider

```bash
--mode anchor \
--attribute tooth_roundness \
--value 63
```

### 8.4 Delta composition

```bash
--mode delta \
--whitening 63 \
--smile-intensity 82 \
--tooth-roundness 70 \
--delta-strength whitening=0.8 \
--delta-strength smile_intensity=0.6
```

### 8.5 Mask composite

입/치아 영역만 생성 결과를 적용하려면 흰색 영역이 편집 영역인 grayscale mask를 전달한다.

```bash
--mask /path/to/mouth_mask.png \
--mask-feather 12
```

생성 이미지에는 layout, 보간 방식, slider 값, seed, step, guidance scale, reference 정보와 `text_encoder_loaded=false`를 PNG metadata로 기록한다. 동일 정보와 VRAM 및 실행 시간은 출력 디렉터리의 `manifest_<run-id>.json`에도 저장한다.

## 9. 제품 추론 시 모델 로딩

제품 파이프라인은 다음과 같이 생성된다.

```text
Flux2KleinPipeline.from_pretrained(
    text_encoder=None,
    tokenizer=None,
    ...
)
```

따라서 Qwen text encoder와 tokenizer를 제품 GPU에 로드하지 않는다. 로컬 snapshot의 transformer가 Diffusers shard가 아니라 root 단일 `.safetensors`인 경우에도 자동으로 감지하여 `Flux2Transformer2DModel.from_single_file(...)`로 로드한다.

distilled 모델은 별도 지정이 없으면 4 step, guidance scale 1.0을 사용하고 base 모델은 50 step, guidance scale 4.0을 사용한다.

## 10. Cached embedding parity 검증

동일 prompt, reference, seed로 다음 두 경로를 비교한다.

1. 실행 시 Qwen으로 prompt encoding
2. library에서 cached embedding 로드

```bash
env \
  CUDA_VISIBLE_DEVICES=1 \
  HF_HUB_OFFLINE=1 \
  /opt/conda/envs/diffuser_lora/bin/python \
  my/verify_smile_cached_embedding.py \
    --model /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294 \
    --text-encoder-model /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-base-4B/snapshots/a3b4f4849157f664bdbc776fd7453c2783562f4d \
    --library /data1/jooyonglee/smiledesign/embedding_library/smile-v1 \
    --entry-id patient_only/base \
    --patient /path/to/patient.png \
    --steps 4 \
    --guidance-scale 1.0 \
    --height 512 \
    --width 512 \
    --seed 42 \
    --device cuda:0 \
    --local-files-only \
    --output-dir outputs/smile_design_parity
```

검증 결과에는 runtime/cached 이미지, 차이 강조 이미지, latent 및 pixel 오차 JSON이 저장된다.

실제 smoke 검증 결과:

```text
latent_max_abs  = 0.0
latent_mean_abs = 0.0
pixel_max_abs   = 0.0
pixel_mean_abs  = 0.0
```

즉 동일한 encoder 설정으로 만든 cached embedding은 runtime text encoding과 완전히 동일한 결과를 냈다.

## 11. 테스트 및 확인된 결과

단위 테스트:

```bash
/opt/conda/envs/diffuser_lora/bin/python \
  -m unittest -q tests.my.test_smile_design
```

검증 항목:

- 단일 anchor interpolation
- bilinear grid interpolation
- delta composition
- slider 범위 검증
- reference layout 자동 판별
- prompt entry 수와 config fingerprint

실제 smoke inference 결과:

- text encoder 없이 FLUX.2 Klein 4B 제품 추론 성공
- 256x256, 4 step 생성 약 2.22초
- peak CUDA allocated 약 8.91 GiB
- peak CUDA reserved 약 9.19 GiB
- grid 방식과 delta 방식의 whitening/smile 동시 제어 이미지 생성 성공

위 성능 수치는 현재 장비와 smoke 조건에서 측정한 값이며 512x512 제품 성능으로 일반화하면 안 된다.

## 12. 운영상 주의사항

1. library 생성과 inference에서 `--model` 문자열이 다르면 기본적으로 실행을 거부한다. 의도적으로 다른 호환 모델을 사용할 때만 `--allow-library-model-mismatch`를 사용한다.
2. `max_sequence_length`, `text_encoder_out_layers`, tokenizer 및 text encoder config가 달라지면 기존 embedding을 재사용하면 안 된다.
3. reference 이미지 순서가 바뀌면 prompt의 role 의미도 깨지므로 앱 API에서 순서를 고정해야 한다.
4. slider 보간은 embedding 공간의 선형성이 실제 시각적 선형성을 보장하지 않는다. 각 구간의 monotonicity와 identity preservation을 별도로 평가해야 한다.
5. whitening과 smile은 우선 2D grid를 제품 기본값으로 사용한다. delta composition은 속성 간 간섭 평가를 통과한 경우에만 제품에 적용한다.
6. 제품에서는 mouth/teeth segmentation mask를 함께 사용해 얼굴, 피부, 머리카락, 배경 변화 위험을 줄이는 것이 좋다.
7. `/opt/conda` 기본 환경은 현재 `torch`와 `torchaudio` ABI가 맞지 않아 import 오류가 발생한다. 명령은 검증된 `/opt/conda/envs/diffuser_lora/bin/python`으로 실행한다.

## 13. 다음 검증 항목

- 각 slider의 0/25/50/75/100 결과가 시각적으로 단조 변화하는지 확인
- whitening 변경 시 smile/입 모양이 변하는지 cross-attribute leakage 측정
- smile 변경 시 치아 색상과 환자 identity가 변하는지 측정
- grid와 delta를 동일 seed/reference로 비교
- patient-only와 multi-reference 간 identity preservation 비교
- 512x512 기준 latency, peak VRAM 및 동시 요청 처리량 측정
- mask 사용 전후의 비편집 영역 LPIPS/SSIM 또는 pixel 변화 측정
- 치과 전문가 기준으로 해부학적 오류, 잇몸 노출, 치아 개수 및 교합 artifact 평가

