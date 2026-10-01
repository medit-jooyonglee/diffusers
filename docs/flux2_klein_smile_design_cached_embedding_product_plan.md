# FLUX.2 Klein 4B 기반 Smile Design 제품화 실험 계획
## Precomputed Text Embedding + Multi-Reference + Slider Control

작성 목적:
- 모델 경량화는 후순위로 두고 **FLUX.2 Klein 4B pretrained**를 우선 사용
- 제품 추론 시 **Qwen3 계열 text encoder를 제거**
- 자주 사용하는 치아/미소 편집 명령을 사전에 text embedding으로 계산 및 저장
- 사용자 입력은 환자 이미지 + 1~4개의 reference image + slider/프리셋
- 치아 스타일, whitening, restoration, smile intensity, mouth/lip style, pose/view 등을 제어
- 제품화 가능성을 단계별 실험으로 검증

---

# 1. 핵심 아이디어

개발/사전 준비 단계에서는:

```text
Text Prompt
    ↓
Qwen3 Text Encoder
    ↓
Prompt Embedding
    ↓
파일로 저장
```

제품 추론 단계에서는:

```text
환자 이미지
+ Reference image 1~4장
+ Slider / Preset ID
        ↓
미리 저장된 prompt_embeds 로드
        ↓
FLUX.2 Klein 4B
        ↓
Edited Smile Image
```

즉 제품에서는:

```text
Qwen3 text encoder
= 로드하지 않음
```

을 목표로 한다.

Diffusers의 `Flux2KleinPipeline`은 `prompt` 대신 `prompt_embeds`를 직접 받을 수 있으므로 이 구조는 기술적으로 가능하다.

---

# 2. 왜 VRAM 절약 효과가 큰가

FLUX.2 Klein 4B 공식 Diffusers checkpoint의 text encoder 디렉터리는 약 8.05 GB이고 Qwen3 계열 encoder를 사용한다.

따라서 text embedding을 사전에 생성하고 제품에서 text encoder를 로드하지 않으면:

```text
[기존]

Qwen3 Text Encoder
+
FLUX.2 Transformer
+
VAE
+
Reference latents
+
KV cache
+
Denoising working memory
```

에서

```text
[제품]

FLUX.2 Transformer
+
VAE
+
Reference latents
+
KV cache
+
Denoising working memory
```

구조로 단순화할 수 있다.

주의:
- 실제 peak VRAM 절감량은 dtype, offload, quantization, CUDA allocator에 따라 달라진다.
- "8.05 GB 파일 크기 = 정확히 8.05 GB peak VRAM 절감"으로 단정해서는 안 된다.
- 하지만 text encoder 자체를 제품에서 제거할 수 있다는 점은 매우 큰 메모리/초기화 시간 절감 요인이다.

---

# 3. 제품의 기본 입력 구조

가장 단순한 제품 입력은 다음과 같이 고정한다.

```text
Image 0 : 환자 원본 이미지 (필수)

Image 1 : Tooth Style Reference (선택)
Image 2 : Mouth / Lip / Smile Style Reference (선택)
Image 3 : Pose / View Reference (선택)
```

또는 최대 4장의 reference를 허용한다면:

```text
Image 0 : Patient
Image 1 : Tooth Shape / Tooth Style
Image 2 : Smile / Mouth / Lip Style
Image 3 : Pose / View
Image 4 : Extra Dental Style / Restoration Reference
```

중요:
**reference 순서는 항상 고정한다.**

예:

```text
ref[0] = patient
ref[1] = tooth style
ref[2] = mouth style
ref[3] = pose
```

처럼 제품 API 수준에서 의미를 고정해야 한다.

---

# 4. 왜 reference 역할을 고정해야 하는가

FLUX.2는 여러 reference image를 받을 수 있지만 모델이 자동으로:

```text
이 이미지는 치아
이 이미지는 입술
이 이미지는 pose
```

라고 API 차원에서 role ID를 받는 구조는 아니다.

따라서 역할은 주로 prompt semantic으로 전달된다.

예:

```text
Use image 1 as the original patient.
Use image 2 only as a reference for tooth shape and tooth proportions.
Use image 3 only as a reference for mouth and smile style.
Preserve the patient's identity, face shape, skin, hair, background, and head pose.
```

제품에서 text encoder를 제거한다면 위와 같은 role instruction도
**사전에 embedding으로 만들어야 한다.**

즉 reference 조합별로 prompt preset이 필요하다.

---

# 5. Reference configuration preset

예:

## P0 - Patient only

```text
Image 0 = Patient
```

Embedding:

```text
preset_patient_only.pt
```

---

## P1 - Patient + Tooth Style

```text
Image 0 = Patient
Image 1 = Tooth Style
```

Prompt 의미:

```text
Preserve image 1 as the patient.
Use image 2 only as reference for tooth morphology,
tooth proportions, incisal edge shape, and tooth arrangement.
Preserve the patient's face, lips, identity, skin, hair,
background and head pose.
```

---

## P2 - Patient + Tooth + Mouth Style

```text
Image 0 = Patient
Image 1 = Tooth Style
Image 2 = Mouth/Smile Style
```

---

## P3 - Patient + Tooth + Mouth + Pose

```text
Image 0 = Patient
Image 1 = Tooth Style
Image 2 = Mouth/Smile Style
Image 3 = Pose/View
```

이렇게 reference role 조합마다 base embedding을 미리 생성한다.

---

# 6. 제어 항목 분류

모든 치아 편집을 diffusion slider 하나로 처리하려 하면 제품 안정성이 떨어진다.

제어 항목을 세 종류로 나누는 것이 좋다.

---

## A. FLUX text/reference control에 적합한 항목

### Tooth Style
- 자연스러운 치아
- cosmetic / veneer-like
- rounded tooth
- squared tooth
- soft incisal edge
- sharp incisal edge
- broad anterior teeth
- narrow anterior teeth
- long / short crown appearance

### Smile
- smile intensity
- mouth opening
- teeth exposure
- gum exposure
- smile arc
- broad / narrow smile

### Appearance
- natural whitening
- mild whitening
- strong whitening
- translucency appearance
- glossy / natural enamel appearance

### General restoration appearance
- natural restored tooth appearance
- chipped edge restoration appearance
- uniform anterior tooth appearance

이런 항목은 semantic control로 비교적 자연스럽다.

---

## B. FLUX 단독 제어는 가능하지만 주의가 필요한 항목

### Pose / View
- slight left yaw
- slight right yaw
- slight up/down camera/view
- 3/4 view

문제:
환자 identity와 geometry가 같이 바뀔 수 있다.

따라서 pose는 text slider보다:

```text
pose reference
+
landmark/geometry
```

기반 제어가 더 안정적일 수 있다.

---

### Lip / Mouth Shape
- lip opening
- broad smile
- narrow smile
- corner elevation
- mouth width

이것도 생성형 모델이 얼굴 전체를 수정할 수 있으므로
ROI 또는 mask 기반 보존 실험이 필요하다.

---

## C. FLUX에 맡기지 않는 것이 좋은 항목

### 특정 치아 1개만 whitening
### 특정 치아의 정확한 색상 수치 변경
### 정확한 FDI tooth instance 이동
### 정확한 mm 단위 crown 길이 조정
### 교정 시뮬레이션처럼 위치를 정확히 제어
### 특정 치아만 형태 변경하면서 주변 치아 완전 고정

이런 항목은:

```text
instance mask
3D tooth library
render/composite
color processing
geometry transform
```

으로 먼저 처리하고,

FLUX는:

```text
harmonization
realism
local restoration
```

에 사용한다.

---

# 7. 추천 제품 역할 분리

```text
[Deterministic Dental Control]

3D Tooth Library
Tooth Instance Mask
Color Transform
Geometry
Registration
Rendering
        ↓
정확한 치아 형상 / 위치 / 색상 제어
        ↓
[FLUX.2 Klein]
        ↓
Photorealistic Harmonization
Smile / Lip / Lighting adaptation
```

즉:

> **FLUX를 CAD처럼 쓰지 않고 최종 시각적 harmonizer/editor로 사용**

하는 것이 제품적으로 더 안정적이다.

---

# 8. Slider 설계

제품 UI 예:

```text
Whitening        [0 ---------- 100]
Smile Intensity  [0 ---------- 100]
Tooth Length     [0 ---------- 100]
Tooth Width      [0 ---------- 100]
Roundness        [0 ---------- 100]
Gum Exposure     [0 ---------- 100]
Mouth Opening    [0 ---------- 100]
Natural <-------> Cosmetic
```

하지만 이 slider 값을 바로 embedding scalar multiplication으로 넣는 것은
첫 제품 방식으로 권장하지 않는다.

---

# 9. 가장 안전한 Slider Embedding 방식: Anchor Embeddings

예: Whitening.

사전에 prompt를 5단계 만든다.

```text
W0 = original natural tooth color
W1 = very subtle whitening
W2 = mild natural whitening
W3 = clear cosmetic whitening
W4 = strong bright whitening
```

Qwen3으로 각각:

```text
E_W0
E_W1
E_W2
E_W3
E_W4
```

를 계산하여 저장한다.

제품 slider가 0~100이면:

```text
0       -> E_W0
25      -> E_W1
50      -> E_W2
75      -> E_W3
100     -> E_W4
```

우선 이 discrete 단계부터 검증한다.

---

# 10. 연속 slider: 인접 embedding interpolation 실험

Anchor가 안정적으로 동작하면:

```text
slider = 0.35

E = (1-a) E_W1 + a E_W2
```

처럼 인접 anchor를 interpolation한다.

예:

```text
Whitening 25~50 구간

a = (slider - 25) / 25

E = (1-a) * E_W1
  + a     * E_W2
```

장점:
- runtime Qwen 불필요
- continuous UI 가능
- 구현 간단

하지만:

> Qwen hidden embedding 공간이 모든 semantic attribute에 대해 선형이라는 보장은 없다.

따라서 반드시 이미지 결과로 monotonicity를 확인해야 한다.

---

# 11. Delta embedding 방식

조금 더 공격적인 실험:

```text
E_base = embedding(
    "Preserve the patient and create a natural smile."
)

E_white = embedding(
    "Preserve the patient and create a smile with bright white teeth."
)

Delta_white = E_white - E_base
```

제품에서는:

```text
E = E_base + alpha * Delta_white
```

형태로 사용한다.

여러 control이면:

```text
E =
E_base
+ a * Delta_whitening
+ b * Delta_smile
+ c * Delta_roundness
+ d * Delta_gum
```

이 방식이 잘 동작한다면 매우 강력하다.

하지만 이것은 **실험 항목**이다.

문제:
- Qwen embedding은 contextual sequence representation
- token 위치가 달라질 수 있음
- attribute 간 interaction이 강할 수 있음
- delta vector가 완전히 독립적이지 않을 수 있음

따라서 처음부터 제품 구조로 확정하지 않는다.

---

# 12. Slider 전략 추천 우선순위

```text
Phase 1
Discrete prompt preset
★★★★★ 안정성

Phase 2
Adjacent anchor interpolation
★★★★☆

Phase 3
Embedding delta arithmetic
★★★☆☆

Phase 4
여러 delta 동시 조합
★★☆☆☆ 초기에는 위험
```

---

# 13. Text embedding을 실제로 저장하는 방법

개발 PC:

```python
with torch.no_grad():
    prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt,
        device="cuda",
    )

torch.save(
    {
        "prompt_embeds": prompt_embeds.cpu(),
        "text_ids": text_ids.cpu(),
    },
    "preset_whitening_50.pt",
)
```

실제 Diffusers 버전에 따라 `encode_prompt` 반환 형식은 확인해야 한다.

FLUX.2 Klein Diffusers 구현은 기본적으로:

```text
prompt_embeds
text_ids
```

를 생성한다.

---

# 14. 제품 추론

제품 실행에서는 Qwen tokenizer/text encoder를 로드하지 않는다.

개념 코드:

```python
preset = torch.load(
    "preset_whitening_50.pt",
    map_location="cpu",
)

prompt_embeds = preset["prompt_embeds"].to(
    device="cuda",
    dtype=torch.bfloat16,
)
```

그리고:

```python
result = pipe(
    image=[
        patient_image,
        tooth_reference,
        mouth_reference,
    ],
    prompt=None,
    prompt_embeds=prompt_embeds,
    num_inference_steps=4,
    generator=generator,
).images[0]
```

현재 Diffusers의 `Flux2KleinPipeline`은 prompt가 없으면 `prompt_embeds`를 직접 받을 수 있다.

주의:
실제 사용 중인 Diffusers commit/version의 pipeline signature에 맞춰
`text_ids` 또는 내부 생성 경로가 필요한지 반드시 확인한다.

---

# 15. Text Encoder 제거 로딩

제품 pipeline 구성은 가능하면:

```text
tokenizer = None
text_encoder = None
```

으로 로드하거나,

```text
Transformer
VAE
Scheduler
```

만 직접 구성하는 형태를 목표로 한다.

중요 실험:

```text
A. 전체 pipeline + Qwen
B. prompt_embeds + Qwen GPU offload
C. Qwen 완전 미로딩
```

3개에서 출력이 동일한지 확인한다.

동일 seed / 동일 reference / 동일 embedding에서 pixel 또는 latent 수준까지 비교한다.

---

# 16. Embedding 저장 포맷

권장:

```text
embeddings/
  version.json

  patient_only/
    natural.pt
    whitening_0.pt
    whitening_25.pt
    whitening_50.pt
    whitening_75.pt
    whitening_100.pt

  patient_tooth_ref/
    natural.pt
    whitening_*.pt

  patient_tooth_mouth_ref/
    ...

  patient_tooth_mouth_pose_ref/
    ...
```

Metadata:

```json
{
  "model": "FLUX.2-klein-4B",
  "text_encoder": "Qwen3",
  "diffusers_version": "...",
  "max_sequence_length": 512,
  "text_encoder_out_layers": [9, 18, 27],
  "prompt_template_version": "smile-v1",
  "dtype": "bf16"
}
```

이 metadata가 중요하다.

Text encoder / prompt extraction 방식이 바뀌면
기존 embedding과 호환되지 않을 수 있기 때문이다.

---

# 17. 반드시 Prompt Template versioning

예:

```text
smile-v1
smile-v2
smile-v3
```

각 버전에:

```text
Prompt text
Embedding
Model hash
Diffusers version
Date
```

를 저장한다.

제품에서는 prompt 문자열 자체는 쓰지 않더라도
QA 및 재현을 위해 반드시 원문을 함께 관리한다.

---

# 18. 주요 Dental Preset 제안

## Tooth Shape

```text
Natural
Rounded
Square
Tapered
Broad
Narrow
Long
Short
```

---

## Incisal Edge

```text
Soft
Rounded
Flat
Sharp
Natural irregularity
```

---

## Whitening

```text
Original
Very subtle
Mild
Moderate
Bright cosmetic
Strong
```

---

## Smile

```text
Neutral
Subtle smile
Natural smile
Broad smile
Strong smile
```

---

## Teeth Exposure

```text
Low
Medium
High
```

---

## Gum Exposure

```text
Minimal
Natural
Moderate
High
```

---

## Mouth Opening

```text
Closed
Slightly open
Natural
Open
```

---

## Style

```text
Natural
Natural aesthetic
Cosmetic
Veneer-like
Hollywood-style
```

제품에서는 의학적 결과로 오해되지 않도록
UI 명칭과 설명을 임상팀과 검토할 필요가 있다.

---

# 19. Reference Image Library

제품에 사용자가 reference를 직접 넣게 할 수도 있지만,
내장 reference library를 제공하는 편이 재현성이 좋다.

예:

```text
Tooth Library
├─ Natural A
├─ Natural B
├─ Rounded
├─ Square
├─ Broad
├─ Narrow
├─ Long
└─ Cosmetic
```

각 reference는:

```text
고정 crop
고정 해상도
고정 배경
유사 lighting
표준 frontal view
```

로 normalize한다.

이렇게 해야 reference 효과를 비교하기 쉽다.

---

# 20. Pose reference 설계

Pose를 단순히 좌/우/상/하 text만으로 제어하기보다는
reference image를 준비할 수 있다.

예:

```text
frontal
yaw_left_15
yaw_left_30
yaw_right_15
yaw_right_30
pitch_up_10
pitch_down_10
```

하지만 Smile Design에서는:

> 환자 원래 pose를 보존하는 것이 일반적으로 더 중요

할 가능성이 높다.

따라서 제품 기본값:

```text
Preserve original pose
```

로 하고,
pose change는 별도 advanced 기능으로 실험하는 것을 권장한다.

---

# 21. 가장 중요한 제품 편집 원칙

## Preserve First

모든 prompt preset의 기본 문장은:

```text
Preserve:
- identity
- head pose
- face geometry
- eyes
- nose
- skin
- hair
- background

Modify only:
- teeth
- local mouth/smile attributes requested
```

를 기본으로 한다.

---

# 22. 하지만 Prompt만 믿으면 안 됨

생성형 모델은:

```text
"do not change..."
```

라고 해도 주변 영역을 변경할 수 있다.

따라서 제품에서는:

```text
원본
+
FLUX 결과
+
mouth/tooth mask
```

를 이용한 compositing 실험을 병행한다.

예:

```text
Final =
Original * (1 - M)
+
Generated * M
```

경계:

```text
feather
Poisson/Laplacian blend
local color harmonization
```

사용.

---

# 23. 정확한 단일 치아 제어

앞서 실험한:

```text
한 치아 whitening
→ 전체 치아 변화
```

문제를 고려하면:

```text
FDI tooth mask
        ↓
deterministic recolor / render
        ↓
FLUX local harmonization
```

방식으로 처리한다.

즉:

```text
Exact geometry/color
= deterministic

Natural appearance
= FLUX
```

역할 분리가 바람직하다.

---

# 24. 제품화 실험 계획

## Phase 0 - Baseline 고정

목적:
FLUX.2 Klein 4B 자체 성능 확인.

고정:

```text
Model
Diffusers version
Resolution
Inference steps
Seed set
Reference preprocessing
```

Validation set:

```text
50~100 patient images
```

다양성:

```text
frontal / side
male / female
different skin
different smile
different tooth exposure
different lighting
```

---

# 25. Phase 1 - Qwen 제거 가능성 검증

같은 prompt에서:

```text
A.
runtime Qwen prompt encoding

B.
cached prompt_embeds
```

비교.

조건:

```text
same seed
same patient
same references
same inference params
```

확인:

```text
latent difference
pixel difference
visual output
runtime
peak VRAM
startup memory
```

목표:

```text
A와 B 결과가 사실상 동일
```

이면 Qwen 제거 경로 확정.

---

# 26. Phase 2 - 기본 Dental Preset 검증

우선 reference 없이 text preset만 검증한다.

```text
Natural
Whitening
Smile intensity
Gum exposure
Mouth opening
Tooth style
```

각 항목 3~5단계.

평가:

```text
Control success
Identity preservation
Non-target change
Visual realism
Consistency
```

---

# 27. Phase 3 - Slider monotonicity

예: Whitening.

```text
0
25
50
75
100
```

결과가 실제로:

```text
L* / LAB brightness
tooth whiteness index
```

등에서 단조 증가하는지 측정한다.

단순 주관 평가만 하지 않는다.

Smile intensity도:

```text
lip corner distance
mouth width
teeth exposure ratio
```

같은 landmark metric으로 평가할 수 있다.

---

# 28. Phase 4 - Embedding interpolation

Discrete preset이 성공한 항목만:

```text
anchor interpolation
```

을 활성화한다.

예:

```text
37%
62%
88%
```

같은 중간값을 생성해:

```text
monotonic
smooth
artifact-free
```

인지 확인한다.

---

# 29. Phase 5 - Reference image 실험

각 reference 역할을 독립적으로 추가한다.

### Exp A

```text
Patient only
```

### Exp B

```text
Patient + Tooth Style
```

### Exp C

```text
Patient + Mouth Style
```

### Exp D

```text
Patient + Tooth + Mouth
```

### Exp E

```text
Patient + Tooth + Mouth + Pose
```

평가:

```text
reference similarity
patient identity preservation
cross-reference leakage
non-target changes
```

---

# 30. Reference leakage 평가

예:

```text
Tooth reference의 얼굴 특징이 환자로 복사됨
Mouth reference의 피부색이 복사됨
Pose reference의 identity가 섞임
```

이런 문제가 생길 수 있다.

따라서:

```text
Tooth Reference
→ tooth morphology만 전달됐는가?

Mouth Reference
→ lip/smile 특성만 전달됐는가?

Pose Reference
→ pose만 전달됐는가?
```

를 분리 평가한다.

---

# 31. Phase 6 - Multi-slider interaction

한 slider씩 성공한 뒤:

```text
Whitening + Smile
Whitening + Roundness
Smile + Gum exposure
Tooth style + Whitening
```

2개 조합부터 테스트한다.

바로:

```text
7~8 sliders simultaneously
```

실험하지 않는다.

Interaction matrix:

```text
              White Smile Round Gum
White           -     O     O    O
Smile           O     -     O    X
Round           O     O     -    O
Gum             O     X     O    -
```

형태로 interference를 기록한다.

---

# 32. Phase 7 - Delta Embedding 연구

Anchor interpolation까지 성공한 후:

```text
Delta_white
Delta_smile
Delta_round
...
```

를 계산한다.

검증:

```text
E_base + αΔ
```

가 실제로 원하는 속성만 변화시키는가?

그 다음:

```text
E_base
+ αΔwhite
+ βΔsmile
```

같은 조합을 실험한다.

---

# 33. 평가 지표

## A. Identity Preservation

```text
Face embedding similarity
Landmark displacement
Non-mouth LPIPS/SSIM
```

---

## B. Target Control

Whitening:

```text
tooth LAB
whiteness index
```

Smile:

```text
mouth width
lip corner height
teeth exposure
```

Gum:

```text
gingiva visible ratio
```

Tooth shape:

```text
mask / landmark morphology
```

---

## C. Non-target Preservation

```text
outside-mouth pixel difference
face shape change
eye/nose change
background difference
```

---

## D. Human Evaluation

```text
Realism
Dental plausibility
Identity
Control accuracy
Preference
```

---

# 34. 제품 API 예시

```python
edit_smile(
    patient_image,
    tooth_reference=None,
    mouth_reference=None,
    pose_reference=None,

    whitening=0.4,
    smile_intensity=0.6,
    gum_exposure=0.2,
    tooth_roundness=0.7,

    seed=1234,
)
```

내부:

```text
slider values
      ↓
preset selector / interpolation
      ↓
prompt_embeds
      ↓
reference images
      ↓
FLUX.2 Klein
      ↓
mask composite
      ↓
final image
```

---

# 35. Runtime Architecture

최종 제품 구조 후보:

```text
                     ┌─────────────────────┐
                     │ Embedding Presets   │
                     │ .pt / safetensors   │
                     └──────────┬──────────┘
                                │
UI Sliders ────────────────> Preset Mixer
                                │
                                ▼
Patient Image ────────┐
Tooth Ref ────────────┤
Mouth Ref ────────────┤──> FLUX.2 Klein 4B
Pose Ref ─────────────┘          │
                                ▼
                         Generated Image
                                │
Mouth/Tooth Mask ───────────────┤
                                ▼
                         Local Composite
                                │
                                ▼
                            Final Result
```

제품 runtime에는:

```text
NO Qwen
NO tokenizer
```

를 목표로 한다.

---

# 36. Deployment 단계

## V1

```text
FLUX.2 Klein 4B BF16
Cached embeddings
1 patient + 1 tooth ref
Discrete preset
```

가장 단순하게 검증.

---

## V2

```text
1~3 reference
Anchor interpolation
Mask compositing
```

---

## V3

```text
Multiple slider
Embedding delta
Reference library
```

---

## V4

성능이 검증된 뒤에만:

```text
FP8 transformer
INT8 weight-only
TensorRT / torch.compile
model distillation
```

을 검토.

즉 현재는:

> **경량화보다 제품 동작 원리와 제어 안정성 검증이 먼저**

이다.

---

# 37. 추천 첫 실험 세트

가장 먼저 아래 6개만 만든다.

```text
Preset 1 : Natural smile
Preset 2 : Mild whitening
Preset 3 : Strong whitening
Preset 4 : Subtle smile
Preset 5 : Broad smile
Preset 6 : Cosmetic rounded teeth
```

Reference:

```text
Patient
+
Tooth Ref 1장
```

고정.

50명 정도에서:

```text
6 presets × fixed seeds
```

평가한다.

이 실험이 안정적으로 되면 다음 단계로 확장한다.

---

# 38. 가장 먼저 구현할 기술 검증

우선순위:

```text
1. encode_prompt 결과 저장
2. text encoder 제거 후 prompt_embeds 직접 inference
3. 출력 동일성 확인
4. peak VRAM 측정
5. 3~5단계 whitening preset
6. interpolation
7. tooth reference 추가
8. mouth reference 추가
9. multiple slider
```

이 순서가 좋다.

---

# 39. 예상 리스크

### 1. Embedding interpolation이 비선형

해결:
discrete anchors 사용.

### 2. 여러 slider 조합 시 semantic interference

해결:
2-way 조합부터 검증 / 조합별 preset 추가.

### 3. Reference 역할 혼선

해결:
reference order 고정 + role-specific prompt preset.

### 4. Identity change

해결:
mask compositing + low edit strength + preserve prompt.

### 5. 치아 하나만 정확히 제어 불가

해결:
instance mask / geometry / deterministic preprocess.

### 6. Pose change가 얼굴 identity를 흔듦

해결:
pose는 advanced/reference 기반 기능으로 분리.

---

# 40. 최종 권장 방향

현재 Smile Design 제품에는 다음 구조가 가장 현실적이다.

```text
정확한 치아 제어
(instance / 3D / color / geometry)
            ↓
Patient + Dental Reference
            ↓
Precomputed Dental Prompt Embedding
            ↓
FLUX.2 Klein 4B
            ↓
Local Harmonization
            ↓
Mask Composite
            ↓
Final Smile Simulation
```

그리고 runtime text encoder는 제거한다.

```text
Development:
Prompt
  ↓
Qwen3
  ↓
Embedding Library 생성

Production:
Slider / Preset
  ↓
Cached Embedding
  ↓
FLUX.2 Klein
```

이 구조의 가장 큰 장점:

```text
- Qwen text encoder VRAM 제거 가능
- prompt 작성 UI 불필요
- 제품 동작을 제한된 preset으로 통제 가능
- 결과 재현성이 좋아짐
- QA가 쉬워짐
- slider UI 구성 가능
- 이후 FP8/INT8/Distillation 최적화와 독립적으로 개발 가능
```

따라서 모델 경량화보다 먼저:

> **"Cached Embedding + Reference Role + Slider Control"이 실제 치과 편집에서 안정적으로 동작하는지 검증**

하는 것을 최우선 과제로 권장한다.
