# FLUX.2 Klein 4B → ~1B T2I Distillation 최종 실험 계획

## 1. 목표

현재 최우선 목표는 **FLUX.2 Klein Base 4B의 Text-to-Image 능력을 작은 Student DiT로 안정적으로 증류할 수 있는지 검증**하는 것이다.

현재 T2I distillation 자체가 충분히 성공하지 않은 상태이므로, Image Editing / Multi-reference Editing 데이터 구축과 학습은 뒤로 미룬다.

```text
Phase 1  T2I Distillation 검증        ← 현재
Phase 2  Single-image Editing
Phase 3  Multi-reference Editing
Phase 4  Dental specialization
```

즉 현재는 문제를 최대한 단순화하여 다음을 확인한다.

1. Student가 Teacher의 output/flow를 따라갈 수 있는가?
2. Hidden representation까지 Teacher를 따라갈 수 있는가?
3. 동일한 parameter budget에서 depth와 width를 어떻게 배분하는 것이 좋은가?
4. 5K 수준의 작은 데이터에서 어떤 architecture가 가장 빠르고 안정적으로 Teacher 특성을 학습하는가?

---

## 2. GPU 환경 및 병렬 실험 전략

환경:

```text
NVIDIA A6000 48GB × 6
```

6개 GPU를 하나의 거대한 학습에 사용하는 것보다, 현재는 **서로 다른 architecture를 동시에 비교하는 것이 중요**하다.

따라서:

```text
GPU0 + GPU1 → Experiment A
GPU2 + GPU3 → Experiment B
GPU4 + GPU5 → Experiment C
```

각 실험은 기본적으로:

```text
GPU X     : Teacher 4B (frozen)
GPU X + 1 : Student (~1B)
```

구성으로 독립 실행한다.

Teacher:

```python
teacher.eval()
teacher.requires_grad_(False)

with torch.no_grad():
    teacher_output = teacher(...)
```

Teacher에는 optimizer/gradient가 필요 없다.

---

## 3. 현재 Teacher 구조

기준 Teacher:

```text
FLUX.2 Klein Base 4B

Total DiT params ≈ 3.876B

hidden dimension = 3072
double-stream blocks = 5
single-stream blocks = 20
```

기존 Student:

```text
hidden ≈ 3072
double = 2
single = 3

params ≈ 1.054B
```

기존 방식은 width를 거의 유지한 상태에서:

```text
25 major blocks → 5 major blocks
```

로 depth를 매우 공격적으로 줄였다.

현재 loss는 감소하지만 Teacher 특성이 충분히 전달되지 않는 현상이 있으므로, 이번 실험에서는 **depth를 더 유지하면서 width를 줄이는 방향을 우선 검증**한다.

---

# 4. 데이터 전략

## 현재는 T2I만 사용

Editing 데이터는 사용하지 않는다.

```text
(image, caption)
```

pair만 사용한다.

현재 단계에서는 다음을 하지 않는다.

```text
Single Edit      X
Multi-ref Edit   X
Mask Edit        X
Dental Edit      X
```

T2I distillation이 확실히 동작한 이후 순차적으로 추가한다.

---

## 5. 데이터 수량

### Debug 단계

```text
100 samples
```

목적:

> Student가 Teacher를 의도적으로 overfit할 수 있는가?

100장에서도 Teacher output을 따라가지 못하면 데이터 규모 문제가 아니라 다음을 우선 의심한다.

```text
KD implementation
Teacher/Student input mismatch
architecture
initialization
loss
scheduler/timestep
conditioning
```

### Small validation

```text
500 samples
```

100장에서 성공한 뒤 학습 추이를 확인한다.

### Architecture comparison

```text
5,000 samples
```

**5K면 현재 architecture/KD 비교 목적에는 충분하다.**

5K의 목적은 최종 generalization이 아니다.

확인 대상:

```text
Loss convergence
Teacher flow imitation
Hidden representation imitation
Fixed-seed image quality
Teacher-like composition
Prompt following
Architecture 간 상대 비교
```

5K에서 성공한 뒤에만:

```text
50K
→ 200K
→ 1M
→ 필요 시 3M+
```

으로 확장한다.

---

# 6. 핵심 Architecture 실험 3개

모든 실험은 동일한:

```text
5K dataset
caption
text embedding
noise
timestep
scheduler
seed
validation prompts
```

을 사용한다.

architecture만 다르게 한다.

---

## Experiment A — Narrow + Full Depth

가장 중요한 실험.

```text
Teacher
hidden = 3072
D = 5
S = 20

        ↓

Student A
hidden = 1536
D = 5
S = 20
```

특징:

```text
Width ≈ 50%
Depth = 100%
```

목적:

> Teacher와 동일한 computational depth를 유지하고 width만 줄이면 Teacher 특성을 잘 전달할 수 있는가?

장점:

- Teacher/Student block correspondence가 정확히 1:1
- Hidden KD 구현이 가장 명확
- depth reduction 자체가 기존 실패 원인인지 확인 가능

GPU:

```text
GPU0 : Teacher 4B
GPU1 : Student A
```

---

## Experiment B — Medium Width + Mild Depth Reduction

```text
Teacher
hidden = 3072
D = 5
S = 20

        ↓

Student B
hidden = 1792
D = 5
S = 16
```

특징:

```text
Width         ≈ 58%
Double depth  = 100%
Single depth  = 80%
```

목적:

> Full-depth가 반드시 필요한가, 아니면 약간의 depth reduction으로 더 좋은 capacity allocation이 가능한가?

Double-stream 5개는 전부 유지한다.

특히 T2I 이후 Editing까지 고려하면 text/image interaction을 담당하는 multimodal 부분을 지나치게 줄이지 않는 방향을 우선 검증한다.

GPU:

```text
GPU2 : Teacher 4B
GPU3 : Student B
```

---

## Experiment C — Wider + Moderate Depth Reduction

```text
Teacher
hidden = 3072
D = 5
S = 20

        ↓

Student C
hidden = 2048
D = 4
S = 12
```

특징:

```text
Width         ≈ 67%
Double depth  = 80%
Single depth  = 60%
```

목적:

> A/B보다 더 넓은 representation을 주는 것이 일부 depth 감소를 보상할 수 있는가?

기존 `2D + 3S`보다 훨씬 많은 depth를 유지하면서 block 자체의 width도 상대적으로 크게 가져간다.

GPU:

```text
GPU4 : Teacher 4B
GPU5 : Student C
```

---

# 7. Architecture 비교 요약

| Model | Hidden | Double | Single | 핵심 질문 |
|---|---:|---:|---:|---|
| Teacher | 3072 | 5 | 20 | 기준 |
| 기존 Student | ~3072 | 2 | 3 | 너무 얕았는가? |
| A | 1536 | 5 | 20 | Full depth + narrow width가 좋은가? |
| B | 1792 | 5 | 16 | Mild depth reduction이 효율적인가? |
| C | 2048 | 4 | 12 | Wider representation이 depth 감소를 보상하는가? |

이번 단계에서는 정확히 모든 Student를 `1.000B`로 맞추는 것보다 **depth/width allocation에 따른 학습 특성을 먼저 파악**한다.

승자 architecture가 결정되면 그 구조를 기준으로 정확한:

```text
1.0B
0.7B
0.5B
```

parameter target을 맞춘다.

---

# 8. Distillation Loss

기본 loss는 다음 네 가지다.

```text
1. Forward Flow KD
2. On-policy Trajectory Flow KD
3. Intermediate Hidden KD
4. Flow Direction KD
```

`z`를 trajectory 배치 여부라고 하면 전체 loss는 다음과 같다.

\[
L =
(1-z)(\lambda_{gt} L_{gt} + \lambda_{flow} L_{flow})
+ z\lambda_{trajectory} L_{trajectory}
+ \lambda_{hidden} L_{hidden}
+ \lambda_{direction} L_{direction}
\]

trajectory 배치에서는 대응되는 clean/noise GT가 더 이상 정확하지 않으므로 `L_gt`를 사용하지 않는다. 실제 학습 objective는 forward-noised 배치와 student-rollout 배치의 혼합이다.

기본 설정:

```text
λ_gt          = 0.05
λ_flow        = 1.0
λ_hidden      = 0.25
λ_direction   = 0.05
λ_trajectory  = 1.0

trajectory probability       = 0.20
trajectory rollout steps     = 2
trajectory inference steps   = 50
trajectory warmup steps      = 1000
```

일반 배치 80%는 기존 real-data forward-noising 학습을 유지한다. 나머지 20%는 inference scheduler 간격으로 student를 두 번 `no_grad` rollout한 상태에서 teacher와 student의 flow를 비교한다. rollout 상태는 detach하므로 rollout 전체를 역전파하지 않으며, T2I와 editing에서 같은 경로를 사용한다.

`x0` reconstruction KD와 one-step state-transition KD는 rectified-flow parameterization에서 flow MSE의 timestep 재가중 형태이므로 기본 loss로 중복 추가하지 않는다. Adversarial loss와 multi-step teacher reconstruction은 few-step distillation용 별도 실험으로 남긴다.

---

# 9. Final Flow KD

Teacher와 Student에 완전히 동일한 입력을 넣는다.

```text
same text embedding
same x_t
same noise
same timestep
same positional IDs
same scheduler state
same guidance
```

Teacher:

\[
v_T(x_t,t,c)
\]

Student:

\[
v_S(x_t,t,c)
\]

Loss:

\[
L_{flow}
=
\|v_S-v_T\|^2
\]

이것이 가장 기본적인 Teacher imitation loss다.

---

# 10. Hidden KD가 필요한 이유

Final Flow KD만 사용하면 Student에게:

> 중간 과정은 어떻게 계산해도 좋으니 마지막 output만 Teacher와 비슷하게 만들어라.

라고 학습시키는 것과 비슷하다.

4B → ~1B처럼 compression ratio가 큰 경우 Student의 중간 representation이 Teacher와 전혀 다른 방향으로 갈 수 있다.

Hidden KD는 중간 representation에도 Teacher target을 제공한다.

```text
Teacher

Input
 ↓
T0 ─────────→ hidden target
 ↓
T1
 ↓
T2 ─────────→ hidden target
 ↓
...
 ↓
Tn ─────────→ hidden target
 ↓
Final Flow
```

Student:

```text
Input
 ↓
S0 ─────────→ match Teacher hidden
 ↓
S1
 ↓
S2 ─────────→ match Teacher hidden
 ↓
...
 ↓
Sn
 ↓
Final Flow ─→ match Teacher flow
```

즉 Student가 Teacher의 **최종 답뿐 아니라 중간 계산 경로까지 어느 정도 따라가도록 유도**한다.

---

# 11. Depth가 같아야 Hidden KD를 사용할 수 있는가?

아니다.

**Depth가 달라도 Hidden KD를 사용할 수 있다.**

차이는 layer mapping 방식이다.

---

## Case 1 — Depth 동일

Experiment A:

```text
Teacher D0 ↔ Student D0
Teacher D1 ↔ Student D1
Teacher D2 ↔ Student D2
Teacher D3 ↔ Student D3
Teacher D4 ↔ Student D4
```

Single:

```text
Teacher S0  ↔ Student S0
Teacher S1  ↔ Student S1
...
Teacher S19 ↔ Student S19
```

가장 깔끔한 1:1 mapping이다.

---

## Case 2 — Depth 다름

예:

```text
Teacher Single = 20
Student Single = 10
```

normalized depth 기준으로 mapping한다.

일반적으로 Student layer `i`에 대응하는 Teacher layer:

\[
j_i =
round
\left(
i \frac{N_T-1}{N_S-1}
\right)
\]

예:

```text
Student → Teacher

0 → 0
1 → 2
2 → 4
3 → 6
4 → 8
5 → 11
6 → 13
7 → 15
8 → 17
9 → 19
```

즉 depth가 달라도 network 진행 위치가 비슷한 block끼리 대응시킬 수 있다.

---

# 12. Width가 다를 때 Hidden KD

Teacher:

```text
hidden = 3072
```

Student A:

```text
hidden = 1536
```

이므로 직접 MSE를 계산할 수 없다.

따라서 projection을 사용한다.

```text
Student hidden
[B, N, 1536]

      ↓

Linear
1536 → 3072

      ↓

[B, N, 3072]

      ↓

MSE

      ↑

Teacher hidden
[B, N, 3072]
```

식:

\[
L_{hidden}
=
\|P(h_S)-h_T\|^2
\]

여기서 `P`는 학습 가능한 projection layer다.

예:

```python
student_projected = projection(student_hidden)

loss_hidden = F.mse_loss(
    student_projected,
    teacher_hidden.detach()
)
```

Projection은 distillation training용이므로 최종 inference 모델에서는 제거할 수 있다.

---

# 13. 모든 Hidden Layer에 Loss를 걸 필요는 없음

처음부터 모든 block에 Hidden KD를 적용할 필요는 없다.

Double-stream은:

```text
D0
D2
D4
```

정도.

Single-stream은:

```text
early
25%
50%
75%
final
```

위치 정도부터 시작한다.

Teacher Single 20개라면 예:

```text
S0
S4
S9
S14
S19
```

총 Hidden KD point:

```text
Double : 3
Single : 5

Total : 8
```

정도로 시작한다.

---

# 14. Hidden Loss

K개의 matched feature가 있다면:

\[
L_{hidden}
=
\frac{1}{K}
\sum_{k=1}^{K}
\|
P_k(h^S_k)
-
h^T_{map(k)}
\|^2
\]

예:

```python
hidden_loss = 0.0

for s_hidden, t_hidden, proj in matched_features:
    s_hidden = proj(s_hidden)

    hidden_loss += F.mse_loss(
        s_hidden,
        t_hidden.detach()
    )

hidden_loss /= len(matched_features)

loss = flow_loss + 0.25 * hidden_loss
```

---

# 15. Experiment A/B/C Hidden Mapping

## A — 1536 / D5 / S20

Depth가 Teacher와 동일하다.

```text
Double:
D0 → D0
D2 → D2
D4 → D4

Single:
S0  → S0
S4  → S4
S9  → S9
S14 → S14
S19 → S19
```

단 width projection:

```text
1536 → 3072
```

필요.

A는 **Hidden KD 자체가 제대로 작동하는지 검증하기 가장 좋은 architecture**다.

---

## B — 1792 / D5 / S16

Double은 동일 depth:

```text
D0 → D0
D2 → D2
D4 → D4
```

Single은 normalized mapping:

```text
Student S0  → Teacher S0
Student S4  → Teacher ~S5
Student S8  → Teacher ~S10
Student S12 → Teacher ~S15
Student S15 → Teacher S19
```

width projection:

```text
1792 → 3072
```

---

## C — 2048 / D4 / S12

Double과 Single 모두 depth mapping 필요.

예:

```text
Student D0 → Teacher D0
Student D1 → Teacher D1
Student D2 → Teacher D3
Student D3 → Teacher D4
```

Single:

```text
Student S0  → Teacher S0
Student S3  → Teacher ~S5
Student S6  → Teacher ~S10
Student S9  → Teacher ~S16
Student S11 → Teacher S19
```

width projection:

```text
2048 → 3072
```

---

# 16. 학습 순서

## Step 0 — Teacher Clone Sanity Check

Student를 Teacher와 완전히 동일한 구조/weight로 만들어 forward 결과를 비교한다.

```text
Teacher 5D + 20S
Student 5D + 20S
```

동일 input에서:

```python
max_diff = (teacher_output - student_output).abs().max()
```

가 numerical error 수준이어야 한다.

이 단계에서 다음을 검증한다.

```text
text embeddings
noise
x_t
timestep
guidance
position IDs
token ordering
scheduler preprocessing
dtype
```

---

## Step 1 — 100 Sample Overfit

A/B/C를 돌리기 전에 최소 한 구조에서:

```text
100 samples
```

로 Teacher imitation이 가능한지 확인한다.

성공 조건:

```text
flow loss 크게 감소
hidden loss 감소
fixed seed 결과가 Teacher 방향으로 이동
```

100장도 못 외우면 5K/50K로 확대하지 않는다.

---

## Step 2 — 500 Sample Test

```text
500 samples
```

로 학습 안정성과 초기 generalization을 확인한다.

---

## Step 3 — 5K A/B/C Parallel Experiment

```text
GPU0-1 → A
GPU2-3 → B
GPU4-5 → C
```

세 실험을 동시에 실행한다.

모든 조건을 동일하게 유지한다.

---

# 17. Checkpoint 비교

5K dataset을 여러 epoch 반복해도 된다.

현재 목적은 generalization보다:

> Teacher를 얼마나 빠르고 정확하게 모방할 수 있는가?

이기 때문이다.

예시 checkpoint:

```text
step 0
step 250
step 500
step 1,000
step 2,000
step 4,000
```

각 checkpoint마다 동일 validation prompt/seed로 이미지를 생성한다.

---

# 18. Validation

고정 prompt:

```text
100~200 T2I prompts
```

Teacher와 모든 Student에:

```text
same prompt
same text embedding
same seed
same initial noise
same scheduler
same inference steps
same resolution
```

을 사용한다.

비교:

```text
Teacher
Existing 2D+3S
Student A
Student B
Student C
```

---

# 19. 평가 항목

정량:

```text
Flow KD loss
Hidden KD loss
Teacher/Student output similarity
CLIP similarity
DINO/image feature similarity (선택)
```

정성:

```text
Prompt following
Composition
Object identity
Anatomy
Color
Texture
Image quality
Teacher-like characteristics
```

특히 fixed seed에서:

```text
Student training step 증가
        ↓
Teacher output 방향으로 이동
```

하는지를 시각적으로 확인한다.

---

# 20. Architecture 성공 기준

좋은 Student:

```text
Flow KD 빠르게 감소
Hidden KD 안정적으로 감소
Teacher image 특징 재현
Prompt following 유지
5K train data overfit 가능
Validation에서도 일부 Teacher behavior 유지
```

나쁜 Student:

```text
Loss만 감소
Teacher와 image distribution이 다름
Prompt 무시
Composition 붕괴
고정 seed 결과 개선 없음
```

후자의 경우 데이터 수를 늘리는 것으로 해결하려 하지 않는다.

---

# 21. Architecture 선정 후 Hidden KD Ablation

A/B/C 중 가장 좋은 architecture를 선정한 다음 6 GPU를 다시 세 실험으로 사용한다.

예:

```text
GPU0-1
Winner
Flow KD only

GPU2-3
Winner
Flow KD + Hidden KD 0.1

GPU4-5
Winner
Flow KD + Hidden KD 0.5
```

즉:

| Experiment | Flow | Hidden |
|---|---:|---:|
| H0 | 1.0 | 0 |
| H1 | 1.0 | 0.1 |
| H2 | 1.0 | 0.5 |

이를 통해 Hidden KD가 실제 이미지 품질과 Teacher imitation에 얼마나 기여하는지 확인한다.

필요하면 이후:

```text
0.25
```

등으로 세밀하게 조정한다.

---

# 22. 데이터 Scaling

Architecture와 KD loss가 결정된 이후에만 데이터를 확장한다.

```text
100
 ↓
500
 ↓
5K        ← architecture/KD 검증
 ↓
50K
 ↓
200K
 ↓
1M
 ↓
3M+       ← 필요 시
```

5K에서 실패하면 1M으로 넘어가지 않는다.

---

# 23. Editing은 언제 시작하는가?

T2I distillation이 성공한 이후다.

```text
Teacher 4B
    ↓
T2I Distillation
    ↓
Student ~1B T2I
    ↓
T2I 성공 확인
    ↓
Single-image Edit 추가
    ↓
T2I + Single Edit 성공
    ↓
Multi-reference Edit 추가
    ↓
General T2I/Edit Student
    ↓
Dental specialization
```

따라서 지금은:

```text
UltraEdit 대규모 구축       X
Multi-ref synthetic 생성    X
Mask Edit 구축              X
Dental Edit 구축            X
```

에 시간을 쓰지 않는다.

---

# 24. 전체 실험 Flow

```text
FLUX.2 Klein Base 4B
          │
          ▼
Teacher Clone Sanity Check
          │
          ▼
100 Sample Overfit
          │
          ▼
500 Sample Test
          │
          ▼
5K T2I
          │
          ├──────────────┬──────────────┐
          ▼              ▼              ▼
      Student A       Student B       Student C
     1536/5/20       1792/5/16       2048/4/12
      GPU0-1          GPU2-3          GPU4-5
          │              │              │
          └──────────────┴──────────────┘
                         │
                         ▼
                Best Architecture
                         │
                         ▼
                Hidden KD Ablation
                 0 / 0.1 / 0.5
                         │
                         ▼
                   T2I KD 검증
                         │
                         ▼
               50K → 200K → 1M
                         │
                         ▼
                  Single Editing
                         │
                         ▼
                  Multi Editing
                         │
                         ▼
               Dental specialization
```

---

# 25. 현재 최종 결론

현재 가장 중요한 것은 데이터 양을 늘리는 것이 아니다.

핵심 질문은:

> 기존 ~1B Student가 실패하는 이유가 `5D+20S → 2D+3S`라는 극단적인 depth reduction 때문인가?

이를 검증하기 위해 **width-first architecture**를 우선 비교한다.

```text
A : Narrow + Full Depth
B : Medium Width + Mild Depth Reduction
C : Wider + Moderate Depth Reduction
```

6×A6000은:

```text
2 GPU × 3 independent experiments
```

로 사용한다.

그리고 Hidden KD는:

- depth가 같을 때만 가능한 것이 아니다.
- 동일 depth에서는 1:1 mapping이 가장 깔끔하다.
- 다른 depth에서는 normalized-depth mapping을 사용한다.
- 다른 width에서는 trainable projection으로 dimension을 맞춘다.

현재 단계의 권장 loss:

```text
Flow KD   = 1.0
Hidden KD = 0.25
```

Architecture 선정 후 Hidden KD weight 자체를 별도의 3-way 병렬 실험으로 검증한다.

**5K T2I 데이터면 현재 architecture와 KD 방식의 학습 추이를 비교하기에 충분하다.**

T2I distillation이 명확하게 성공한 뒤에 Single Edit → Multi-reference Edit 순서로 확장한다.

---

# 26. Phase 2/3 MagicBrush 구현

T2I A/B/C 실험에서 editing 학습으로 전환할 수 있는 수준의 수렴을 확인했으므로 `my/mydisillation.py`에 MagicBrush 데이터 경로를 추가했다.

```text
Phase 2 single
  noisy target latent + clean current-source latent + instruction

Phase 3 multi_reference
  noisy target latent + clean original/current-source latents + instruction
```

Teacher와 Student는 동일한 target noise, timestep, text embedding, reference latent를 입력받는다. Flow/hidden/GT loss 구성은 T2I와 동일하게 유지한다.

MagicBrush의 multi-turn은 여러 독립 reference를 조합하는 정식 multi-image 데이터가 아니다. 이전 turn의 target이 다음 turn의 source가 되는 session 구조다. 현재 Phase 3은 이 session history에서 최초 source와 현재 source를 선택해 FLUX.2의 복수 image-conditioning 경로를 검증하는 실험이다.

설정 파일:

```text
my/configs/editing/phase2_magicbrush_single_{a,b,c}.yaml
my/configs/editing/phase3_magicbrush_multiref_{a,b,c}.yaml
```

실행 명령과 재개 방법은 `my/configs/editing/README.md`에 정리한다.

---

# 27. Student EMA

T2I A/B/C와 Phase 2/3 A/B/C는 최종 추론용 student transformer의 FP32 EMA를 사용한다. EMA 갱신 단위는 epoch나 micro-batch가 아니라 optimizer step이다. 현재 2 GPU, per-device batch 1, gradient accumulation 4 설정에서는 8 samples를 처리할 때마다 optimizer와 EMA가 각각 한 번 갱신된다.

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

기본 GPU EMA는 rank 0 GPU에 B/C 기준 약 4.3~4.4GiB를 추가 사용한다. 실제 B 모델(1.158B) 측정값은 optimizer step당 GPU 약 0.021초, CPU 약 0.262초다. VRAM이 부족하면 `--ema_device cpu`로 override한다. Checkpoint는 raw `transformer/`와 EMA `transformer_ema/`를 함께 저장하며 validation도 `validation/student`와 `validation/student_ema`를 분리해 기록한다.
