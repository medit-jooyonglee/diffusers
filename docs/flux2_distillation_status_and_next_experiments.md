# FLUX.2 Distillation 학습 현황과 다음 실험

> 학습 현황 기준: 2026-09-23 05:28 UTC; EMA 구현 갱신: 2026-09-28
> 관련 계획: [flux2_t2i_distillation_final_plan.md](flux2_t2i_distillation_final_plan.md)
> 학습 구현: [mydisillation.py](../my/mydisillation.py)

이 문서는 FLUX.2 Klein Base 4B teacher를 약 1B student로 줄이는 capacity distillation의 현재 상태를 정리한다. Text-to-Image(T2I), MagicBrush single-image editing, multi-reference editing을 모두 포함하며, 실행 중인 job의 step과 로그 수치는 계속 변한다.

## 1. 결론 요약

- **T2I distillation 파이프라인 자체는 정상적으로 학습된다.** Flow/hidden loss가 감소했고, 고정 prompt/seed validation에서도 이미지 윤곽과 의미가 형성된다.
- 아직 **teacher 수준의 최종 이미지 품질에 도달했다고 결론 내리기는 이르다.** 단일 timestep loss 감소와 50-step trajectory의 누적 오차는 별개의 문제다.
- A/B/C 중에는 **B가 현재 잠정 우세**다. 초기 15K 부근에는 C가 좋아 보였지만, 최근 고정 validation에서는 B가 더 안정적인 경향이 있다. 정량 평가가 없으므로 확정 결론은 아니다.
- B/C는 checkpoint-48000에서 새 loss를 켜고 재개했으며, 현재 새 objective로 학습한 구간은 약 3.3K step뿐이다. **Trajectory/Direction KD의 장기 효과는 아직 미검증**이다.
- MagicBrush Phase 2는 학습 경로가 동작한다. 과거 Edit-C는 약 15.9K까지 학습했고, 현재 Edit-B가 새 loss로 약 1.9K까지 진행 중이다.
- 현재 Edit-B validation은 별도 dev가 아니라 **train의 첫 4개 샘플**이다. `/dev/metadata.jsonl`이 없으므로 generalization을 판단할 수 없다.
- Edit-B의 `--max_train_steps 800000`은 현재 속도에서 약 40일, 8,807개 데이터 기준 약 727 dataset pass다. 이는 목표 step이라기보다 과도한 상한이며, **독립 validation 없이 끝까지 실행하면 과적합·비용 낭비 위험이 매우 크다.**
- Phase 3 multi-reference는 single/2-reference 및 trajectory GPU smoke test까지 통과했지만, 장시간 학습은 아직 시작하지 않았다.

현재 판단은 다음과 같다.

```text
T2I 구현 검증             완료
T2I 학습 가능성 확인      완료
T2I 최종 품질 검증        진행 중
Editing 데이터 경로       완료
Single-edit 장기 학습      초기 진행 중
독립 editing validation   미완료
Multi-reference smoke     완료
Multi-reference 장기 학습 미시작
```

## 2. 모델 구조와 학습 공통 조건

Teacher와 T2I/editing student는 모두 같은 FLUX.2 DiT 계열 transformer를 사용한다. Editing에서 별도의 DiT를 붙이는 것이 아니라, 동일한 dual-stream/single-stream 블록에 reference image latent가 추가 입력된다.

| 모델 | Hidden | Double blocks | Single blocks | Student params | 학습 전용 hidden projection |
|---|---:|---:|---:|---:|---:|
| Teacher | 3072 | 5 | 20 | 3.876B | - |
| A | 1536 | 5 | 20 | 0.975B | 37.7M |
| B | 1792 | 5 | 16 | 1.158B | 44.0M |
| C | 2048 | 4 | 12 | 1.183B | 50.3M |

Hidden projection은 width가 다른 student hidden을 teacher hidden dimension으로 투영하기 위한 학습 전용 모듈이다. 최종 추론용 student transformer parameter에는 포함되지 않는다.

현재 장기 job의 공통 조건은 다음과 같다.

- A6000 48GB 2장, bf16, gradient checkpointing, TF32
- per-device batch 1, gradient accumulation 4, global batch 8
- `FlowMatchEulerDiscreteScheduler`
- 학습 timestep은 `logit_normal(mean=0, std=1)` 분포로 표본화
- T2I learning rate `2e-5`, editing learning rate `1e-5`
- unconditional conditioning dropout 10%
- validation은 50 inference steps, CFG 4.0
- 2026-09-28 이후 설정: student transformer FP32 EMA를 매 optimizer step 갱신, 기본 GPU 저장

현재 `accelerate` 2-GPU 실행은 일반 DDP다. 각 GPU에 teacher와 student가 모두 복제되므로 처리량은 늘지만 모델 메모리가 두 GPU에 분할되지는 않는다. 더 큰 teacher-clone 실험에는 FSDP 또는 ZeRO 같은 model/optimizer sharding이 별도로 필요하다.

EMA는 teacher나 hidden projection이 아니라 최종 추론용 student transformer에만 적용한다. 기본 `ema_device: gpu`는 rank 0 GPU에 B/C 기준 약 4.3~4.4GiB를 추가하므로 여유가 부족하면 `--ema_device cpu`로 바꾼다. 실제 B 모델(1.158B) 측정값은 optimizer step당 GPU 약 0.021초, CPU 약 0.262초다. CPU 모드는 rank 0 pinned host memory를 사용한다. Raw/EMA validation과 checkpoint를 모두 보존한다. 이 변경은 이미 실행 중인 Python process에는 소급 적용되지 않으며 재시작 이후부터 유효하다.

현재 설정 파일:

- T2I: [A](../my/student_configure_t2i_a.yaml), [B](../my/student_configure_t2i_b.yaml), [C](../my/student_configure_t2i_c.yaml)
- Phase 2 single-edit: [A](../my/configs/editing/phase2_magicbrush_single_a.yaml), [B](../my/configs/editing/phase2_magicbrush_single_b.yaml), [C](../my/configs/editing/phase2_magicbrush_single_c.yaml)
- Phase 3 multi-reference: [A](../my/configs/editing/phase3_magicbrush_multiref_a.yaml), [B](../my/configs/editing/phase3_magicbrush_multiref_b.yaml), [C](../my/configs/editing/phase3_magicbrush_multiref_c.yaml)

## 3. T2I 학습 현황

T2I 데이터는 CommonCatalog 로컬 parquet 474개를 사용한다. 현재 A/B/C YAML의 `max_train_samples`는 1,000,000으로 확장했지만, A의 기존 학습 구간은 과거 1,000-sample 반복 설정으로 수행됐다.

| 실험 | 현재 상태 | 최근 학습 step | 최근 저장 checkpoint | 최근 validation | 비고 |
|---|---|---:|---:|---:|---|
| T2I-A | 중단 | 24,796 | 24,000 | 24,500 | Full-depth/narrow. 수렴과 형상 형성이 상대적으로 느렸음 |
| T2I-B | 실행 중 | 약 51.3K | 48,000 | 51,000 | 현재 잠정 우세. 48K부터 새 loss로 재개 |
| T2I-C | 실행 중 | 약 51.3K | 48,000 | 51,000 | 초기에는 가장 좋아 보였으나 최근에는 B가 더 안정적인 경향 |

현재 output 사용량은 대략 A 27GB, B 287GB, C 143GB다. B/C의 차이는 남아 있는 checkpoint 수 차이가 크기 때문이다. 현재 코드에는 checkpoint 보존 개수 제한이 없으므로 장기 실행 전에 retention 기능이 필요하다.

현재 A YAML에도 trajectory/direction 설정이 들어 있지만 A job은 그 설정으로 재시작하지 않았다. 따라서 위 A 결과는 기존 GT/forward-flow/hidden objective의 결과다.

### 최근 500 step 조건부 평균

아래 수치는 새 loss로 재개된 B/C TensorBoard scalar의 최근 500 step 평균이다. 일반 배치와 trajectory 배치를 분리했으며, 이미지 품질을 직접 나타내는 점수는 아니다.

| Run | Total | Forward flow | Trajectory flow | GT, 일반 배치 | Teacher GT, 일반 배치 | Hidden | Direction | Trajectory 비율 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T2I-B | 0.0924 | 0.0487 | 0.0405 | 0.8284 | 0.7832 | 0.0355 | 0.0205 | 20.1% 전체 구간 |
| T2I-C | 0.0947 | 0.0492 | 0.0401 | 0.8265 | 0.7804 | 0.0427 | 0.0206 | 20.1% 전체 구간 |

해석:

- B/C의 flow 계열 수치는 매우 비슷하다.
- B의 hidden loss가 C보다 낮지만, projection 크기와 architecture가 달라 절대값만으로 우열을 확정할 수 없다.
- `student GT - teacher GT` 차이는 최근 평균 약 0.04~0.05다. Teacher 자체도 GT target에 대해 약 0.78의 loss를 가지므로 student GT가 0에 수렴해야 한다고 기대하면 안 된다.
- `lambda_gt=0.05`라도 raw GT가 약 0.83이므로 가중 기여는 약 0.041이다. 이는 forward flow 약 0.049와 같은 규모다. 따라서 0.05는 현재 loss scale에서 결코 무시할 만큼 작은 값이 아니다.
- 현재 수치만으로 B/C의 생성 품질 차이를 설명하기 어렵다. 동일 prompt/seed의 trajectory 이미지 평가가 반드시 함께 필요하다.

## 4. Editing 학습 현황

MagicBrush train에는 실제 사용 가능한 edit turn이 8,807개 있다. `max_train_samples: null`이므로 전체 train split을 사용한다.

### Phase 2: single-image editing

학습 입력은 다음과 같다.

```text
instruction + clean source latent + noisy target latent
```

| 실험 | 초기화 | 상태 | 최근 step | 최근 checkpoint | 최근 validation | Loss 구성 |
|---|---|---|---:|---:|---:|---|
| Edit-C | T2I-C checkpoint-24000 | 중단 | 15,875 | 12,000 | 15,000 | 기존 GT + forward flow + hidden |
| Edit-B | T2I-B checkpoint-48000 | 실행 중 | 약 1.9K | 아직 없음 | 1,000 | 새 trajectory + direction 포함 |

현재 Edit-C YAML에도 새 loss가 정의돼 있지만 위의 과거 Edit-C run은 YAML 변경 전에 수행됐다.

Edit-B 최근 500 step 조건부 평균은 다음과 같다.

| Total | Forward flow | Trajectory flow | GT, 일반 배치 | Teacher GT, 일반 배치 | Hidden | Direction | Trajectory 비율 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0731 | 0.0414 | 0.0452 | 0.2138 | 0.2019 | 0.0873 | 0.0128 | 21.1% 전체 구간 |

수치상으로는 학습이 안정적으로 진행 중이며 NaN/OOM은 없다. 다만 다음 이유로 Edit-B와 Edit-C loss를 직접 비교하면 안 된다.

- architecture가 다르다.
- 초기 T2I checkpoint가 24K와 48K로 다르다.
- Edit-C에는 trajectory/direction loss가 없었다.
- 학습 step이 15.9K와 1.9K로 크게 다르다.

### Validation 상태

`--validation_editing_dataset_root`를 지정하지 않으면 validation을 생략하는 것이 아니라 `--editing_dataset_root`로 fallback한다. 현재 dev metadata가 없어서 Edit-B는 train의 정렬상 첫 4개 샘플을 반복 검증한다.

따라서 현재 validation은 다음 용도로만 유효하다.

- 같은 샘플이 step에 따라 좋아지는지 확인
- source/target/teacher/student 입출력 연결 확인
- 학습 붕괴, 색상 폭주, NaN 확인

다음 판단에는 사용할 수 없다.

- unseen instruction generalization
- unseen source image editing 성능
- 과적합 여부
- B/C architecture의 공정한 비교

### `800000` step 문제

현재 Edit-B 실행값은 `--max_train_steps 800000`이다.

```text
8,807 samples / global batch 8 ≈ 1,101 optimizer steps per dataset pass
800,000 / 1,101 ≈ 727 passes
현재 약 4.3 sec/step → 약 40일
```

Diffusion 학습은 매 반복마다 noise와 timestep이 달라지므로 동일 이미지를 반복한다고 즉시 동일한 의미의 과적합이 발생하는 것은 아니다. 그래도 instruction/image 다양성은 8,807개로 고정되므로 800K를 독립 dev 없이 끝까지 실행하는 것은 권장하지 않는다.

첫 checkpoint-6000을 확보한 뒤 dev 평가를 붙이고, 우선 6K/12K/18K/30K milestone에서 선택하는 편이 안전하다. `max_train_steps`는 절대 global step 기준이므로 재개할 때 `--resume_from_checkpoint latest --max_train_steps 30000`처럼 사용할 수 있다.

### Phase 3: multi-reference editing

구현된 입력은 다음과 같다.

```text
instruction
+ 동일 MagicBrush session의 최초 source
+ 현재 turn source
+ noisy target latent
```

현재 상태:

- A/B/C용 Phase 3 YAML 작성 완료
- 2-reference forward/hidden/trajectory 경로 GPU smoke test 완료
- B, batch 1, gradient checkpointing, 512 해상도에서 A6000 48GB 실행 확인
- 장시간 학습 및 품질 평가는 미수행

MagicBrush multi-turn은 서로 독립적인 여러 reference를 조합하는 일반적인 multi-image edit 데이터가 아니다. 이전 edit 결과가 다음 turn의 source가 되는 session-history 데이터이므로, 이 Phase 3 결과를 곧바로 범용 multi-reference 능력으로 해석하면 안 된다.

## 5. 구현된 loss

일반 배치 여부를 `z=0`, trajectory 배치를 `z=1`이라고 하면 현재 objective는 다음과 같다.

\[
L =
(1-z)(\lambda_{gt}L_{gt} + \lambda_{flow}L_{flow})
+ z\lambda_{trajectory}L_{trajectory}
+ \lambda_{hidden}L_{hidden}
+ \lambda_{direction}L_{direction}
\]

기본값:

```text
lambda_gt          = 0.05
lambda_flow        = 1.0
lambda_hidden      = 0.25
lambda_direction   = 0.05
lambda_trajectory  = 1.0

trajectory_probability       = 0.20
trajectory_rollout_steps     = 2
trajectory_num_inference_steps = 50
trajectory_warmup_steps      = 1000
```

### 5.1 GT flow loss

```text
L_gt = weighted MSE(student_flow, noise - clean_latent)
```

Real-data forward path의 rectified-flow target을 직접 학습한다. Teacher imitation만 했을 때 student가 teacher의 작은 오차까지 그대로 복제하는 문제를 완화한다.

`teacher_gt`는 teacher prediction과 같은 target의 차이를 기록하는 진단값이며 teacher를 업데이트하지 않는다. Teacher GT가 0이 아니므로 student GT 절대값만 보지 말고 다음도 함께 봐야 한다.

```text
student_gt - teacher_gt
validation image quality
teacher-student trajectory distance
```

### 5.2 Forward Flow KD

```text
L_flow = weighted MSE(student_flow(x_t, t, c), teacher_flow(x_t, t, c))
```

Teacher와 student에 같은 clean latent, noise, timestep, prompt embedding, guidance, reference latent를 넣는다. 기존 distillation의 핵심 output KD다.

YAML의 `lambda_kd: 0.0`은 KD가 꺼졌다는 뜻이 아니다. `lambda_kd`는 예전 설정과의 호환용 alias이며, 현재는 명시적인 `lambda_flow: 1.0`이 사용된다. TensorBoard의 `train/loss_kd`도 호환을 위해 `train/loss_flow`와 같은 값을 기록한다.

### 5.3 Intermediate Hidden KD

```text
student hidden → train-only linear projection → teacher hidden
L_hidden = selected block feature matching loss
```

Student의 일부 double/single block feature를 대응하는 teacher block과 맞춘다. Width가 다르므로 학습 전용 projection을 사용한다. Output flow만 맞추는 것보다 내부 표현 전달을 돕지만, 지나치게 큰 가중치는 작은 student에게 teacher representation을 과도하게 강제할 수 있다.

### 5.4 Flow Direction KD

```text
L_direction = 1 - cosine_similarity(student_flow, teacher_flow)
```

Flow MSE가 크기와 방향을 함께 맞추는 반면 direction loss는 이동 방향을 별도로 강조한다. 일반 배치와 trajectory 배치 모두에 적용된다.

이 loss는 trajectory를 생성하지 않는다. **Direction은 주어진 상태에서 무엇을 맞출지**, trajectory KD는 **어떤 상태에서 가르칠지**를 결정한다.

### 5.5 Local On-policy Trajectory Flow KD

구현 흐름은 다음과 같다.

1. Real latent를 inference schedule상의 timestep으로 forward-noise한다.
2. Student를 2 Euler step `no_grad` rollout한다.
3. Student가 도달한 상태를 detach한다.
4. 그 상태에서 teacher와 student flow를 다시 계산한다.
5. `L_trajectory`로 teacher flow를 따라가게 한다.

Editing에서도 같은 경로를 사용하며 rollout 동안 reference latent는 고정한다.

이 loss의 목적은 teacher의 정상 trajectory 위에서만 답을 외우는 것이 아니라, student가 실제 inference 중 조금 벗어난 상태에서도 teacher의 복귀 방향을 배우게 하는 것이다. 다만 현재 구현은 정확히 표현하면 **2-step local/truncated on-policy KD**다.

현재 구현하지 않은 범위:

- 전체 50-step student trajectory rollout
- rollout graph 전체 역전파
- 여러 step의 누적 state/reconstruction loss
- DAgger식 장기 state dataset aggregation
- consistency/distillation 기반 few-step sampler 학습

Trajectory 배치에서는 rollout 후 상태에 원래 `noise-clean_latent` GT를 그대로 적용할 수 없으므로 `L_gt`를 0으로 두고 teacher flow만 사용한다.

## 6. 구현 및 실행 검증 완료 항목

- Python syntax와 YAML schema 검사 통과
- T2I-B 단일 GPU, trajectory 100% 강제 smoke 통과
- T2I-B 2-GPU DDP trajectory smoke 통과
- MagicBrush single-edit trajectory smoke 통과
- MagicBrush 2-reference trajectory smoke 통과
- 기본 20% trajectory 혼합 8-step smoke 통과
- rank 간 trajectory branch 결정을 동기화하여 DDP collective 불일치 방지
- TensorBoard에 다음 tag 기록

```text
train/loss_gt
train/loss_flow
train/loss_hidden
train/loss_direction
train/loss_trajectory
train/teacher_gt
train/unconditional_fraction
train/trajectory_fraction
```

실제 장기 job은 GPU당 약 45GB를 사용하고 있다. 현재 B/C 및 Edit-B 구조는 A6000 48GB에 들어가지만 여유가 작다.

## 7. 현재 문제점과 한계

### P0. Editing 독립 validation 부재

가장 큰 판정 장애물이다. train 첫 4개 이미지는 memorization과 generalization을 분리하지 못한다. MagicBrush dev 528개를 내려받고 고정 subset을 사용해야 한다.

### P0. Edit-B 800K 상한과 checkpoint 운영

800K는 약 40일이고 727 dataset pass다. 또한 checkpoint 보존 제한이 없어 디스크 사용량이 계속 증가한다. `checkpoints_total_limit`과 milestone 보존 정책이 필요하다.

### P1. 이미지 품질 정량 평가 부재

현재 B/C job은 validation마다 4장만 생성하며, B 우세 판단도 TensorBoard 이미지 육안 비교다. Loss가 낮아도 50회 적분 중 오차 방향이 일관되면 이미지가 무너질 수 있고, 반대로 raw loss 차이가 작아도 이미지 품질 차이는 클 수 있다.

### P1. 새 loss의 효과가 분리되지 않음

B/C 장기 run은 trajectory와 direction을 동시에 켰다. 기존 checkpoint에서 이어 학습했기 때문에 다음 요인이 섞여 있다.

- 추가 학습 step 효과
- 1,000 → 1,000,000 sample 변경 효과
- Direction KD 효과
- Trajectory KD 효과

따라서 현재 run이 좋아져도 어느 변경이 원인인지 알 수 없다.

### P1. Trajectory KD가 짧은 local rollout임

2-step rollout은 분포 이탈을 일부 노출하지만 50-step 누적 오차 전체를 대표하지 않는다. 긴 horizon 안정성이 목표라면 rollout 길이, 시작 timestep, NFE별 평가가 필요하다.

### P1. CFG/unconditional coverage가 제한적임

현재 unconditional dropout은 10%다. CFG 1.0은 정상인데 4.0만 무너지면 architecture보다 conditional/unconditional 차이 추정이 불안정할 가능성이 높다. 현재는 paired conditional/unconditional teacher prediction을 한 배치에서 직접 맞추는 전용 CFG loss가 없다.

### P2. Timestep 구간별 오류를 보지 않음

전체 평균 loss만 기록하므로 어떤 noise level에서 student가 실패하는지 알 수 없다. 낮은 noise 영역의 작은 방향 오차가 디테일에 큰 영향을 줄 수 있고, 높은 noise 영역 오류는 전체 구도를 망가뜨릴 수 있다.

### P2. T2I와 editing의 catastrophic forgetting 가능성

Editing은 같은 transformer 전체를 계속 업데이트한다. 현재 loader는 T2I와 editing을 동시에 섞지 않으므로 editing 성능이 좋아지는 동안 T2I 생성 능력이 떨어질 수 있다.

### P2. MagicBrush mask 미사용

다운로드한 mask를 conditioning이나 loss에 사용하지 않는다. Full-target flow 학습만으로도 edit은 가능하지만, 편집 외 영역 보존을 명시적으로 강제하지 못한다.

### P2. Phase 3 데이터 의미의 한계

현재 multi-reference는 MagicBrush session history에서 만든 두 reference다. 서로 독립적인 두 이미지의 identity/object/style을 결합하는 범용 multi-reference task와는 다르다.

### P2. Capacity distillation과 step distillation은 다름

현재 목적은 작은 student가 teacher의 flow field를 모사하는 것이다. 10-step 생성이 50-step과 비슷해 보일 수는 있지만, few-step 품질을 직접 보장하는 consistency/progressive distillation objective는 구현하지 않았다.

### 기타 운영 한계

- TensorBoard scalar는 gradient accumulation 내 마지막 microbatch 값으로 기록되어 분산이 크다.
- B output이 이미 약 287GB다.
- 현재 DDP는 memory sharding이 아니다.
- A/B/C parameter 수가 완전히 동일하지 않아 순수 depth/width 비교에는 작은 budget 차이가 섞인다.

## 8. 다음 실험 우선순위

### 8.1 즉시 해야 할 운영 작업

1. MagicBrush dev 528개를 완성하고 `--validation_editing_dataset_root`를 반드시 지정한다.
2. Edit-B는 checkpoint-6000을 확보한 뒤 6K validation을 검토한다.
3. Edit-B의 목표를 우선 30K 정도의 절대 step으로 낮추고 6K 간격으로 선택한다. 800K는 수동 중단을 전제로 한 상한으로만 취급한다.
4. checkpoint retention을 구현해 최근 2~3개와 수동 지정 milestone만 남긴다.
5. B/C는 새 loss 적용 후 첫 checkpoint-54000과 60000에서 같은 evaluation matrix를 실행한다.

### 8.2 고정 evaluation matrix 구축

모든 비교는 prompt, seed, scheduler, resolution을 고정한다.

#### T2I

```text
checkpoints: 48K baseline, 54K, 60K, 이후 milestone
CFG:         1.0, 2.0, 4.0
NFE:         10, 20, 50
prompts:     최소 32개
seeds:       prompt당 최소 4개
teacher:     동일 prompt/seed 결과 저장
```

권장 지표:

- teacher/student 동일 seed LPIPS 또는 DINO distance
- prompt-image CLIP similarity
- aesthetic/quality metric
- 인체·동물·복수 객체·텍스트 등 failure category별 육안 pairwise 평가

#### Editing

```text
dev 고정 subset: 최소 64개
CFG:             1.0, 2.0, 4.0
NFE:             20, 50
저장 항목:       source, target, teacher, student, instruction
```

권장 지표:

- target과의 LPIPS/DINO similarity
- instruction CLIP directional similarity
- mask 내부 edit 성공률
- mask 외부 source preservation
- editing checkpoint의 T2I 고정 prompt 성능

### 8.3 Loss ablation

같은 B checkpoint-48000, 같은 데이터 순서, 같은 seed에서 짧게 비교한다.

| Run | Direction | Trajectory | 목적 |
|---|---:|---:|---|
| L0 | 0 | 0 | 기존 forward/hidden/GT baseline |
| L1 | 0.05 | 0 | Direction 단독 효과 |
| L2 | 0 | 1.0, p=0.2 | Trajectory 단독 효과 |
| L3 | 0.05 | 1.0, p=0.2 | 현재 조합 |

각 run을 6K~12K 추가 학습하고 동일 matrix로 평가한다. 현재 장기 B/C 결과만으로는 이 ablation을 대신할 수 없다.

### 8.4 GT weight ablation

현재 loss scale을 기준으로 다음만 우선 비교한다.

```text
lambda_gt = 0.025, 0.05, 0.10
```

`0.5` 또는 `1.0`부터 시도하면 raw GT가 커서 teacher imitation을 압도할 가능성이 높다. 판단 기준은 GT 절대값 하나가 아니라 다음 세 항목이다.

- `student_gt - teacher_gt`
- teacher-student flow/trajectory 거리
- 고정 이미지 품질

### 8.5 Trajectory 범위 ablation

다음 순서로 한 축씩 변경한다.

```text
trajectory_probability:   0.10, 0.20, 0.30
trajectory_rollout_steps: 1, 2, 4
trajectory_warmup_steps:  1K, 5K
```

우선 probability 0.2를 고정하고 rollout 1/2/4를 비교하는 것이 좋다. Rollout을 늘리면 분포 이탈 학습은 강해지지만 계산량과 student의 초기 잘못된 상태 노출도 함께 증가한다.

### 8.6 CFG 실험

고정 checkpoint에서 CFG 1/2/4를 먼저 평가한다.

- 1.0은 정상이고 4.0만 붕괴: `unconditional_probability = 0.10, 0.20` 비교 및 paired conditional/unconditional KD 검토
- 1.0부터 흐림/붕괴: flow imitation, timestep coverage, architecture capacity, trajectory 누적 오차를 우선 점검

### 8.7 Timestep 진단과 sampling 실험

Loss를 timestep decile별로 기록한 뒤 취약 구간을 확인한다.

```text
[0.0, 0.1), ... [0.9, 1.0]
각 구간의 forward flow, trajectory flow, direction, GT
```

그다음에만 `logit_normal`, uniform/none, `cosmap` 등의 분포를 비교한다. 평균 loss만 보고 timestep sampler를 바꾸면 원인을 더 섞을 수 있다.

### 8.8 Architecture 최종 선택

B와 C를 다음 조건으로 다시 맞춰 비교한다.

- 같은 T2I 초기 step 또는 같은 initialization 정책
- 같은 신규 데이터 수
- 같은 loss ablation 결과
- 같은 wall-clock 또는 같은 optimizer step
- 같은 evaluation matrix

B가 계속 우세하면 Phase 2/3의 주력 architecture로 확정하고 A는 우선순위를 낮춘다. 그 후 B 구조를 기준으로 정확한 1.0B/0.7B/0.5B budget을 설계한다.

### 8.9 Editing 안정화 실험

1. Phase 2 B를 dev 기준으로 먼저 선택한다.
2. 같은 조건의 Phase 2 C를 짧게 실행해 architecture 차이를 확인한다.
3. T2I replay 10~25%를 editing batch와 섞는 기능을 구현해 forgetting을 비교한다.
4. MagicBrush mask를 이용한 edit-region/보존-region loss를 실험한다.
5. Phase 2가 안정된 뒤 선택 checkpoint에서 Phase 3 B를 시작한다.
6. Phase 3는 session-history benchmark와 독립 multi-reference benchmark를 분리한다.

### 8.10 Few-step 생성은 별도 단계로 분리

Capacity-distilled B가 50-step에서 충분히 안정된 뒤에 NFE 10/20 평가를 한다. Few-step 품질이 부족하면 현재 objective의 가중치만 계속 조정하기보다 consistency/trajectory-consistency/progressive distillation을 별도 Phase로 설계한다.

## 9. 권장 의사결정 기준

다음 조건을 만족하면 T2I B를 Phase 2의 공식 base로 확정할 수 있다.

- 최소 두 milestone에서 C보다 고정 평가가 일관되게 우세
- CFG 1/2/4에서 급격한 붕괴 없음
- NFE 20/50에서 의미·구도·선명도 유지
- 새 loss ablation에서 trajectory 또는 direction의 실질적 이득 확인
- loss 감소가 아니라 이미지 pairwise 평가에서도 개선 확인

Phase 2에서 Phase 3로 넘어가는 조건은 다음과 같다.

- 별도 MagicBrush dev에서 edit 성공과 source 보존 확인
- 6K/12K/18K 중 과적합 전 checkpoint 선택
- editing 후 T2I 고정 prompt 성능이 허용 범위 내에서 유지
- single-edit failure category를 정리하고 multi-reference 입력 경로와 혼동하지 않음

## 10. 최종 권장 진행 순서

```text
1. MagicBrush dev 완성
2. T2I B/C 54K·60K 고정 평가
3. Direction/Trajectory 2×2 ablation
4. B architecture 확정
5. Edit-B 6K·12K·18K dev 평가
6. GT 0.025/0.05/0.10 및 CFG 실험
7. T2I replay + mask preservation 실험
8. Phase 2 checkpoint 확정
9. Phase 3 B multi-reference 장기 실험
10. 필요할 때만 few-step distillation 별도 진행
```

현재 단계에서 가장 중요한 것은 무작정 step을 늘리는 것이 아니라, **B/C와 각 loss의 효과를 같은 validation 조건에서 분리해 측정하는 것**이다. 학습 파이프라인이 동작한다는 증거는 충분하지만, 최종 architecture와 objective가 확정됐다는 증거는 아직 부족하다.

## 참고 문헌

- [DAgger: A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html) — student가 방문하는 상태에서 supervision을 제공한다는 개념적 배경. 현재 구현은 DAgger 자체는 아니다.
- [BK-SDM: A Lightweight, Fast, and Cheap Version of Stable Diffusion](https://arxiv.org/abs/2305.15798) — block 제거 기반 diffusion model compression과 output/feature distillation 사례.
- [Imagine Flash: Accelerating Emu Diffusion Models with Backward Distillation](https://arxiv.org/abs/2405.05224) — student trajectory 분포에서의 distillation 관련 참고.
- [DMD2: Improved Distribution Matching Distillation for Fast Image Synthesis](https://arxiv.org/abs/2405.14867) — few-step distribution matching의 별도 대안. 현재 구현에는 포함되지 않는다.
- [Trajectory Consistency Distillation](https://arxiv.org/abs/2402.19159) — few-step trajectory consistency의 별도 계열. 현재 local trajectory flow KD와 동일한 알고리즘은 아니다.
