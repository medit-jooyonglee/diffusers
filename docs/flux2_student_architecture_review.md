# FLUX.2 Klein 4B student DiT 구조 리뷰

## 결론

`build_student()`는 teacher의 폭(width), attention head, embedding 차원, 입출력 형식을 유지하고
double-stream 및 single-stream block의 **깊이만 줄이는 capacity distillation baseline**이다.
기본 설정에서는 FLUX.2 Klein base 4B DiT를 약 3.876B에서 1.054B 파라미터로 줄인다.

구조적으로는 teacher weight를 그대로 복사할 수 있고 기존 FLUX.2 pipeline과 호환되므로 타당하다.
다만 single-stream block을 20개에서 3개로 줄이는 비율이 특히 크고, 보존할 layer를 중요도 평가 없이
균등 간격으로 선택하므로 "최적화된 압축 구조"라기보다는 검증이 필요한 공격적인 baseline으로 보는 것이 맞다.

## 전체 구조 비교

| 항목 | Teacher: Klein base 4B | Student 기본값 | 변경 방식 | 평가 |
|---|---:|---:|---|---|
| 전체 DiT 파라미터 | 3,875,544,576 (3.876B) | 1,053,820,672 (1.054B) | 27.19%로 축소 | 압축률이 높음 |
| 공통 모듈 파라미터 | 195,035,136 | 195,035,136 | 전부 유지 및 복사 | 입출력 호환성 유지 |
| double-stream block | 5개 | 2개 | 40% 유지 | teacher의 첫/마지막 block 사용 |
| single-stream block | 20개 | 3개 | 15% 유지 | 가장 공격적으로 축소 |
| hidden width (`inner_dim`) | 3,072 | 3,072 | 유지 | width distillation은 아님 |
| attention heads | 24 | 24 | 유지 | head pruning 없음 |
| head dimension | 128 | 128 | 유지 | block weight 직접 복사 가능 |
| image input channels | 128 | 128 | 유지 | 기존 latent 형식과 호환 |
| text condition dimension | 7,680 | 7,680 | 유지 | 기존 Qwen embedding과 호환 |
| MLP ratio | 3.0 | 3.0 | 유지 | FFN width 축소 없음 |
| guidance embedding | `false` | `false` | teacher config 상속 | base CFG 동작 유지 |
| RoPE 설정 | teacher 설정 | 동일 | teacher config 상속 | position encoding 호환 |

파라미터 수는 로컬 `FLUX.2-klein-base-4B` Diffusers transformer weight의 tensor shape을
기준으로 계산했다. Qwen3 text encoder와 VAE는 이 수치에 포함되지 않는다.

## DiT 모듈별 변경점

| DiT 모듈 | Teacher | Student | 초기화 | 변경 여부 |
|---|---|---|---|---|
| `pos_embed` | RoPE, 학습 파라미터 없음 | 동일 config로 생성 | 별도 weight 없음 | 유지 |
| `time_guidance_embed` | timestep embedding | 동일 구조 | teacher weight 전체 복사 | 유지 |
| `double_stream_modulation_img` | image stream shift/scale/gate | 동일 구조 | teacher weight 전체 복사 | 유지 |
| `double_stream_modulation_txt` | text stream shift/scale/gate | 동일 구조 | teacher weight 전체 복사 | 유지 |
| `single_stream_modulation` | joint stream shift/scale/gate | 동일 구조 | teacher weight 전체 복사 | 유지 |
| `x_embedder` | 128 → 3,072 | 동일 구조 | teacher weight 전체 복사 | 유지 |
| `context_embedder` | 7,680 → 3,072 | 동일 구조 | teacher weight 전체 복사 | 유지 |
| `transformer_blocks` | double-stream 5개 | 2개 | teacher block `[0, 4]` 복사 | 깊이 60% 제거 |
| `single_transformer_blocks` | single-stream 20개 | 3개 | teacher block `[0, 10, 19]` 복사 | 깊이 85% 제거 |
| `norm_out` | adaptive output norm | 동일 구조 | teacher weight 전체 복사 | 유지 |
| `proj_out` | 3,072 → 128 | 동일 구조 | teacher weight 전체 복사 | 유지 |

각 double-stream block은 245,367,296개, 각 single-stream block은 122,683,648개의
파라미터를 가진다. 따라서 기본 student의 구성은 다음과 같다.

```text
shared             195,035,136
2 × double block   490,734,592
3 × single block   368,050,944
--------------------------------
student total    1,053,820,672
```

## Layer mapping 방식

보존할 teacher block은 다음 식으로 균등 선택한다.

```text
round(student_index × (teacher_count - 1) / (student_count - 1))
```

기본 설정의 실제 mapping은 다음과 같다.

| Stream | Student block | 복사되는 teacher block |
|---|---:|---:|
| double | 0 | 0 |
| double | 1 | 4 |
| single | 0 | 0 |
| single | 1 | 10 |
| single | 2 | 19 |

이 방식은 시작과 끝 layer를 보존하고 전체 깊이에서 layer를 고르게 가져온다는 장점이 있다.
하지만 layer별 중요도, activation 유사도, loss sensitivity는 반영하지 않는다. 선택되지 않은 block의
weight를 병합하거나 평균내지도 않는다.

`--random_init`을 사용하지 않으면 student의 모든 학습 파라미터는 공통 모듈 또는 선택된 block의
teacher weight로 덮어써진다. `--random_init`을 사용하면 이 복사를 생략하고 전체 student를 무작위
초기화한다.

## 너무 단순하게 줄인 것인가?

깊이만 줄이는 것 자체가 잘못된 것은 아니다. 다음 이유로 첫 실험용 baseline으로는 합리적이다.

| 장점 | 설명 |
|---|---|
| Weight 재사용 | width와 tensor shape이 같아 선택된 teacher block을 정확히 복사할 수 있다. |
| Pipeline 호환 | latent, Qwen embedding, output projection의 shape이 변하지 않는다. |
| 구현 안정성 | 별도 projection adapter나 checkpoint 변환이 필요 없다. |
| 실제 distillation | 학습 시 ground-truth flow loss와 teacher prediction KD loss를 함께 사용한다. |
| 추론 절감 | 반복 block 수가 25개에서 5개로 줄어 DiT 연산량과 메모리가 크게 감소한다. |

다만 다음 한계 때문에 기본 `2 + 3` 구성을 곧바로 적정 구조라고 단정하기는 어렵다.

| 한계 | 영향 |
|---|---|
| Single-stream 20→3 | joint text/image representation 처리 깊이가 85% 줄어 품질 손실 위험이 크다. |
| 균등 mapping 휴리스틱 | 실제로 중요한 layer가 제거될 수 있다. |
| Output-only KD | hidden state, attention map, 중간 feature를 직접 맞추지 않는다. |
| Layer 병합 없음 | 제거된 layer의 정보를 surviving layer 초기값에 흡수하지 않는다. |
| Width/head 유지 | 구조는 단순하지만 block 하나의 비용은 teacher와 동일하다. |
| 공통 모듈 유지 | 195M 공통 파라미터는 줄지 않아 작은 student일수록 고정 비용 비율이 커진다. |
| Pipeline 전체는 별도 | Qwen3-4B와 VAE는 그대로라 전체 pipeline VRAM은 DiT 파라미터 비율만큼 줄지 않는다. |

## 권장 비교 실험

`2 + 3`만 평가하지 말고 동일한 데이터, seed, 학습 step에서 최소 두 개의 중간 깊이와 비교하는 것이 좋다.

| 구성 (`double + single`) | 예상 DiT 파라미터 | Teacher 대비 | 용도 |
|---|---:|---:|---|
| `2 + 3` | 1.054B | 27.2% | 현재의 공격적 1B target |
| `2 + 5` | 1.299B | 33.5% | single-stream 깊이 민감도 확인 |
| `3 + 6` | 1.667B | 43.0% | 품질/속도 중간 기준점 |
| `3 + 8` | 1.913B | 49.3% | 약 2B급 비교군 |
| `5 + 20` | 3.876B | 100% | teacher 기준선 |

특히 `2 + 3`과 `2 + 5` 비교는 double-stream 수를 고정한 채 single-stream 두 개의 효과를 확인할 수 있어
현재 구조가 지나치게 얕은지 판단하기 좋다. 이후 필요하면 균등 mapping 외에 연속 구간 선택, layer
importance 기반 선택, 인접 teacher layer 평균 초기화, intermediate feature loss를 각각 독립적인 ablation으로
추가할 수 있다.

## 코드 근거

- student config 변경: `my/mydisillation.py`의 `build_student()`
- 공통 모듈 및 block 복사: `initialize_student_from_teacher()`
- 균등 mapping: `evenly_spaced_indices()`
- teacher/student loss: `loss_gt + loss_kd`
- FLUX.2 DiT 모듈 정의: `src/diffusers/models/transformers/transformer_flux2.py`

## 반영한 1B v2 구조

기존 1B 파라미터 예산을 유지하면서 single-stream 표현 깊이를 늘리기 위해 block 구성을 재배치했다.
double-stream block 하나의 파라미터 수가 single-stream block 두 개와 거의 같으므로, `2 + 3`을
`1 + 5`로 바꿔 총 파라미터 수는 약 1.054B로 동일하다.

| 항목 | 기존 1B | 1B v2 | 변경 목적 |
|---|---:|---:|---|
| Double-stream blocks | 2 / 5 | 1 / 5 | 제한된 예산을 joint 처리 깊이에 재배치 |
| Single-stream blocks | 3 / 20 | 5 / 20 | text/image 결합 표현의 과도한 축소 완화 |
| 총 반복 block 수 | 5 | 6 | 유효 네트워크 깊이 증가 |
| Teacher double mapping | `[0, 4]` | `[2]` | 단일 block이 초기/후기 처리 사이의 중간점 담당 |
| Teacher single mapping | `[0, 10, 19]` | `[0, 5, 10, 14, 19]` | 전체 single-stream 깊이를 더 촘촘히 표본화 |
| DiT 파라미터 | 1.054B | 1.054B | 1B 추론 예산 유지 |

구조 변경만으로 checkpoint-270000의 품질 문제를 모두 설명할 수는 없다. 기존 실행에는 구조 외에도
학습과 추론 사이에 다음 차이가 있었다.

| 항목 | 기존 실행 | 1B v2 기본값 | 영향 |
|---|---:|---:|---|
| Training resolution | 512 | 512 | 동일 |
| Validation resolution | 기본 pipeline 크기(관측 결과 1024) | 512 | 학습 분포 밖 해상도 검증 방지 |
| Empty-prompt 학습 | 없음 | 10% conditioning dropout | CFG unconditional prediction 학습 |
| Validation CFG | 4.0 | 4.0 | 비교 조건 유지 |
| Learning rate | `1e-4` | `2e-5` | teacher 초기 weight의 급격한 훼손 완화 |
| Ground-truth/KD 비중 | `1.0 / 1.0` | `0.25 / 1.0` | teacher 출력 보존을 우선 |
| Timestep sampling | uniform | logit-normal | 극단 timestep 편중 완화 |
| Warmup | 500 | 2000 | 초기 최적화 안정화 |

설정은 `my/student_configure.yaml`에서 관리한다. 명령행에 같은 옵션을 지정하면 YAML보다 명령행 값이
우선한다. 실행 시 실제 적용된 설정은 output directory의 `student_configure.resolved.yaml`에 저장된다.

기존 `2 + 3` checkpoint는 새 `1 + 5` 모델과 구조가 다르므로 resume할 수 없다. 학습 코드는 checkpoint의
`transformer/config.json`을 검사해 이 경우를 즉시 거부한다. 새 output directory에서 처음부터 학습해야 한다.
