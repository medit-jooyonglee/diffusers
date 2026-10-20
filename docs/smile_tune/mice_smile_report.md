# MICE 효과 검증 / 스마일 정도 제어 실험

서버 설정은 배포 후보(`--fp8-compute --compile regional`), 모든 호출은 실제 HTTP API. seed 0 기준.
지표: 신원 = ArcFace(InsightFace buffalo_l) 코사인 유사도, 스마일 = MediaPipe Face Landmarker blendshape
(`mouthSmileLeft/Right` 평균, 0~1), 픽셀 변화 = 영역별 평균 절대 차이(0-255).

## 요약

- **MICE는 없애도 된다.** 같은 프롬프트·같은 영역으로 비교했을 때 MICE만의 이점이 없었다 — 어려운 케이스에서 효과가 가장
  약하고, 템플릿을 따라가는 정도가 seed를 바꾼 정도보다 작으며, 마스크 합성(composite) 없이는 얼굴 바깥까지 바꾼다.
  1024²에서 35% 느리고 compile/AOT도 따로 필요하다. 템플릿 치아 교체는 inpaint로 대체한다.
- **스마일 제어를 실험용 스타일 `smile_final_smile.yaml`에 추가했다.** 0(무표정)과 75~100(크게 웃기)은 동작하고,
  50(유지)은 기존 smile_final과 같다. 25(덜 웃기)는 무표정 얼굴에서 오히려 웃게 만들어 다시 설계해야 한다.
  아래 얼굴 마스크의 크기를 얼굴 크기에 맞추는 문제도 남아 있다 → **아직 smile_final에 합치지 않았다.**
- 첫 버전(v1)은 "유지" 문장에 `smile`이라는 단어가 들어가 무표정 얼굴을 웃게 만들었다 — 프롬프트에 단어가 나오기만 해도
  그 속성이 생긴다. v2에서 고쳤다.

## 1. MICE 효과 검증

### 1.1 조건

`patient_tooth`(환자 + 치아 템플릿), 환자 5명, 템플릿 2종(`matmatAlpa.png` 실사, `teeeth1.png` 선화), seed 0/1.
얼굴 변형을 최소로 하도록 모든 방식이 **같은 영역**(`inner_mouth`, dilate 4, feather 6)을 쓴다.

| 방식 | 프롬프트 | 요청 |
| --- | --- | --- |
| `mice` | `teeth_from_template` | `mask.mode: mice`, `composite: true`, `edits: [{prompt: teeth_from_template, region: [inner_mouth]}]` |
| `mice_raw` | 같음 | 위에서 `composite: false` (MICE 자체가 영역을 지키는지) |
| `inpaint_edits` | 같음 | `mask.mode: inpaint`, 같은 `edits` |
| `inpaint_preset` | `align_whiten` (참고: 현재 기본 경로) | `mode: preset`, `mask.mode: inpaint` |

영역을 제한하는 방법은 세 가지이고, 작동하는 단계가 다르다.

| 메커니즘 | 작동 단계 | 하는 일 |
| --- | --- | --- |
| MICE attention bias | transformer attention 점수 | 토큰 쌍마다 허용(0) / 차단 / 감쇠 값을 attention 점수에 더해, 영역 지시문 토큰이 자기 영역 이미지 토큰하고만 주고받게 한다. **이미지 latent는 전체가 새로 생성된다** |
| inpaint latent mask | 매 denoising step 직후 latent | `latents = (1 - mask) × 원본 latent(현재 노이즈 수준) + mask × 생성 latent` (`pipeline_flux2_klein_inpaint.py`). 마스크 밖을 매 step 원본으로 대체해 고정하고 마스크 안만 생성한다 |
| composite | 생성·디코딩이 끝난 RGB 이미지 | `Image.composite(생성, 원본, 마스크)` — 마스크 안은 생성 결과, 밖은 원본 픽셀로 붙여 넣는다. 서버는 inpaint 결과에도 한 번 더 적용한다(크기 조정 오차 제거) |

네 방식의 차이:

| 방식 | 파이프라인 | 텍스트 임베딩 | 영역 제한 | 템플릿(image 2) |
| --- | --- | --- | --- | --- |
| `mice` | 일반 생성 (`Flux2KleinPipeline`) | 전역 지시문("같은 사람, 표정 유지 …") + 영역 지시문 `teeth_from_template`을 이어 붙인 것 | attention bias → 생성 후 RGB composite | 참조 토큰 (모든 토큰이 참조 가능) |
| `mice_raw` | 같음 | 같음 | attention bias만 (전체 생성 이미지를 그대로 반환) | 같음 |
| `inpaint_edits` | inpaint (`Flux2KleinInpaintPipeline`) | 영역 지시문 `teeth_from_template` 하나 (MICE와 같은 지시문, 전역 지시문 없음) | latent mask → RGB composite | `image_reference` |
| `inpaint_preset` | inpaint | preset 문장 `align_whiten` (base 프롬프트 + 정렬·미백 + "Use image 2 only as a reference for tooth shape …") | latent mask → RGB composite | `image_reference` |

- `mice` vs `mice_raw`: 차이는 마지막 RGB composite뿐이다 → MICE의 얼굴 보존이 사후 붙여넣기에 의존하는지 확인용.
- `mice` vs `inpaint_edits`: 같은 지시문을 attention으로 가두느냐, latent 단계에서 마스크 밖을 고정하느냐의 비교.
- `inpaint_edits` vs `inpaint_preset`: 같은 inpaint에서 프롬프트만 다르다 — 복사 지시("image 2의 치아로 바꿔라") vs 목표 문장(정렬·미백, image 2는 모양 참고).

### 1.2 결과

![선화 템플릿](assets/mice_vs_inpaint_line_template.jpg)
![실사 템플릿](assets/mice_vs_inpaint_photo_template.jpg)

환자 5명 × 템플릿 2종 평균 (신원은 ArcFace가 원본에서 얼굴을 찾은 3명 기준):

| 방식 | 신원 | 편집 영역 밖 변화 | 입술 변화 | 입안 변화 | 템플릿 영향 / seed 영향 | 1024² 시간 |
| --- | --- | --- | --- | --- | --- | --- |
| `mice` | 0.933 | 0.003 | 7.1 | 43.7 | 19.7 / 25.1 = **0.79** | 3.52초 |
| `mice_raw` | 0.910 | **12.7** | 19.5 | 47.4 | 22.2 / 28.3 = 0.78 | 3.63초 |
| `inpaint_edits` | 0.925 | 0.000 | 6.1 | 51.4 | 27.3 / 20.4 = **1.34** | 2.62초 |
| `inpaint_preset` | **0.941** | 0.000 | **4.0** | 45.4 | 15.5 / 18.0 = 0.86 | 2.60초 |

- **템플릿 영향 / seed 영향**: 템플릿만 바꿨을 때 입안이 바뀐 정도 ÷ seed만 바꿨을 때 바뀐 정도. 1보다 작으면 템플릿 차이가
  랜덤 seed 차이에 묻힌다 = 템플릿을 따르지 않는다.
- **MICE는 템플릿을 거의 따르지 않는다**(0.79). 같은 프롬프트를 inpaint로 주면 템플릿을 가장 많이 따른다(1.34).
- **어려운 케이스에서 MICE가 가장 약하다**: sample17(치아 틈), teeth01(누런 치아)은 두 템플릿 모두 거의 그대로다
  (그림 1·2행). inpaint 두 방식은 틈을 메우고 색을 바꾼다.
- **MICE는 마스크 합성 없이는 얼굴을 지키지 못한다**: `mice_raw`는 편집 영역 밖이 평균 12.7 바뀌고 입술도 19.5 바뀐다.
  "attention bias로 영역을 제한한다"는 MICE의 목적이 실제로는 composite에 의존한다.
- 신원과 얼굴 바깥 보존은 composite를 쓰면 방식 간 차이가 작다.
- **선화 템플릿 + inpaint_edits는 치아에 검은 윤곽선이 그대로 그려진다**(teeth04, sample17, teeth01). 템플릿을 강하게 따르는
  부작용이다. 실사 템플릿에서는 깨끗하다 → 템플릿은 실사 이미지만 쓰는 것을 권장.

### 1.3 결론: MICE 제거 가능

| | MICE | inpaint (대체) |
| --- | --- | --- |
| 템플릿 따르기 | 약함 (seed 차이보다 작음) | `edits`로 강하게, preset으로 적당히 |
| 어려운 케이스 | 거의 안 바뀜 | 고쳐짐 |
| 얼굴 보존 | composite 필수 | 기본으로 보존 (latent mask) |
| 여러 영역 조합 | 가능 | `edits`로 가능 (치아+입술+눈 검증됨, [robustness_report.md](robustness_report.md) 4장) |
| 속도 (1024²) | 3.5초 | 2.6초 |
| 배포 | 별도 compile 그래프 / AOT 패키지, FlashAttention 불가 | 기본 경로 하나 |

MICE 요청을 쓰지 않으면 AOT 패키지 하나, warmup 한 번으로 끝난다. 서버에서 MICE를 끄는 옵션(요청 시 `400`)과
YAML의 `mice` 섹션 정리는 따로 정하면 된다(inpaint `edits`도 `mice.regions` 지시문을 쓰므로 섹션 자체는 남겨야 한다).

## 2. 스마일 정도 제어

### 2.1 추가한 것

`smile_final.yaml`에는 스마일 제어가 없었다(오히려 "표정·입 벌림·입술을 그대로 유지" 문장이 고정). 검증된 배포 스타일을
건드리지 않기 위해 **실험용 사본 `my/configs/smile_design/smile_final_smile.yaml`**에 추가했다.

- `smile` control: head_yaw처럼 **50 = 원본 유지**, 0 = 무표정(입 다묾), 25 = 덜 웃게, 75 = 더 웃게, 100 = 크게 웃게.
- preset: `smile_neutral`(0), `smile_soft`(25), `smile_more`(75), `smile_big`(100). 다른 control과 `delta`/`anchor` 모드로 조합 가능.
- 기본 문장 `Keep the facial expression, mouth opening, lips, lighting, clothing, and background exactly as in image 1.`을
  `{smile} Keep the lighting, clothing, and background exactly as in image 1.`로 바꾸고, 50의 문장을
  `Keep the facial expression, mouth opening, and lips exactly as in image 1.`로 했다 (기본 프롬프트가 기존과 거의 같다).
- 마스크: 입꼬리와 볼이 움직여야 하므로 치아용 `inner_mouth`로는 안 된다. 아래 얼굴(`mouth` + `cheeks`, dilate 24, feather 16)을 inpaint했다.

### 2.2 v1의 실패: "유지" 문장의 `smile` 단어

v1은 50을 `Smile: keep the facial expression, smile, mouth opening, and lips exactly as in image 1`로 썼다.
그 결과 **원본 유지(natural)인데도 무표정 얼굴이 웃는 얼굴로 바뀌었다**:

| 무표정 사진, natural, 아래 얼굴 마스크 | 기존 smile_final | v1 |
| --- | --- | --- |
| sample13 스마일 | 0.001 | **0.571** |
| 20160426… 스마일 | 0.033 | **0.857** |

![v1 회귀](assets/smile_control_v1_regression.jpg)

v2에서 50 문장에서 `smile`을 빼고 `Smile:` 라벨도 없앴다.

### 2.3 v2 결과

![무표정 얼굴](assets/smile_control_neutral_faces.jpg)
![웃는 얼굴](assets/smile_control_smiling_faces.jpg)

스마일 점수 (원본 → 기존 smile_final natural → v2 0 / 25 / 50 / 75 / 100):

| 환자 | 원본 | 기존 유지 | 0 | 25 | 50 | 75 | 100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20160426… (무표정) | 0.03 | 0.03 | 0.00 | 0.71 | 0.03 | 0.91 | 0.92 |
| sample13 (무표정) | 0.00 | 0.00 | 0.00 | 0.44 | 0.00 | 0.59 | 0.57 |
| sample11 (무표정) | 0.00 | 0.33 | 0.00 | 0.40 | 0.34 | 0.45 | 0.44 |
| sample26 (약간) | 0.43 | 0.34 | 0.00 | 0.10 | 0.28 | 0.20 | 0.59 |
| sample18 (웃음) | 0.91 | 0.88 | 0.00 | 0.40 | 0.86 | 0.93 | 0.94 |
| teeth04 (웃음) | 0.88 | 0.90 | 0.12 | 0.73 | 0.90 | 0.96 | 0.96 |
| sample17 (웃음) | 0.73 | 0.69 | 0.00 | 0.37 | 0.72 | 0.55 | 0.71 |

- **50(유지)은 기존 smile_final과 같다** — 스마일 점수와 신원(20160426 0.97 vs 0.98, sample18 0.97 vs 0.97, teeth04 0.96 vs 0.98) 모두.
  치아 preset(`align_whiten`) 회귀도 입안 차이 0.5~4.7로 작다(sample17만 12).
- **0(무표정)은 9명 모두 입을 다문 무표정으로 안정적이다.**
- **75/100은 무표정 얼굴을 웃게 만든다**(20160426 0.03 → 0.92, sample13 0 → 0.59). 이미 크게 웃는 얼굴은 50과 거의 같다(더 웃을 여지가 없음).
  75와 100의 차이는 작다.
- **25(덜 웃게)는 쓸 수 없다**: 웃는 얼굴에서는 줄어들지만(sample18 0.91 → 0.40), 무표정 얼굴에서는 오히려 웃게 만든다
  (20160426 0.03 → 0.71). 웃지 않는 얼굴에 "smaller smile"은 정의가 모호하고, `smile` 단어가 웃음을 유도한다.
- sample11은 기존 smile_final에서도 유지(50)가 0.33으로 살짝 웃게 된다 — 스마일 control과 무관한 기존 동작.

### 2.4 마스크 범위

| 마스크 | 신원 (무표정 4명 중 측정 가능한 2명, 50 기준) | 표정 변화 | 문제 |
| --- | --- | --- | --- |
| 아래 얼굴 (`mouth`+`cheeks`) | 0.66~0.97 (기존과 동일) | 입·볼만 | 작은 이미지에서 얼굴이 일그러짐, 볼 경계 음영 |
| 얼굴 전체 (`face`) | 0.43~0.65 | 자연스러움 | 신원이 크게 바뀜 |
| 마스크 없음 | 0.43~0.75 (웃는 얼굴 세트에서는 yellowteeth 0.26까지) | 가장 자연스러움 | 신원이 크게 바뀜 (다른 사람처럼 보임) |

![마스크 없음](assets/smile_control_nomask.jpg)

- **아래 얼굴 마스크가 신원 보존에 가장 낫다.** 마스크 없음이나 얼굴 전체는 표정은 자연스럽지만 다른 사람처럼 보인다.
- **teeth01(303×346)은 아래 얼굴 마스크에서 얼굴이 크게 일그러졌다** — 기존 smile_final에서도 같다. dilate가 24px로 고정이라
  작은 얼굴에서는 마스크가 얼굴 대부분을 덮는다 → dilate/feather를 **얼굴 폭 비율**로 정해야 한다.
- yellowteeth는 볼 경계에 어색한 음영이 생겼다.

### 2.5 결론과 다음 단계

스마일 제어는 **"무표정으로(0)"와 "더 웃게(75~100)"는 쓸 수 있는 수준**이지만 바로 배포 스타일에 넣기에는 이르다.

1. 25를 다시 설계: 상대 표현("smaller") 대신, 서버가 입력의 스마일 점수(blendshape)를 재서 낮추기/높이기 문장을 고르거나,
   선택지를 "무표정 / 유지 / 웃게 / 크게 웃게"로 줄인다.
2. 아래 얼굴 마스크의 dilate·feather를 얼굴 폭 비율로 바꾼다(작은 이미지 일그러짐).
3. 치아 편집과 같이 쓰는 흐름 확인: 아래 얼굴을 다시 그리면 치아도 다시 그려지므로, 스마일과 치아 control을 한 요청(`delta`)으로
   합치거나 스마일 → 치아 순서로 두 번 요청하는 방식을 비교.
4. 위가 정리되면 smile_final에 합친다. 기본 프롬프트가 거의 같아 치아 결과는 유지될 것으로 보이지만, 합칠 때 임베딩 재빌드와
   회귀 확인이 필요하다.

## 3. 파일

- 실험용 스타일: `my/configs/smile_design/smile_final_smile.yaml` (template_version `smile-final-smile-v2`)
- 그림: `docs/smile_tune/assets/mice_vs_inpaint_*.jpg`, `smile_control_*.jpg`
- 스크립트(세션 scratchpad, 저장소에 넣지 않음): `mice_smile.py`, `smile2.py`, `smile3.py`, `smile_v2.py`, `metrics.py`
