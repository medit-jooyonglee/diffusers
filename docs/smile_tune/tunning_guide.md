# Smile Design 튜닝 서버 — 앱 연동 가이드

데스크톱/웹 앱에서 `my/design_tune_server.py`를 호출할 때 알아야 할 것과 **주의사항**을 정리한다.
현재 테스트 기준 스타일은 **`smile_final_smile_v3`** (웃는 정도 + 치아 + 고개 방향).

- 서버 실행 방법과 전체 옵션: [README.md](README.md)
- API 스펙 전체: [api.md](api.md), 또는 서버의 Swagger `/docs`
- 설정별 결과와 권장 조합의 근거: [../final/README.md](../final/README.md), [../final/sample17_report.md](../final/sample17_report.md)

## 1. 서버 실행

```bash
cd /workspace/repo/mydiffusers
HF_HUB_OFFLINE=1 python my/design_tune_server.py --port 8603 --device cuda:0
```

| 항목 | 값 |
| --- | --- |
| 포트 | 8601 이상 사용 (앱 기본값은 8603) |
| 준비 완료 신호 | 로그에 `Running on http://0.0.0.0:8603` (약 1분) |
| 종료 | Ctrl+C |
| 백그라운드 실행 | `HF_HUB_OFFLINE=1 nohup python my/design_tune_server.py --port 8603 --device cuda:0 > logs/server.log 2>&1 &` |
| Swagger | `http://<서버 IP>:8603/docs` |
| 상태 확인 | `curl -s http://127.0.0.1:8603/health` |

`--styles-dir` 기본값이 `my/configs/smile_design`이라 v3 스타일이 자동으로 잡힌다. 추가 옵션은 필요 없다.

**GPU 메모리**: 모델은 첫 추론 요청 때 로드된다. 그래서 `/health`가 200이어도 GPU 추론이 된다는 뜻은 아니다.
로드 후 약 7.7GB를 점유한다.

## 2. 앱이 보낼 기본값

| 항목 | 값 |
| --- | --- |
| 엔드포인트 | `POST /api/v1/infer` |
| 형식 | `multipart/form-data` |
| 필드 | `parameters` (JSON **문자열**) + `images` (파일, layout 순서대로) |
| `style_name` | `smile_final_smile_v3` |
| 응답 | **PNG 바이너리**. 설정은 `X-Design-Tune-Metadata` 응답 헤더(JSON)와 PNG의 `design_tune` 텍스트 청크 |
| 출력 크기 | **입력 첫 번째 이미지와 같은 크기** |
| 소요 시간 | 장당 약 1.7~2.7초 (첫 요청은 모델 로드로 수십 초) |

```
POST /api/v1/infer
  parameters = {"style_name":"smile_final_smile_v3","layout":"patient_only","mode":"preset",
                "preset":"align_whiten","seed":0,
                "mask":{"mode":"inpaint","regions":["mouth"],"dilate":16,"feather":12}}
  images     = <환자 사진>
```

## 3. 쓸 수 있는 설정

### preset

| 분류 | 이름 | 내용 |
| --- | --- | --- |
| 기본 | `natural` | 변화 없음 |
| 미백 | `whitening_natural` / `whitening_only` | 색 50 / 100 |
| 정렬 | `align_snaggle` / `align_only` | 배열 50 / 100 |
| **치아 조합(권장)** | `align_whiten` | 정렬 100 + 미백 75 |
| 웃는 정도 | `smile_closed`(0) / `smile_closed_lip`(25) / `smile_upper`(75) / `smile_full`(100) | 0 = 입 다묾, 75 = 윗니만, 100 = 윗니+아랫니 |
| 웃음 + 치아 | `smile_upper_align_whiten` / `smile_full_align_whiten` | 웃음 + 정렬 100 + 미백 75 |
| 고개 | `turn_left` / `turn_right` / `look_up` / `look_down` | yaw 0/100, pitch 100/0 |
| 조합(실험용) | `pose_teeth_exact` / `cmb4_exact` / `cmb4b_exact` | 고개 + 치아(+웃음)를 한 문장으로 |

### control (슬라이더)

`smile`, `alignment`, `tooth_color`, `head_yaw`, `head_pitch` — 각각 **0 / 25 / 50 / 75 / 100**.
`smile`, `head_yaw`, `head_pitch`는 **50이 "원본 유지"**이고, `alignment`와 `tooth_color`는 **0이 "원본 유지"**다.

```json
{"mode":"anchor","attribute":"tooth_color","control_values":{"tooth_color":75}}
{"mode":"grid","grid":"align_color","control_values":{"alignment":100,"tooth_color":75}}
```

`grid`는 `align_color`(정렬×미백)와 `yaw_pitch`(좌우×상하) 두 가지만 있다.

## 4. 주의사항

### 4.1 마스크를 반드시 용도에 맞게 보낸다 — 가장 흔한 실수

| 편집 | `mask` | 이유 |
| --- | --- | --- |
| 치아(미백/정렬) | `{"mode":"inpaint","regions":["mouth"],"dilate":16,"feather":12}` | 입만 다시 그리고 나머지는 원본 픽셀 유지 |
| 웃는 정도 | `{"mode":"inpaint","regions":["mouth","cheeks"],"dilate":24,"feather":16}` | 입꼬리·볼이 움직여야 함 |
| **고개 방향** | **`false`** | 머리 전체를 다시 그려야 함 |

**고개 방향 요청에 입 마스크를 보내면 입만 다시 그려져서 고개가 전혀 돌아가지 않는다.** 에러도 나지 않으므로
"설정이 안 먹는다"로 오해하기 쉽다. `mask`를 생략하면 서버 기본값(마스크 꺼짐)이 적용된다.

### 4.2 고개 + 치아는 2단계로 보낸다

한 번에 처리하면 효과가 약해진다. **고개를 먼저 바꾸고, 그 결과 이미지를 다시 보내 치아를 편집**한다.

```
1단계: images=[환자]        preset=turn_right,    mask=false          → 결과 A
2단계: images=[결과 A]      preset=align_whiten,  mask=입 영역        → 최종
```

순서를 반대로 하면(치아 → 고개) 고개 단계에서 치아가 다시 그려져 효과가 절반쯤 사라진다
(실측: 치아 노란기 b\* 2.6 → 5.0, 올바른 순서는 0.2). 근거는 [sample17_report.md](../final/sample17_report.md) 2장.

한 번에 꼭 해야 한다면 `mode: "delta"`가 아니라 **조합 preset**(`pose_teeth_exact`)을 쓴다.
`delta`는 축이 늘어나면 고개 회전이 사라진다([../final/embedding_modes.md](../final/embedding_modes.md)).

### 4.3 요청은 한 번에 하나씩 처리된다

서버에 GPU 락이 있어서 **동시에 보내도 순서대로 처리**된다. 에러는 나지 않지만 그만큼 기다린다.
앱에서 여러 장을 돌릴 때는 순차 호출하고, 진행 표시를 두는 편이 좋다.

### 4.4 layout이 요구하는 이미지 수를 정확히 맞춘다

| layout | 이미지 |
| --- | --- |
| `patient_only` | 환자 사진 1장 |
| `patient_tooth` | 환자 사진 + 치아 템플릿, **정확히 2장** |

수가 다르면 `400 Layout 'patient_tooth' requires 2 images in order ['patient','tooth'], got 1`이 난다.
**템플릿은 권장하지 않는다** — 조합 프롬프트와 결과가 비슷하고, 템플릿의 분홍 입술색이 환자 입술로 번진다
([../final/README.md](../final/README.md) 4.2절).

### 4.5 스타일을 바꾸는 API를 호출하지 않는다

`PUT /api/v1/styles/{name}`, `DELETE /api/v1/styles/{name}`, `POST /api/v1/styles/{name}/build`는
**앱에서 호출하지 않는다.** Swagger 화면에서 실수로 실행하는 경우도 조심한다.

- `PUT`으로 YAML이 바뀌면 임베딩 해시가 달라져서 **이후 모든 추론이 409**가 된다 (재빌드 필요, 약 20초).
- `DELETE`는 스타일을 지운다. 서버를 재시작해도 파일이 돌아오지 않는다.

### 4.6 얼굴이 검출되지 않으면 400

마스크를 쓰는 요청은 MediaPipe로 얼굴을 찾는다. 실패하면 **GPU 추론 전에** 400을 반환한다.
얼굴이 작거나 심하게 기울어진 사진, 입이 가려진 사진에서 발생할 수 있다.
마스크 영역은 `POST /api/v1/mask`로 미리 확인할 수 있다(생성하지 않고 마스크만 반환).

### 4.7 같은 요청은 같은 결과다 (seed)

`seed`를 고정하면 같은 입력·같은 설정에서 항상 같은 이미지가 나온다. 생략하면 매번 달라진다.
**앱에서는 `seed`를 명시**하고, "다시 생성" 기능을 둘 때만 seed를 바꾼다.

### 4.8 입력 이미지

| 항목 | 내용 |
| --- | --- |
| 형식 | PNG / JPEG 등. RGBA도 되고 서버가 RGB로 변환한다 |
| EXIF 회전 | 서버가 자동 보정한다 |
| 장수 | 요청당 최대 5장 |
| 요청 크기 | 최대 64MiB. 넘으면 **413** |
| 내부 해상도 | 약 1MP를 넘으면 비율을 유지한 채 줄여서 생성하고, **출력은 원본 크기로 되돌린다** |

### 4.9 에러 코드

| 코드 | 의미 | 앱에서 할 일 |
| --- | --- | --- |
| 400 | 파라미터 오류, 이미지 수 불일치, **얼굴 미검출** | 메시지를 그대로 보여 준다 |
| 404 | 없는 스타일 이름 | `style_name` 확인 |
| 409 | 임베딩 미빌드 또는 모델 불일치 | 운영자에게 알린다 (앱에서 빌드 호출 금지) |
| 413 | 요청이 64MiB 초과 | 이미지를 줄여 재전송 |
| 500 | 추론 실패 | 재시도 후에도 나면 서버 로그 확인 |
| 507 | GPU 메모리 부족 | 잠시 후 재시도 |

## 5. 요청 예시

치아 미백·정렬 (가장 많이 쓸 형태):
```json
{"style_name":"smile_final_smile_v3","layout":"patient_only","mode":"preset","preset":"align_whiten","seed":0,
 "mask":{"mode":"inpaint","regions":["mouth"],"dilate":16,"feather":12}}
```

미백만 / 정렬만:
```json
{"style_name":"smile_final_smile_v3","layout":"patient_only","mode":"preset","preset":"whitening_only","seed":0,
 "mask":{"mode":"inpaint","regions":["mouth"],"dilate":16,"feather":12}}
```

슬라이더로 세기 조절:
```json
{"style_name":"smile_final_smile_v3","layout":"patient_only","mode":"grid","grid":"align_color",
 "control_values":{"alignment":100,"tooth_color":50},"seed":0,
 "mask":{"mode":"inpaint","regions":["mouth"],"dilate":16,"feather":12}}
```

웃는 정도 (입 다물기):
```json
{"style_name":"smile_final_smile_v3","layout":"patient_only","mode":"preset","preset":"smile_closed","seed":0,
 "mask":{"mode":"inpaint","regions":["mouth","cheeks"],"dilate":24,"feather":16}}
```

고개 방향 (마스크 끄기):
```json
{"style_name":"smile_final_smile_v3","layout":"patient_only","mode":"preset","preset":"turn_right","seed":0,"mask":false}
```

## 6. 현재 알려진 한계

| 항목 | 내용 |
| --- | --- |
| 웃음 75(윗니만) | 이미 아랫니가 보이는 얼굴에서 거의 동작하지 않는다 (합성 얼굴 112장 중 19%) |
| 웃음 100(아랫니까지) | 절반 정도만 동작한다 (42~45%) |
| 웃음 0 | 잘 동작하지만 아래 얼굴을 크게 다시 그려 **인상이 달라질 수 있다** |
| 고개 방향 | 배경·머리카락도 함께 바뀐다. 근접 촬영은 **구도가 넓어진다** |
| 극단값 | 고개 0/100은 과하다. **25/75가 자연스럽다** |
| `natural` | 변화가 없어야 하는데 치아에 얼룩이 생긴 사례가 있다. 원본을 그대로 보여 줘야 하면 **추론하지 말고 원본을 쓴다** |
| 금속 보철 | 완전히 지워지지 않는다 |

## 7. 스타일 전환 (참고)

`smile_final_smile_v3`는 **실험용**이다. 배포용으로 확정된 스타일은 `smile_final`(웃는 정도 없음)이고,
AWS 배포 설정도 그쪽을 참조한다. 앱에서 스타일을 바꿀 수 있게 하려면 `style_name`만 교체하면 된다.
단, `smile_final`에는 `smile_*` preset이 없고 `patient_tooth` 임베딩도 빌드돼 있지 않다.
