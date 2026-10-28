# Smile Design 튜닝 서버 — 앱 연동 가이드 (최종)

앱은 **설정만 보낸다.** 마스크, 처리 순서, 단계 수, 조합 방식은 서버가 정한다.

- 전용 엔드포인트: **`POST /api/v1/design`**
- 기준 스타일: **`smile-final-smile-v6`** (`my/configs/smile_design/smile_final_smile_v6.yaml`)
- 조합 방식: **concat** — 미리 등록하지 않은 조합도 즉석에서 만든다 (4절)
- 서버 실행과 옵션: [README.md](README.md) · 전체 API: [api.md](api.md) 또는 서버의 `/docs`
- 설정별 결과의 근거: [../final/README.md](../final/README.md)

---

## 1. 앱에서 바꿀 것 (지시)

기존에 `/api/v1/infer`로 직접 호출하던 코드를 아래대로 고친다.

1. **호출 대상을 `POST /api/v1/design`으로 바꾼다.** `/api/v1/infer`는 유지되지만 앱에서 쓰지 않는다.
2. **`style_name`을 `smile-final-smile-v6`으로 바꾼다.**
3. **`mask`를 보내지 않는다.** 마스크 종류·영역·dilate·feather를 요청에서 모두 제거한다. 서버가 편집 종류에 맞춰 정한다.
4. **`layout`을 보내지 않는다.** 이미지를 1장 보내면 `patient_only`, 2장 보내면 `patient_tooth`로 서버가 판단한다.
5. **`mode`, `preset`, `grid`, `attribute`, `control_values`를 보내지 않는다.** 대신 control 값을 **최상위 필드**로 보낸다
   (`tooth_color`, `alignment`, `restoration`, `head_yaw`, `head_pitch`, `smile`).
6. **`delta`와 `delta_strengths`를 완전히 제거한다.** 가중치 슬라이더가 있으면 UI에서 뺀다.
   측정 결과 효과가 절반 이하다 — 고개+웃음+치아를 `delta`로 합치면 고개가 18.3° → 5.4°로 사라지고 치아 b\*는 2.6 → 11.2가 된다.
7. **단계별로 설정을 보낸다.** 한 요청에 치아·고개·웃음을 함께 보내면 서버가 **치아 → 고개/웃음** 순으로 처리하고 최종 이미지 1장을 돌려준다.
   단계를 앱에서 체이닝하지 않는다.
8. **`seed`를 항상 명시한다.** 생략하면 매번 결과가 달라진다. "다시 생성"에서만 값을 바꾼다.
9. **`num_steps`를 보내지 않는다.** 4 step 고정이다. 올리면 오히려 나빠진다(sample17: b\* 2.6 → 3.1 → 3.7).
10. **고개는 25 / 75만 노출한다.** 0과 100은 변화가 과하다.
11. **요청을 직렬로 보낸다.** 서버에 GPU 락이 있어 동시에 보내도 순서대로 처리된다. 앱에서 큐로 관리하고 진행 표시를 둔다.
12. **응답 헤더 `X-Design-Tune-Metadata`를 읽는다.** `identity_warning`이 `true`면 결과를 **"참고용 시뮬레이션"으로 표시**한다.
13. **`unchanged`가 `true`면** 서버가 추론 없이 원본을 그대로 돌려준 것이다. "변경 없음"으로 처리한다.
14. **스타일 변경 API를 호출하지 않는다.** `PUT`/`DELETE /api/v1/styles/{name}`, `POST …/build`. `PUT`으로 YAML이 바뀌면 이후 모든 추론이 409가 된다.

### 앱에 노출할 설정

| 노출한다 | 노출하지 않는다 |
| --- | --- |
| 치아 미백 `tooth_color` 0/25/50/75/100 | 마스크 종류·영역·dilate·feather |
| 치아 정렬 `alignment` 0/25/50/75/100 | step 수 |
| 치아 복원 `restoration` 0/50/75/100 | 조합 모드(preset/concat/grid/delta) |
| 치아 템플릿 사용 여부 | 가중치(`delta_strengths`) |
| (선택) 고개 `head_yaw` 25/75 | layout |
| (선택) 웃음 `smile` 0/25/75/100 | 처리 순서·단계 수 |
| seed | |

### 치아 스타일 묶음 (앱 프리셋 권장)

v6는 용도별로 스타일을 묶어 두었다. 앱은 이 조합만 노출해도 된다.

| 묶음 | 설정 | 용도 |
| --- | --- | --- |
| **미백 단독** | `tooth_color` 25 / 50 / 75 / 100 | 색만. 단계가 또렷하게 벌어진다 (b\* 12.2 → 4.2 → −0.1 → −0.7) |
| **정렬 단독** | `alignment` 50 / 75 / 100 | 모양만. 색은 거의 그대로 (b\* 11 내외) |
| **통합(기본)** | `alignment` 100 + `tooth_color` 75 | 가장 많이 쓸 조합 |
| **통합(최대)** | `alignment` 100 + `tooth_color` 100 | 가장 희게 |
| **복원** | 위 + `restoration` 100 | 결손·금속이 있는 케이스. **치아 템플릿을 같이 보낼 것** |

---

## 2. `POST /api/v1/design` 사용법

`multipart/form-data`로 `parameters`(JSON 문자열)와 `images`(파일)를 보낸다.
응답은 **PNG 바이너리**, 설정은 `X-Design-Tune-Metadata` 헤더(JSON)와 PNG의 `design_tune` 청크.

### 요청 필드

| 필드 | 필수 | 설명 |
| --- | --- | --- |
| `style_name` | O | `"smile-final-smile-v6"` |
| `tooth_color` | | 0(유지) / 25 / 50 / 75 / 100 |
| `alignment` | | 0(유지) / 25 / 50 / 75 / 100 |
| `restoration` | | 0(없음) / 50(결손 메우기) / 75(금속 제거) / 100(둘 다) |
| `head_yaw` | | 50(유지) / 25 / 75 |
| `head_pitch` | | 50(유지) / 25 / 75 |
| `smile` | | 50(유지) / 0 / 25 / 75 / 100 |
| `seed` | | 기본 0 |
| `open_closed_mouth` | | 기본 `true`. 다문 입 자동 2단계를 끄려면 `false` |
| `images` | O | 환자 사진 1장, 또는 환자 사진 + 치아 템플릿 2장 |

### 예시

**치아만 (가장 많이 쓰는 형태)**
```json
{"style_name":"smile-final-smile-v6","tooth_color":75,"alignment":100,"seed":0}
```

**결손·금속 복원 + 템플릿** — `images`에 환자 사진, 템플릿 순서로 2장
```json
{"style_name":"smile-final-smile-v6","tooth_color":75,"alignment":100,"restoration":100,"seed":0}
```

**고개 + 웃음 + 치아** — 한 요청으로 보내면 서버가 2단계로 처리한다
```json
{"style_name":"smile-final-smile-v6","head_yaw":75,"smile":100,"tooth_color":75,"alignment":100,"seed":0}
```

### 응답 메타데이터

실제 응답 (고개 75 + 웃음 100 + 정렬 100 + 미백 75):

```json
{"style_name":"smile-final-smile-v6",
 "requested":{"smile":100.0,"head_yaw":75.0,"tooth_color":75.0,"alignment":100.0},
 "seed":0,
 "mouth_openness":{"inner_mouth":34.5,"teeth":16.1,"face_width":379.0},
 "closed_mouth":false, "unchanged":false, "identity_warning":true, "warnings":[],
 "stages":[{"stage":"teeth","index":0,"template":false,"seconds":1.9,
            "mix":{"mode":"preset","preset":"teeth_basic"},
            "mask":{"enabled":true,"regions":["inner_mouth"],"dilate":8,"feather":6.0}},
           {"stage":"pose_smile","index":1,"template":false,"seconds":1.83,
            "mix":{"mode":"preset","preset":"yaw75_smile_full"},"mask":{"enabled":false}}],
 "text_encoder_loaded":false}
```

`mix.mode`가 `"concat"`이면 미리 등록된 조합이 없어 서버가 즉석에서 이어 붙인 것이다 (4절).

```json
"mix":{"mode":"concat","values":{"tooth_color":25.0,"alignment":75.0}}
```

| 필드 | 쓰임 |
| --- | --- |
| `unchanged` | `true`면 원본 그대로 반환 (추론 0회, 약 0.2초) |
| `identity_warning` | `true`면 고개·웃음이 포함돼 신원이 떨어진다 → **참고용 표시** |
| `closed_mouth` | 다문 입으로 판정돼 자동 2단계를 거쳤는지 |
| `warnings` | 사람이 읽을 수 있는 경고 문장 |
| `stages` | 몇 단계로 어떤 임베딩을 썼는지 (디버깅용) |

---

## 3. 서버가 자동으로 하는 일

```
바꿀 것이 없음        → 추론하지 않고 원본 반환
치아만               → inpaint 1회
고개/웃음만           → 마스크 없이 1회
고개 + 웃음           → 조합 preset이 있으면 1회, 없으면 concat 1회
치아 + 큰 변화        → 순차 2단계: 치아 inpaint 먼저 → 고개/웃음(마스크 없음)
입이 다물려 있음       → 자동 2단계: 웃음 75로 입을 벌린 뒤 → 치아 inpaint (이때만 치아가 뒤)
```

| 편집 | 마스크 | 근거 |
| --- | --- | --- |
| **치아 (미백·정렬·복원)** | `inpaint`, **`inner_mouth`, dilate 8, feather 6** | `mouth`는 입술까지 다시 그린다 (입술 변화 8.8 vs 4.4) |
| 고개 방향 | **없음** | 머리 전체를 다시 그려야 한다 |
| 웃는 정도 | **없음** | 입 모양이 커지면 마스크를 벗어나 잘린다 (최대 38%) |

**치아를 먼저 두는 이유**: 고개/웃음 단계는 마스크 없이 입까지 다시 그린다. 치아를 나중에 하면
그 단계가 만든 **결손·금속 같은 구조적 결함**이 남은 채로 좁은 마스크 안에서 복구해야 하고, 실제로
복구되지 않는다 (sample17: 검은 틈이 끝까지 남음). 치아를 먼저 메우면 고개 단계가 메워진 치열을
그대로 돌려 그린다.

색(b\*)만 보면 반대 순서가 약간 더 희다 (0.2 vs 4.8). 하지만 **빠진 이가 남은 결과는 쓸 수 없으므로
자연스러움을 기준으로 치아를 먼저 한다.** 근거 이미지: `outputs/order_final/`, `outputs/p10_diag/`.

**예외**: 입을 다문 환자는 치아를 넣을 자리가 없으므로 웃음 단계가 먼저 간다.

**다문 입 판정** (얼굴 폭² 대비 ×1000): `teeth` < 2 또는 `inner_mouth` < 5.
실제 샘플 28장에서 다문 입은 1.6 / 0.1, 벌린 입은 17.5 / 6.9 이상이라 경계가 뚜렷하다.

---

## 4. 조합 방식 — concat

**조합을 미리 등록하지 않아도 된다.** 서버는 이 순서로 고른다.

```
① 값이 정확히 일치하는 조합 preset이 있으면 그것 (가장 정확)
② 없으면 concat — 공통 서문 + 요청한 control 지시문을 토큰 축으로 이어 붙인다
```

### concat이 무엇인가

프롬프트 임베딩은 숫자 하나가 아니라 **512칸짜리 토큰 줄**이다. 문장을 앞에서부터 읽은 기록이라
문장 길이가 달라지면 뒤 내용이 통째로 밀린다. `delta`(덧셈)는 서로 다른 칸끼리 더해서 깨진다 —
**오차의 64~70%가 비어 있어야 할 패딩 자리에 쌓인다.**

concat은 더하지 않고 **나열**한다.

```
delta  : [문장A 512칸] + [문장B 512칸]  → 같은 칸끼리 더함 (칸이 어긋남)
concat : [서문 85칸][정렬 46칸][미백 29칸] → 160칸으로 이어 붙임 (어긋날 일이 없음)
```

토큰 길이는 512 고정이 아니다. 실측으로 **2,316토큰까지 품질 저하 없이** 동작했고, 속도는 +28%,
VRAM은 거의 그대로였다. control 6개를 전부 이어 붙여도 **288토큰**이라 여유가 많다.

### 실측 (sample17 · medit)

| 방식 | 2축 b\* | 3축 b\* / yaw | 4축 b\* / yaw |
| --- | --- | --- | --- |
| preset (조합 문장) | 2.3 / −2.4 | 2.9 / 8.6° | 3.6 / 8.6° |
| **concat** | **2.1 / −0.7** | 3.1 / 9.3° | 4.3 / 4.5° |
| delta | 4.8 / 3.0 | 5.2 / 13.9° | 6.7 / **−1.9°** |

concat은 preset과 대등하고 delta보다 확실히 낫다. 다만 **4축에서 고개가 약해진다** —
고개+웃음 조합은 preset을 등록해 두는 편이 낫다.

### 라이브러리에 필요한 것

concat은 `clause` 엔트리를 쓴다 (`{layout}/clause/identity`, `{layout}/clause/{control}/{값}`).
v6 라이브러리에는 들어 있다. 없으면 서버가 자동으로 grid → anchor로 물러난다.

```bash
python my/build_smile_embedding_library.py \
  --config my/configs/smile_design/smile_final_smile_v6.yaml \
  --output-dir outputs/design_tune_embedding_libraries/smile_final_smile_v6/<config_hash> \
  --device cuda:0 --dtype bf16 --local-files-only
```

### 조합 preset을 추가하고 싶다면

고개와 웃음은 따로 돌리면 뒤 단계가 앞 단계를 되돌린다(실측: 웃음 0.94 → 0.17).
`presets:`에 control 값을 적고 재빌드하면 서버가 **값이 정확히 일치하는 preset을 자동으로 찾아 쓴다.**
이름은 자유롭고 **서버 코드를 고칠 필요가 없다.**

```yaml
  yaw25_smile_upper:
    head_yaw: 25
    smile: 75
```

YAML을 저장하면 임베딩 해시가 바뀌어 **재빌드가 끝날 때까지 추론이 409**가 된다 (약 30~60초).

---

## 5. 서버 실행

```bash
cd /workspace/repo/mydiffusers
HF_HUB_OFFLINE=1 python my/design_tune_server.py --port 8603 --device cuda:0
```

| 항목 | 값 |
| --- | --- |
| 포트 | 8601 이상 (앱 기본 8603) |
| 준비 완료 | 로그에 `Running on http://0.0.0.0:8603` (약 1분) |
| 백그라운드 | `... > logs/server.log 2>&1 &` |
| Swagger | `http://<서버 IP>:8603/docs` |
| 상태 확인 | `curl -s http://127.0.0.1:8603/health` |
| 기본 설정 | fp8 가중치 저장 + bf16 연산. 옵션 불필요 |

모델은 **첫 추론 요청 때** 로드된다. `/health`가 200이어도 GPU 추론 준비 완료를 뜻하지 않는다.

### 에러 코드

| 코드 | 의미 | 앱에서 |
| --- | --- | --- |
| 400 | 파라미터 오류, 이미지 수 불일치, **얼굴 미검출** | 메시지를 그대로 표시 |
| 404 | 없는 스타일 | `style_name` 확인 |
| 409 | 임베딩 미빌드 / 모델 불일치 | 운영자에게 알림 (앱에서 빌드 호출 금지) |
| 413 | 요청 64MiB 초과 | 이미지를 줄여 재전송 |
| 500 | 추론 실패 | 재시도 후에도 나면 서버 로그 확인 |
| 507 | GPU 메모리 부족 | 잠시 후 재시도 |

입력 이미지: PNG/JPEG, RGBA 가능(서버가 RGB 변환), EXIF 회전 자동 보정, 요청당 최대 5장.
약 1MP를 넘으면 줄여서 생성하고 **출력은 원본 크기로 되돌린다.**

---

## 6. 치아 템플릿

템플릿 이미지를 두 번째로 보내면 **치아가 덜 표백되고 질감이 살며, 결손·금속이 복원된다.**

| | 치아 b\* | 결과 |
| --- | --- | --- |
| 템플릿 없음 | 1~3 | 균일한 흰 덩어리(틀니 느낌)가 되기 쉽다 |
| **템플릿 사용** | **3~7** | 치아 하나하나가 구분되고 음영이 산다 |

**b\*가 낮을수록 좋은 것이 아니다.** 1~3은 과하게 표백된 상태다.

**결손·금속이 있으면 템플릿이 사실상 필수다.** sample17(금속 보철 + 앞니 결손)에서
"빠진 이를 메워라", "전체 치열을 재건하라" 같은 문장을 넣어도 **검은 틈이 끝까지 남았다.**
템플릿을 넣는 순간 틈과 금속이 모두 사라졌다. 모델에게 "어떤 치아를 그릴지"를
글이 아니라 **그림으로** 줘야 해결되는 문제다. 근거: `outputs/hard_case_prompts/sheet.jpg`.

템플릿은 **치아 단계에서만** 참조로 쓰인다. 고개·웃음 단계는 환자 사진 1장만 쓴다.

---

## 7. 알려진 한계

| 항목 | 내용 |
| --- | --- |
| **신원 유지** | 치아만 0.89~0.96 / 고개·웃음 포함 0.26~0.62. **참고용 시뮬레이션으로 표시** |
| 웃음 75(윗니만) | 이미 아랫니가 보이는 얼굴에서 거의 동작하지 않는다 (112장 중 19%) |
| 웃음 100 | 절반 정도만 동작 (42~45%) |
| 고개 방향 | 배경·머리카락도 바뀐다. 근접 촬영은 구도가 넓어진다 |
| 조합 preset | 고개가 단독 적용(23.7°)보다 약간 약하다(19.4°) |
| 금속 보철 | `restoration` 75/100 + **치아 템플릿**을 같이 써야 지워진다. 문장만으로는 안 된다 |
| **미백 단독인데 치열이 바뀜** | inpaint가 마스크 안을 새로 그려서다. **프롬프트로 막을 수 없다** (아래) |
| 검증 범위 | 실제 사진 28장 + 합성 얼굴 112장. **실제 임상 사진으로 추가 검증 필요** |

### 미백 단독에서 치열이 바뀌는 문제

`tooth_color`만 올려도 깨진 이가 온전해지고 틈이 메워진다. **사양으로 받아들여야 한다.**

- **원인**: inpaint는 마스크 안을 **새로 그린다.** strength 1.0이면 입 안 영역이 노이즈 100%에서
  시작하므로 사실상 백지에서 다시 그린다. 문장은 "무엇을 그릴지"만 정하고 "다시 그릴지 말지"는 못 정한다.
- **대조군**: "아무것도 바꾸지 마라"만 7줄 쓴 `natural` 프롬프트로 돌려도 치열이 바뀌었다
  (`outputs/whiten_only/sheet.jpg`).
- **프롬프트 5종 실험**: "모양·위치 바꾸지 말고 색만", "포토 필터처럼", "결점을 그대로 둬라" 등을 시도했다.
  구조를 지키면(상관 0.954) 미백이 거의 안 되고(b\* 12.2), 미백이 되면(b\* 0.8) 구조가 깨진다(0.606).
  **동시에 얻을 수 없다** (`outputs/color_only/sheet.jpg`).
- **strength를 낮추는 것도 안 된다**: 4 step 증류 모델이라 조절 단계가 1/2/3/4 step 네 칸뿐이고,
  0.8 이상은 전부 노이즈 100%로 같다. 2 step으로 줄이면 모양은 남지만 품질이 떨어진다.
- **진짜 해법은 생성이 아니다**: 치아 세그멘테이션 + 색 보정이면 모양이 전혀 안 바뀐다.
  다만 MediaPipe 폴리곤·Otsu·밝기 규칙은 경계가 부정확해 쓸 수 없었다
  (`outputs/classic_whiten/sheet_masks.jpg`). **별도 치아 분할 모델이 필요한 과제다.**

---

## 8. 스타일 전환

| 스타일 | 상태 | 비고 |
| --- | --- | --- |
| **`smile-final-smile-v6`** | **현재 기준** | concat용 `clause` 엔트리 포함. `restoration` control 있음 |
| `smile-final-smile-v5` | v6의 직전 | 범위 제한 문구 없음, restoration 문구에 색이 섞임 |
| `smile-final-smile-v3` | 이전 운영본 | control마다 "원본 그대로 두라"가 들어 있어 문장이 길다 |
| `smile_final` | 배포 확정본 | 웃는 정도 없음. 조합 preset·`patient_tooth` 임베딩 미빌드 |

앱에서는 `style_name`만 바꾸면 된다.

**v3 → v6에서 달라진 것**

1. 바꾸지 않는 control은 문장에 **아예 넣지 않는다** (v3는 "keep ... exactly as in image 1"을 전부 적었다).
2. `restoration` control 추가 — 결손 메우기 / 금속 제거.
3. `clause` 엔트리 추가 — concat 조합용.
4. 용도별 스타일 묶음(미백 5 / 정렬 5 / 통합 5)으로 정리.

## 9. 근거 데이터

| 내용 | 위치 |
| --- | --- |
| 조합 preset 효과 (6명) | `outputs/combo_preset_test/` |
| 고개+웃음 조합 방식 비교 | `outputs/pose_smile_combo/` |
| `/api/v1/design` 동작 확인 (7가지) | `outputs/design_api_test/` |
| 처리 순서 비교 | `outputs/order_compare/` |
| 자연스러움·템플릿·마스크 영역 (10명) | `outputs/natural_compare/` |
| 입 다문 환자 2단계 (6명) | `outputs/closed_mouth/` |
| inpaint vs 마스크 없음 (28명) | `outputs/inpaint_vs_direct/` |
| 임베딩 조합 방식 | `outputs/mode_compare/`, [../final/embedding_modes.md](../final/embedding_modes.md) |
| **concat vs preset vs delta** | `outputs/combo_modes/`, `outputs/concat_test/`, `outputs/clause_test/` |
| **토큰 길이 한계 (2,316까지)** | `outputs/seqlen_limit/` |
| **v5 전체 실험 (28장 × 20스타일)** | `outputs/v5_full/sheet_*.jpg` |
| **미백 단독 변형 원인** | `outputs/whiten_only/`, `outputs/color_only/`, `outputs/v6_whiten_ab/` |
| **어려운 케이스 문구·템플릿** | `outputs/hard_case_prompts/` |
| **세그멘테이션 미백 (마스크 품질)** | `outputs/classic_whiten/` |
