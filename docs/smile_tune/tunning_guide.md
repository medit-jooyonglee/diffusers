# Smile Design 튜닝 서버 — 앱 연동 가이드 (최종)

앱은 **설정만 보낸다.** 마스크, 처리 순서, 단계 수, 조합 방식은 서버가 정한다.

- 전용 엔드포인트: **`POST /api/v1/design`** (구현 완료)
- 기준 스타일: **`smile_final_smile_v3`**
- 서버 실행과 옵션: [README.md](README.md) · 전체 API: [api.md](api.md) 또는 서버의 `/docs`
- 설정별 결과의 근거: [../final/README.md](../final/README.md)

---

## 1. 앱에서 바꿀 것 (지시)

기존에 `/api/v1/infer`로 직접 호출하던 코드를 아래대로 고친다.

1. **호출 대상을 `POST /api/v1/design`으로 바꾼다.** `/api/v1/infer`는 유지되지만 앱에서 쓰지 않는다.
2. **`mask`를 보내지 않는다.** 마스크 종류·영역·dilate·feather를 요청에서 모두 제거한다. 서버가 편집 종류에 맞춰 정한다.
3. **`layout`을 보내지 않는다.** 이미지를 1장 보내면 `patient_only`, 2장 보내면 `patient_tooth`로 서버가 판단한다.
4. **`mode`, `preset`, `grid`, `attribute`, `control_values`를 보내지 않는다.** 대신 control 값을 **최상위 필드**로 보낸다
   (`tooth_color`, `alignment`, `head_yaw`, `head_pitch`, `smile`).
5. **`delta`와 `delta_strengths`를 완전히 제거한다.** 가중치 슬라이더가 있으면 UI에서 뺀다.
   측정 결과 조합이 깨진다 — 고개+웃음을 `delta`로 합치면 고개 회전이 23.7° → 1.3°로 사라졌다.
6. **단계를 앱에서 체이닝하지 않는다.** 고개·웃음·치아를 한 요청에 함께 보내면 서버가 순서대로 처리하고 최종 이미지 1장을 돌려준다.
7. **`seed`를 항상 명시한다.** 생략하면 매번 결과가 달라진다. "다시 생성"에서만 값을 바꾼다.
8. **`num_steps`를 보내지 않는다.** 4 step 고정이다. 올리면 오히려 나빠진다(sample17: b\* 2.6 → 3.1 → 3.7).
9. **고개는 25 / 75만 노출한다.** 0과 100은 변화가 과하다.
10. **요청을 직렬로 보낸다.** 서버에 GPU 락이 있어 동시에 보내도 순서대로 처리된다. 앱에서 큐로 관리하고 진행 표시를 둔다.
11. **응답 헤더 `X-Design-Tune-Metadata`를 읽는다.** `identity_warning`이 `true`면 결과를 **"참고용 시뮬레이션"으로 표시**한다.
12. **`unchanged`가 `true`면** 서버가 추론 없이 원본을 그대로 돌려준 것이다. "변경 없음"으로 처리한다.
13. **스타일 변경 API를 호출하지 않는다.** `PUT`/`DELETE /api/v1/styles/{name}`, `POST …/build`. `PUT`으로 YAML이 바뀌면 이후 모든 추론이 409가 된다.

### 앱에 노출할 설정

| 노출한다 | 노출하지 않는다 |
| --- | --- |
| 치아 미백 (0/50/100) | 마스크 종류·영역·dilate·feather |
| 치아 정렬 (0/50/100) | step 수 |
| 치아 템플릿 사용 여부 | 조합 모드(preset/anchor/grid/delta) |
| (선택) 고개 25/75 | 가중치(`delta_strengths`) |
| (선택) 웃음 0/25/75/100 | layout |
| seed | 처리 순서·단계 수 |

---

## 2. `POST /api/v1/design` 사용법

`multipart/form-data`로 `parameters`(JSON 문자열)와 `images`(파일)를 보낸다.
응답은 **PNG 바이너리**, 설정은 `X-Design-Tune-Metadata` 헤더(JSON)와 PNG의 `design_tune` 청크.

### 요청 필드

| 필드 | 필수 | 설명 |
| --- | --- | --- |
| `style_name` | O | `"smile_final_smile_v3"` |
| `tooth_color` | | 0(유지) / 50 / 75 / 100 |
| `alignment` | | 0(유지) / 50 / 75 / 100 |
| `head_yaw` | | 50(유지) / 25 / 75 |
| `head_pitch` | | 50(유지) / 25 / 75 |
| `smile` | | 50(유지) / 0 / 25 / 75 / 100 |
| `seed` | | 기본 0 |
| `open_closed_mouth` | | 기본 `true`. 다문 입 자동 2단계를 끄려면 `false` |
| `images` | O | 환자 사진 1장, 또는 환자 사진 + 치아 템플릿 2장 |

### 예시

**치아만 (가장 많이 쓰는 형태)**
```json
{"style_name":"smile_final_smile_v3","tooth_color":75,"alignment":100,"seed":0}
```

**치아 + 템플릿** — `images`에 환자 사진, 템플릿 순서로 2장
```json
{"style_name":"smile_final_smile_v3","tooth_color":75,"alignment":100,"seed":0}
```

**고개 + 웃음 + 치아** — 한 요청으로 보내면 서버가 2단계로 처리한다
```json
{"style_name":"smile_final_smile_v3","head_yaw":75,"smile":100,"tooth_color":75,"alignment":100,"seed":0}
```

### 응답 메타데이터

```json
{"style_name":"smile_final_smile_v3",
 "requested":{"smile":100.0,"head_yaw":75.0,"tooth_color":75.0,"alignment":100.0},
 "seed":0,
 "mouth_openness":{"inner_mouth":34.5,"teeth":16.1,"face_width":379.0},
 "closed_mouth":false, "unchanged":false, "identity_warning":true, "warnings":[],
 "stages":[{"stage":"pose_smile","index":0,"template":false,"seconds":3.34,
            "mix":{"mode":"preset","preset":"yaw75_smile_full"},"mask":{"enabled":false}},
           {"stage":"teeth","index":1,"template":false,"seconds":2.1,
            "mix":{"mode":"preset","preset":"align_whiten"},"mask":{"enabled":true,"regions":["inner_mouth"]}}]}
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
고개 + 웃음           → 조합 preset이 있으면 1회, 없으면 2회 (경고)
치아 + 큰 변화        → 순차: 치아 inpaint 먼저 → 고개/웃음(마스크 없음)
입이 다물려 있음       → 자동 2단계: 웃음 75로 입을 벌린 뒤 → 치아 inpaint (이때만 치아가 뒤)
```

| 편집 | 마스크 | 근거 |
| --- | --- | --- |
| **치아 (미백·정렬)** | `inpaint`, **`inner_mouth`, dilate 8, feather 6** | `mouth`는 입술까지 다시 그린다 (입술 변화 8.8 vs 4.4) |
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

**조합 방식**: `delta`(덧셈)는 쓰지 않는다. **조합 preset → grid → anchor** 순으로 실제 문장을 인코딩한 임베딩만 쓴다.

---

## 4. 조합 preset 추가하기

고개와 웃음을 따로 돌리면 뒤 단계가 앞 단계를 되돌린다(실측: 웃음 0.94 → 0.17).
두 축을 **한 문장으로 인코딩한 조합 preset**을 YAML에 넣으면 서버가 자동으로 찾아 쓴다.

현재 `smile_final_smile_v3.yaml`에 6개가 있다.

| preset | head_yaw | smile |
| --- | --- | --- |
| `yaw25_smile_closed` / `_upper` / `_full` | 25 | 0 / 75 / 100 |
| `yaw75_smile_closed` / `_upper` / `_full` | 75 | 0 / 75 / 100 |

**효과** (6명, 고개 75 + 웃음 100 + 정렬 100 + 미백 75):

| 방식 | yaw | 웃음 | 치아 b\* |
| --- | --- | --- | --- |
| delta 한 번에 | **1.3°** | 0.94 | 12.4 |
| 순차 고개→웃음 | 23.7° | 0.72 | 12.9 |
| 순차 웃음→고개 | 22.9° | **0.60** | 11.3 |
| **조합 preset** | **19.4°** | **0.81** | **−0.2** |

**추가 방법**: `my/configs/smile_design/smile_final_smile_v3.yaml`의 `presets:`에 control 값을 적고 재빌드한다.

```yaml
  yaw25_smile_upper:
    head_yaw: 25
    smile: 75
```

```bash
curl -X POST localhost:8603/api/v1/styles/smile_final_smile_v3/build -H 'Content-Type: application/json' -d '{}'
```

- 서버는 **control 값이 정확히 일치하는 preset**을 manifest에서 찾는다. 이름은 자유롭다.
- **서버 코드를 고칠 필요가 없다.**
- YAML을 저장하면 임베딩 해시가 바뀌어 **재빌드가 끝날 때까지 추론이 409**가 된다 (약 30~60초).

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

템플릿 이미지를 두 번째로 보내면 **치아가 덜 표백되고 질감이 산다.**

| | 치아 b\* | 결과 |
| --- | --- | --- |
| 템플릿 없음 | 1~3 | 균일한 흰 덩어리(틀니 느낌)가 되기 쉽다 |
| **템플릿 사용** | **6~11** | 치아 하나하나가 구분되고 음영이 산다 |

**b\*가 낮을수록 좋은 것이 아니다.** 1~3은 과하게 표백된 상태다.
다만 원본 치아가 거의 없는 케이스에서는 누런 치아가 남을 수 있으니 **선택 옵션**으로 둔다.

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
| 금속 보철 | 완전히 지워지지 않는다 (지시문에 제거 문구가 없음) |
| 검증 범위 | 실제 사진 28장 + 합성 얼굴 112장. **실제 임상 사진으로 추가 검증 필요** |

---

## 8. 스타일 전환

`smile_final_smile_v3`는 **실험용**이다. 배포 확정 스타일은 `smile_final`(웃는 정도 없음)이고 AWS 배포도 그쪽을 참조한다.
앱에서는 `style_name`만 바꾸면 된다. 단 `smile_final`에는 `smile_*` preset과 조합 preset이 없고 `patient_tooth` 임베딩도 빌드돼 있지 않다.

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
