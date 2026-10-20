# Smile Design Tune API (클라이언트용)

치아/얼굴 편집 추론 서버(`my/design_tune_server.py`)의 HTTP API.

## Swagger로 바로 테스트하기

서버가 떠 있으면 브라우저에서 아래 주소를 연다.

| 주소 | 내용 |
| --- | --- |
| `http://<서버>:5000/docs` | Swagger UI. 각 API의 **Try it out → Execute**로 실제 호출 |
| `http://<서버>:5000/openapi.yaml` | OpenAPI 3.0 스펙 원본 (`my/openapi.yaml`) |
| `http://<서버>:5000/openapi.json` | 같은 스펙의 JSON (코드 생성기용) |

`POST /api/v1/infer`에서 **Try it out**을 누르고, `images`에 환자 사진을 고른 뒤 **Execute**를 누르면
결과 PNG가 화면에 바로 표시된다. 같은 요청의 curl 명령도 만들어 준다.

![Swagger Try it out](assets/api_swagger_tryout.png)

- Swagger UI 화면 파일은 CDN(jsdelivr)에서 받는다. 브라우저가 인터넷에 접근할 수 없으면 `/docs`가 빈 화면이 된다
  (API 자체는 상관없음).
- 클라이언트 코드 생성: `openapi.json`을 OpenAPI Generator 등에 넣으면 된다. 단 `parameters` 필드는 JSON 문자열이라
  생성된 코드에서는 직접 직렬화해서 넣어야 한다(아래 1장).

## 1. 추론: `POST /api/v1/infer`

`multipart/form-data`로 보낸다.

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `parameters` | 텍스트 (JSON 문자열) | 추론 옵션. 아래 표 |
| `images` | 파일, 1개 이상 | layout 순서대로. `patient_only`는 환자 사진 1장, `patient_tooth`는 환자 사진 + 치아 템플릿 |

성공하면 **`200` + PNG 바이너리**(입력과 같은 크기). 메타데이터는 `X-Design-Tune-Metadata` 응답 헤더(JSON)에 있다.

```bash
curl -X POST http://<서버>:5000/api/v1/infer \
  -F 'parameters={"style_name": "smile_final", "layout": "patient_only", "mode": "preset", "preset": "align_whiten", "seed": 0, "mask": {"mode": "inpaint"}}' \
  -F 'images=@patient.jpg' \
  -o result.png -D headers.txt
```

```python
import json, requests

parameters = {"style_name": "smile_final", "layout": "patient_only", "mode": "preset",
              "preset": "align_whiten", "seed": 0, "mask": {"mode": "inpaint"}}
with open("patient.jpg", "rb") as f:
    r = requests.post("http://<서버>:5000/api/v1/infer",
                      data={"parameters": json.dumps(parameters)},
                      files=[("images", ("patient.jpg", f, "image/jpeg"))], timeout=600)
r.raise_for_status()
open("result.png", "wb").write(r.content)
metadata = json.loads(r.headers["X-Design-Tune-Metadata"])
```

### parameters

| 키 | 필수 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `style_name` | O | | 스타일 이름. 배포 스타일은 `smile_final` |
| `layout` | O | | `patient_only` 또는 `patient_tooth` (스타일에 정의된 것) |
| `mode` | | `preset` | `preset` / `anchor` / `grid` / `delta` |
| `preset` | | `natural` | `mode: preset`일 때. `smile_final`: `natural`, `whitening_natural`, `whitening_only`, `align_snaggle`, `align_only`, `align_whiten`, `turn_left`, `turn_right`, `look_up`, `look_down` |
| `control_values` | | `{}` | control 이름 → 0~100. 예: `{"tooth_color": 75, "alignment": 100}` (`anchor`/`grid`/`delta`) |
| `seed` | | `0` | 같은 seed + 같은 입력이면 같은 결과 |
| `num_steps` | | 서버 설정 (4) | 바꾸지 않는 것을 권장 |
| `mask` | | 서버 설정 (off) | 편집 영역 제한. 아래 표 |

### mask

치아만 바꾸려면 **`"mask": {"mode": "inpaint"}`**를 보낸다. 생략한 항목은 서버 기본값(`inner_mouth`, dilate 4,
feather 6)을 쓴다. 마스크 밖 픽셀은 원본 그대로 유지된다.

| 키 | 기본값 | 설명 |
| --- | --- | --- |
| `mode` | `inpaint` | `inpaint` 권장 (`composite`, `mice`는 레거시) |
| `regions` | `["inner_mouth"]` | `inner_mouth`, `teeth`, `lips`, `upper_lip`, `lower_lip`, `mouth`, `eyes`, `cheeks`, `face` |
| `dilate` | `4` | 마스크 확장 (px) |
| `feather` | `6` | 경계 blur (px) |
| `strength` | `1.0` | inpaint 강도 (0~1] |
| `edits` | | 여러 영역을 한 요청에서 조합. `[{"prompt": "teeth_white"}, {"prompt": "lips_red"}]` (스타일의 `mice.regions` 이름) |

- **치아 편집에는 `inner_mouth`**(입술 안쪽 윤곽선으로 둘러싼 영역)를 쓴다. `mouth`는 입술까지 포함해서 입술 색이 바뀐다
  ([fp8_4090_report.md 5장](fp8_4090_report.md)).
- 얼굴을 못 찾으면 GPU 추론 없이 바로 `400`이 온다.

### 응답 메타데이터 (`X-Design-Tune-Metadata`)

```json
{"style_name": "smile_final", "layout": "patient_only", "roles": ["patient"],
 "mix": {"mode": "preset", "preset": "align_whiten"}, "seed": 0, "steps": 4, "guidance_scale": 1.0,
 "input_size": [303, 346], "internal_size": [304, 352], "output_size": [303, 346],
 "text_encoder_loaded": false,
 "mask": {"enabled": true, "regions": ["inner_mouth"], "dilate": 4, "feather": 6.0, "mode": "inpaint", "strength": 1.0, "...": "..."},
 "step_warning": null}
```

- `internal_size`: 실제 생성 해상도. 면적이 1024×1024를 넘으면 비율을 유지해 줄이고(업스케일 없음), 결과는 원본 크기로 되돌려 준다.
- 같은 JSON이 PNG의 `design_tune` 텍스트 청크에도 들어 있다.

### 에러

에러는 항상 JSON `{"error": "..."}` (상황에 따라 필드 추가).

| 코드 | 의미 | 대응 |
| --- | --- | --- |
| 400 | 파라미터 오류, 얼굴 미검출, 이미지 수가 layout과 다름 | `error` 메시지 확인 |
| 404 | 스타일 없음 | `GET /api/v1/styles`로 이름 확인 |
| 409 | 스타일의 임베딩이 아직 빌드되지 않음, 또는 서버 모델과 임베딩 모델이 다름 | 운영자에게 문의 (`build_endpoint` 필드 참고) |
| 413 | 요청이 64MiB 초과 | 이미지 크기 줄이기 |
| 500 | 추론 실패 | 재시도, 반복되면 운영자에게 문의 |
| 507 | GPU 메모리 부족 | 잠시 후 재시도 |

### 입력 이미지 조건

실제로 보내 보고 확인한 동작이다([robustness_report.md](robustness_report.md) 1장).

| 항목 | 동작 |
| --- | --- |
| 형식 | JPEG, PNG(8/16bit, 투명도 포함), WebP, BMP, GIF(첫 프레임), CMYK JPEG 모두 처리 |
| 결과 형식 | 항상 RGB PNG. 흑백 입력도 RGB로, 투명도(alpha)는 버려진다 |
| EXIF 회전 | 반영한다. 결과는 보이는 방향 그대로(회전 정보 없이) 돌아온다 |
| 최소 크기 | 가로·세로 64px 이상. 미만이면 `400` |
| 비율 | 8:1 이내. 넘으면 `400` |
| 큰 이미지 | 면적 1024×1024를 넘으면 그 크기로 줄여 생성한 뒤 원본 크기로 키워서 돌려준다 — 고해상도 사진은 편집 영역이 주변보다 덜 선명할 수 있다 |
| 요청 크기 | 64MiB 이하. 넘으면 `413` |
| 얼굴 여러 명 | **한 명만** 편집된다(검출된 첫 얼굴). 어느 얼굴인지는 지정할 수 없다 |
| 얼굴 없음 | 마스크 요청이면 `400`, 마스크 없는 요청은 그대로 생성 |
| 깨진 파일 / 빈 파일 / 이미지 아님 | `400` (`Could not decode image`) |

### 처리 시간

RTX 4090, `--fp8-compute --compile` 기준 1024×1024 약 2초, 작은 이미지(300px대) 0.3초
([fp8_4090_report.md](fp8_4090_report.md)). 요청은 GPU 하나에서 **순서대로** 처리되므로 동시에 보내면 대기 시간이 늘어난다.
`--compile` 서버는 재시작 직후 첫 요청(MICE를 쓰면 MICE 첫 요청도)이 수십 초~수 분 걸릴 수 있다 — 타임아웃은 넉넉히(600초) 잡는다.
동시에 여러 요청을 보내도 처리량은 늘지 않는다(약 0.8건/초) — [robustness_report.md](robustness_report.md) 5장.

## 2. 마스크 미리보기: `POST /api/v1/mask`

생성 없이 편집 영역만 확인한다. `images` 첫 장만 쓴다.

```bash
curl -X POST http://<서버>:5000/api/v1/mask \
  -F 'parameters={"mask": {"regions": ["inner_mouth"]}, "overlay": true}' \
  -F 'images=@patient.jpg' -o mask.png
```

`overlay: true`면 원본 위에 빨간색으로 겹친 PNG, 아니면 흑백 마스크 PNG. 메타데이터는 `X-Design-Tune-Mask` 헤더.

## 3. 스타일 조회

| API | 설명 |
| --- | --- |
| `GET /api/v1/styles` | 스타일 목록. `presets`, `controls`, `layouts`, `library_ready`(false면 infer가 409) |
| `GET /api/v1/styles/{name}` | 스타일 YAML 원문 (`Accept: application/json`이면 `{"name", "yaml"}`) |

목록의 `name`은 YAML의 `template_version`(예: `smile-final-v1`)이다. 요청에는 파일 이름(`smile_final`)과
`template_version` 둘 다 쓸 수 있다.

## 4. 운영자용 (클라이언트는 쓰지 않음)

| API | 설명 |
| --- | --- |
| `PUT /api/v1/styles/{name}` | 스타일 YAML 생성/수정. body `{"yaml": "..."}` |
| `DELETE /api/v1/styles/{name}` | 스타일 YAML 삭제 (임베딩 캐시는 남음) |
| `POST /api/v1/styles/{name}/build` | 임베딩 빌드. 이미 빌드됐으면 `200 skipped`, 새로 빌드하면 `202` + `job_id` |
| `GET /api/v1/jobs/{job_id}` | 빌드 상태 (`queued` / `running` / `succeeded` / `failed`) |
| `GET /health` | 서버 상태와 기본 설정 (모델, dtype, fp8, compile, 마스크 기본값) |

현재 **인증이 없다** — 이 API들도 네트워크에 열려 있으면 누구나 호출할 수 있다([deployment_review.md](deployment_review.md) 4장).
