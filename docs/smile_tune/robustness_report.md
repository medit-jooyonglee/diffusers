# 배포 전 검증: 이상 입력 / 재현성 / compile warmup / MICE / 동시 요청

AWS(g6.2xlarge, L4) 빌드 테스트 전에 4090에서 확인한 것. 서버 설정은 배포 후보인 `--fp8-compute --compile`
(마스크 기본 `inner_mouth`)이고, 모든 호출은 실제 HTTP API로 했다.
앞선 성능 측정은 [fp8_4090_report.md](fp8_4090_report.md), 배포 구성은 [deployment_review.md](deployment_review.md).

## 요약

- **버그 2건을 찾아 고쳤다.**
  1. `--fp8-compute --compile` 서버에서 첫 요청이 inpaint가 아니면(MICE, 포즈 preset, 마스크 없는 요청) 그 뒤 **모든 inpaint
     요청이 500**이 됐다 (4장).
  2. 64MiB를 넘는 요청이 `413`이 아니라 `500`으로 응답했다 (1장).
- **compile은 해상도마다 다시 하지 않는다.** 16가지 해상도에 재컴파일 0회. 대신 **MICE는 transformer 호출 인자가 달라
  첫 요청에서 한 번 더** 컴파일한다. `--compile regional`(새 옵션)은 속도가 같으면서 첫 컴파일이 251초 → 98초(캐시 있으면 92초 → 34초).
- **재현성**: 같은 서버 프로세스 안에서는 같은 요청 → 같은 결과(바이트 단위). compile 서버는 **재시작하면 미세하게 달라진다**
  (평균 0.01/255, 최대 24/255). bf16 eager와 AOT 패키지는 재시작해도 동일.
- **AOT export(torch.export + AOTInductor)가 동작한다.** fp8 패키지(3.88GB)를 로드하는 데 2.1초, 속도는 compile과 같다 —
  런타임 컴파일이 없어진다. 서버 연동은 아직 안 했다 (3.4절).
- **동시 요청**: GPU 처리는 한 번에 하나라 처리량은 약 0.8건/초에서 더 늘지 않고, 동시성만큼 대기 시간이 늘어난다
  (동시 8개: p95 12.4초). 에러 0, VRAM 증가 없음, 결과는 순차 처리와 동일.
- **MICE와 모든 inpaint 경로**가 fp8 연산 + compile에서 bf16 eager와 같은 결과(평균 차이 0.02~0.3/255)로, 30~40% 빠르게 동작한다.

## 1. 이상 입력

`edge_test.py`, 37가지. 서버는 끝까지 정상(`/health` ok).

| 분류 | 입력 | 결과 |
| --- | --- | --- |
| 형식 | RGBA PNG, 흑백, CMYK JPEG, 16bit PNG, WebP, BMP, 애니메이션 GIF | 모두 `200`. 결과는 RGB PNG (흑백도 RGB, alpha는 버림), GIF는 첫 프레임 |
| EXIF | 픽셀을 90° 돌려 저장 + Orientation=6 | `200`, 결과는 보이는 방향(정상 방향) 그대로 |
| 크기 | 15×15 | `400` "at least 64px" (diffusers Flux2 전처리 검증) |
| | 64×64 + 마스크 | `200` (얼굴 검출됨) |
| | 4000×200 | `400` "Maximum allowed ratio is 8:1" |
| | 5000×5000 PNG | `200`, 6.3초. 1024×1024로 줄여 생성 후 원본 크기로 키움 |
| | 70MB 요청 | **수정 전 `500`** → 수정 후 `413` "Request exceeds 64 MiB" |
| 내용 | 얼굴 없음 + 마스크 | `400` "no face was detected" (GPU 추론 전에 실패, 0.02초) |
| | 얼굴 없음, 마스크 없음 | `200` |
| | 두 사람 | `200`, **한 명만 편집**(`num_faces=1`) — 어느 얼굴인지 지정 불가 |
| 파일 | 텍스트, 잘린 JPEG, 빈 파일 | `400` "Could not decode image" |
| 파라미터 | 잘못된 JSON, JSON 배열, style_name 없음, 없는 preset/mode/region, 모르는 mask 키 | 모두 `400`, 원인이 메시지에 나온다 |
| | strength 0, dilate -1, num_steps 0 | `400` |
| | seed -1 | `200` (허용) |
| | seed 2⁶⁴, seed "abc", control_values 문자열, control 값 500 | `400` |
| | 이미지 2장(patient_only), 6장(최대 5) | `400` |
| | num_steps 20 | `200`, 7.6초 — **상한이 없다** |
| | guidance_scale 10 (distilled 모델) | `200`, 시간 동일 — 에러 없이 받아들인다 |

**남은 주의점**

- `num_steps`에 상한이 없어 큰 값을 보내면 GPU를 오래 점유한다 — 배포 전에 상한(예: 8)을 두는 것을 권장.
- 두 사람 이상인 사진은 한 명만 편집된다. 클라이언트 안내가 필요하다([api.md](api.md)에 반영).
- 고해상도 사진은 편집 영역이 최대 1024² 면적에서 생성된 뒤 키워지므로 주변보다 덜 선명할 수 있다(이번 테스트 이미지는
  원본 자체가 1024px 확대본이라 확인하지 못했다 — 실제 고해상도 사진으로 확인 필요).

## 2. 재현성

`repro_test.py`. 같은 요청(teeth04, `align_whiten`, seed 0)을 조건별로 보내 결과 PNG를 비교했다.

| 조건 | fp8 연산 + compile | fp8 연산 eager | bf16 eager |
| --- | --- | --- | --- |
| 같은 프로세스에서 반복 | 동일 | 동일 | 동일 |
| 다른 해상도 3개를 처리한 뒤 | 동일 | 동일 | 동일 |
| **서버 재시작** | **다름** (평균 0.012, 최대 24 / 255) | — | 동일 |
| 동시 요청 중 (5장) | 동일 | — | — |
| seed 1 | 다름 (정상, 평균 0.07) | | |

- 같은 프로세스 안에서는 완전히 결정적이다. 동시 요청이 섞여도 같다.
- compile 서버는 재시작마다 커널 선택이 달라질 수 있어 결과가 미세하게 다르다(육안 구분 불가 수준). 클라이언트가
  "같은 요청이면 바이트까지 같은 결과"를 기대하면 안 된다 — 필요하면 bf16 eager를 쓰거나, AOT 패키지(3.4절)로 고정해야 한다.
- 정밀도 사이 차이(fp8 vs bf16)는 [fp8_4090_report.md](fp8_4090_report.md) 3장.

## 3. compile warmup

### 3.1 해상도: 다시 컴파일하지 않는다

`warmup.py`. 빈 캐시로 서버를 띄우고 16가지 해상도(304×352 ~ 1080×1920 입력)를 두 번씩 보냈다.

| | 첫 요청 (1024×1024) | 나머지 15개 해상도 (첫 처리) | 재컴파일 | 재컴파일 한도 도달 |
| --- | --- | --- | --- | --- |
| 빈 캐시 | 251초 | 0.25~4.8초 (= 평소 속도) | 0회 | 0회 |
| 캐시 있음 (재시작) | 92초 | 같음 | 0회 | 0회 |

`dynamic=True`라 해상도를 기호로 처리해 그래프 하나를 재사용한다. 해상도를 몇 가지로 고정(bucketing)할 필요는 없다.

### 3.2 MICE는 따로 컴파일한다

torch.compile은 transformer 호출 인자의 구성이 달라지면 그래프를 새로 만든다. inpaint(`Flux2KleinInpaintPipeline`)와 일반 생성
(`Flux2KleinPipeline`, 마스크 없음/포즈)은 transformer를 **같은 인자**로 부른다(`joint_attention_kwargs=None`). MICE만 hook이
`joint_attention_kwargs={"attention_mask": ...}`를 추가한다.

MICE 테스트에서 MICE 첫 요청 108초 후 inpaint 첫 요청에서 다시 127초가 걸렸다(재컴파일 원인: `joint_attention_kwargs` 타입이
dict → None으로 바뀜). MICE hook 내부 상태 때문에 작은 재컴파일이 3회 더 있었다(각 수백 ms). inpaint ↔ 일반 생성 전환은 인자가
같아 같은 그래프를 쓸 것으로 보지만 직접 측정하지는 않았다.

→ warmup은 MICE를 쓰면 MICE 요청과 일반(inpaint) 요청을 각각 한 번씩 보낸다. MICE를 쓰지 않으면 한 번이면 된다.

### 3.3 regional compile (`--compile regional`, 새 옵션)

diffusers `compile_repeated_blocks()` — Flux2의 반복 블록(`Flux2TransformerBlock`, `Flux2SingleTransformerBlock`)만 컴파일해 모든 블록에 재사용한다.

| | full (`--compile`) | **regional (`--compile regional`)** |
| --- | --- | --- |
| transformer 단독 첫 호출 (빈 캐시) | 290초 | **115초** |
| transformer 단독 1024² inpaint 크기 | 348ms | 352ms |
| 서버 첫 요청, 빈 캐시 | 251초 | **98초** |
| 서버 첫 요청, 캐시 있음 | 92초 | **34초** |
| 이후 요청 (1024² / 704×1024 / 304×352) | 2.0 / 1.35 / 0.25초 | 2.0 / 1.36 / 0.26초 |

**속도 손해 없이 컴파일 시간이 60% 짧다 → 배포는 `--compile regional`을 권장한다.** (eager 대비 결과 차이도 full과 같은 수준.)
vCPU가 8개인 g6.2xlarge는 컴파일이 이보다 느릴 수 있어 차이가 더 중요해진다.

### 3.4 AOT export (torch.export + AOTInductor)

컴파일을 배포 전에 한 번 해서 `.pt2` 패키지(컴파일된 커널 + 실행 코드 + fp8 가중치)로 저장하고, 서버는 로드만 하는 방식.
`compile_opts.py`에서 fp8 연산 transformer를 시퀀스 길이 64~16384 범위로 export했다(서버 연동은 아직 안 함).

| 단계 | 결과 |
| --- | --- |
| `torch.export` (torchao fp8 + 가변 시퀀스 길이) | 성공, 7.6초 |
| `aoti_compile_and_package` | 성공, 220초 (20코어 CPU). **CUDA toolkit(nvcc + 헤더) 필요** — 런타임 이미지에는 없다 |
| 패키지 크기 | 3.88GB (fp8 가중치 포함 — 추론 서버에는 원본 transformer 가중치 7.7GB가 필요 없어진다) |
| 새 프로세스에서 로드 | **2.1초**, 첫 호출 0.4초 (런타임 컴파일 없음) |
| 속도 (8192 / 5632 / 836 토큰) | 348 / 216 / 38ms — full compile(348 / 216 / 39ms)과 같다 |
| eager 대비 차이 | full/regional compile과 같은 수준 |
| 재현성 (두 프로세스에서 로드) | **완전히 동일** — compile 서버의 재시작 간 차이(2장)가 없다 |

| 방식 | 서버 첫 요청까지 | 속도 | 재시작 간 결과 | 추가 준비 |
| --- | --- | --- | --- | --- |
| full compile | 251초 (캐시 92초) | 기준 | 미세하게 다름 | 없음 |
| regional compile | 98초 (캐시 34초) | 같음 | (미측정, full과 같을 것으로 예상) | 없음 |
| **AOT 패키지** | **수 초** (transformer 로드 2.1초) | 같음 | 동일 | 빌드 단계, 서버 연동 코드 |

**제약**

- 패키지는 GPU 아키텍처와 torch 버전에 묶인다. 4090과 L4는 둘 다 sm_89라 4090에서 만든 패키지를 L4에서 쓸 수 있을
  가능성이 높지만 **확인 필요**. A10G(sm_86)용은 따로 만들어야 하고, torch를 올리면 다시 만든다.
- 입력 시그니처가 고정된다. 이번 패키지는 inpaint/일반 생성용이고, **MICE(`attention_mask` 추가)는 별도 패키지**가 필요하다.
- 빌드 환경에 CUDA toolkit이 필요하다(이번에는 conda로 CUDA 12.8 nvcc/헤더를 따로 설치). 빌드를 Docker 이미지 빌드 단계에서
  하려면 GPU가 있는 빌드 머신이 필요하다 — 별도 빌드 작업으로 만들어 S3 등에 올리고 런타임에 받는 쪽이 현실적이다.
- 서버 연동(transformer를 패키지로 바꿔 끼우기, 파이프라인이 기대하는 속성 유지)은 아직 하지 않았다.
- 이 측정의 `max_memory_allocated`는 AOT 런타임의 메모리를 잡지 못해 VRAM 비교는 서버 연동 후 nvidia-smi로 다시 재야 한다.

**판단**: 오토스케일링으로 인스턴스가 자주 새로 뜨면 AOT가 가장 좋다(첫 요청 대기 없음, 재현성 고정, 서버에 원본 가중치
불필요). 당장은 `--compile regional`로 시작하고, AOT는 서버 연동 + L4 호환성 확인 후 전환하는 순서를 권장한다.

## 4. MICE와 다른 경로

`mice_test.py`. 환자 3명(teeth01 304×352, yellowteeth 1024², sample18 704×1024) × 6가지, fp8 연산 + compile vs bf16 eager.

### 4.1 버그: compile 후 inpaint 파이프라인 생성 실패 (수정함)

inpaint 파이프라인은 첫 inpaint 요청 때 만들어지며 `inpaint.to(device)`를 불렀다. 이미 compile된 fp8 transformer에 `.to()`를
다시 걸면 torchao fp8 가중치 swap이 dynamo weakref 때문에 실패한다(`Couldn't swap Linear.weight`) — 이후 inpaint 요청이 모두
`500`. 앞선 테스트는 첫 요청이 항상 inpaint라 드러나지 않았다. 공유 모듈은 이미 device에 있으므로 `.to()` 호출을 없앴다.
수정 후 MICE → inpaint 순서로 다시 돌려 전부 `200`을 확인했다.

### 4.2 결과 (yellowteeth 1024², 두 번째 요청 기준)

| 경로 | 요청 | fp8 연산 + compile | bf16 eager | bf16 대비 평균 차이 (0-255) |
| --- | --- | --- | --- | --- |
| MICE 텍스트 | `mask.mode: mice`, `teeth_align_white` | 2.57초 | 3.57초 | 0.04 |
| MICE 텍스트, 영역 inner_mouth | 위 + `region: [inner_mouth]` | 2.57초 | 3.57초 | 0.03 |
| MICE 템플릿 | `patient_tooth` + 치아 템플릿, `teeth_from_template` | 3.47초 | 4.72초 | 0.06 |
| inpaint edits 1영역 | `edits: [teeth_white]` | 1.97초 | 3.00초 | 0.02 |
| inpaint edits 3영역 | `pose-makeup-v2`, 치아+입술+눈 | 5.40초 | 8.47초 | 0.07 |
| patient_tooth preset | 템플릿 + `align_whiten`, inpaint | 2.55초 | 3.85초 | 0.04 |

모든 경로가 fp8 연산 + compile에서 동작하고 bf16과 같은 결과를 30~40% 빠르게 낸다.

### 4.3 MICE 지시문의 영역도 입술을 포함한다

`smile_final.yaml`의 MICE 지시문 중 `teeth_align`, `teeth_align_white`, `teeth_from_template`은 `region: [mouth]`(입술 포함)다.
입술 영역 평균 변화: `teeth_align_white` 17~22 → 요청에서 `region: [inner_mouth]`로 덮어쓰면 3~9
([fp8_4090_report.md](fp8_4090_report.md) 5장과 같은 문제). YAML의 해당 region도 `inner_mouth`로 바꾸는 것을 권장한다
(바꾸면 임베딩 재빌드가 필요할 수 있다 — YAML 내용 해시가 바뀌므로).

## 5. 동시 요청 부하

`load_test.py`. 5명 이미지를 섞어 20건씩, 동시성 1/2/4/8. 서버는 warmup 후.

| 동시 요청 | 처리량 | p50 | p95 | 최대 | 에러 | 순차 결과와 동일 | VRAM 피크 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.69건/초 | 1.71초 | 2.00초 | 2.00초 | 0 | 20/20 | 9.6GB |
| 2 | 0.81건/초 | 2.80초 | 3.17초 | 3.30초 | 0 | 20/20 | 9.6GB |
| 4 | 0.81건/초 | 4.75초 | 6.50초 | 7.44초 | 0 | 20/20 | 9.6GB |
| 8 | 0.81건/초 | 7.93초 | 12.39초 | 15.70초 | 0 | 20/20 | 9.6GB |

- GPU 추론은 락으로 한 번에 하나씩 처리된다. 동시성 2에서 처리량이 18% 오르는 건 업로드·마스크·PNG 인코딩(CPU)이
  다른 요청의 GPU 처리와 겹치기 때문이고, 그 이상은 늘지 않는다. **GPU 1장의 한계는 약 0.8건/초(시간당 약 2,900건, 이 이미지 구성 기준)**.
- 대기 시간은 대기열 길이에 비례해 늘어난다. 서버에 대기열 상한이 없어서, 몰리면 클라이언트 타임아웃이 먼저 난다 —
  인스턴스를 늘리거나(오토스케일링 기준: 대기열 길이 또는 p95) 앞단에서 대기열을 제한해야 한다.
- 동시 요청이 VRAM을 늘리지 않는다(한 번에 하나만 GPU에 올라감).
- **같은 GPU에서 임베딩 빌드 중에는 추론이 멈춘다**: 빌드(11초) 동안 들어온 요청이 12초 걸렸다. 운영 중 빌드는 다른 GPU
  (`--build-device`)에서 하거나 배포 전에 미리 해 둔다.

## 6. 코드 변경

- `my/design_tune_server.py`
  - `load_inpaint()`: `inpaint.to(device)` 제거 (4.1절 버그)
  - `infer`: `HTTPException`을 `500`으로 바꾸지 않고 그대로 올림 (413 버그)
  - `--compile [full|regional]`: 값 없이 `--compile`이면 기존처럼 full
- 문서: "해상도마다 컴파일한다"는 이전 설명을 정정 ([fp8_4090_report.md](fp8_4090_report.md), [deployment_review.md](deployment_review.md),
  [api.md](api.md)), api.md에 입력 이미지 조건 추가.

## 7. 다음 할 일

1. 배포 설정을 `--fp8-compute --compile regional`로, 서버 시작 시 warmup 요청(MICE를 쓰면 MICE 요청도)을 보낸 뒤 준비 완료로 표시.
2. `num_steps` 상한, 대기열 상한(또는 앞단 제한).
3. `smile_final.yaml`의 MICE region을 `inner_mouth`로 바꾸고 재빌드할지 결정.
4. g6.2xlarge(L4)에서 같은 스크립트로 재측정 — 특히 컴파일 시간(vCPU 8개)과 처리량. 4090에서 만든 AOT 패키지가 L4에서 로드되는지도.
5. AOT 패키지 서버 연동(`--aot-package` 같은 옵션), MICE용 패키지가 필요한지 결정.
6. 텍스트 인코더 없이 추론 서버를 띄우는 테스트 (보류 중).

실험 스크립트: `warmup.py`, `repro_test.py`, `edge_test.py`, `mice_test.py`, `load_test.py`, `compile_opts.py`, `regional_server.py`, `aoti_repro.py`
(세션 scratchpad, 저장소에는 넣지 않음).
