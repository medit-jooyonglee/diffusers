# Smile Design 튜닝 서버 / 클라이언트 가이드

FLUX.2 Klein 4B 기반 Smile Design 튜닝 서버(`my/design_tune_server.py`)와
테스트용 CLI 클라이언트(`my/design_tune_client.py`)의 실행 방법, API, 얼굴 마스크 옵션을 정리한다.

- 서버: 스타일 YAML 관리, 프롬프트 임베딩 빌드, 이미지 추론
- 클라이언트: 서버 API를 터미널에서 호출 (빌드 → 폴더 순차 추론까지 한 번에 가능)
- 데스크톱 앱용 클라이언트 함수는 `my/temp/flux2_backend.py`에 있다 (마스크 옵션은 아직 미지원).
- 스타일 / 프롬프트 / pose / 메이크업 실험 결과와 다음 실험 계획은 [experiments.md](experiments.md)에 있다.

## 1. 구성 파일

| 파일 | 역할 |
| --- | --- |
| `my/design_tune_server.py` | Flask 서버. 스타일 CRUD, 임베딩 빌드 job, 추론, 마스크 미리보기 |
| `my/smile_design/face_mask.py` | MediaPipe Face Landmarker로 입술/치아/얼굴 영역 마스크 생성 |
| `my/design_tune_client.py` | 테스트용 CLI 클라이언트 (`health`, `styles`, `build`, `mask`, `infer`, `run`) |
| `my/configs/smile_design/*.yaml` | 스타일 정의 (프롬프트 템플릿, layout, control, preset, grid) |
| `my/build_smile_embedding_library.py` | 임베딩 빌더. 서버가 build 요청 시 별도 프로세스로 실행 |
| `weights/face_landmarker.task` | MediaPipe 얼굴 랜드마크 모델 (3.7MB, Apache 2.0) |

## 2. 환경 설정

`diffuser_lora` conda 환경 기준.

```bash
# MediaPipe (numpy / protobuf 버전은 바뀌지 않음)
pip install mediapipe==1.1.0

# MediaPipe 1.1은 CPU로만 돌려도 EGL / GLES 라이브러리가 필요하다
apt-get install -y libegl1 libgles2

# 얼굴 랜드마크 모델
curl -L -o weights/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

주의: `mediapipe`가 `opencv-contrib-python`을 함께 설치한다. 기존 `opencv-python`과 같은 `cv2` 폴더를 쓰므로
두 패키지의 버전을 같게 유지한다 (한쪽만 업그레이드/삭제하면 `cv2`가 깨질 수 있다).

## 3. 서버 실행

```bash
python my/design_tune_server.py --port 5000 --device cuda:0
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--port` | `5000` | 데스크톱 앱 기본 주소(`http://10.100.1.45:8603`)에 맞추려면 `8603` |
| `--device` | `cuda:0` | 추론 / 빌드에 쓸 GPU |
| `--styles-dir` | `my/configs/smile_design` | 스타일 YAML 폴더 |
| `--libraries-dir` | `outputs/design_tune_embedding_libraries` | 임베딩 라이브러리 저장 위치 |
| `--steps` / `--guidance-scale` | `4` / `1.0` | 요청에서 `num_steps`, `guidance_scale`로 덮어쓸 수 있음 |
| `--mask` / `--no-mask` | off | 얼굴 마스크 합성 기본값 (요청별로 덮어쓸 수 있음) |
| `--mask-regions` | `mouth` | 쉼표 구분 영역 목록 (5장 참고) |
| `--mask-dilate` | `8` | 마스크 확장 (픽셀) |
| `--mask-feather` | `12.0` | 마스크 경계 blur 반경 (픽셀) |
| `--face-landmarker-model` | `weights/face_landmarker.task` | MediaPipe 모델 경로 |
| `--fp8-weights` / `--no-fp8-weights` | on | transformer 가중치를 fp8로 **저장**하고 레이어별로 bf16으로 올려 계산 (9장 참고) |
| `--build-timeout` | `7200` | 임베딩 빌드 제한 시간 (초) |

## 4. REST API

| Method | Path | 용도 |
| --- | --- | --- |
| `GET` | `/health` | 서버 상태, 모델 로드 여부, 마스크 기본값 / 사용 가능한 영역 |
| `GET` | `/api/v1/styles` | 스타일 목록 (`library_ready`로 빌드 여부 확인) |
| `GET` | `/api/v1/styles/{name}` | YAML 조회 |
| `PUT` | `/api/v1/styles/{name}` | YAML 추가 / 수정 (`{"yaml": "..."}`) |
| `DELETE` | `/api/v1/styles/{name}` | 스타일 삭제 (임베딩 캐시는 남는다) |
| `POST` | `/api/v1/styles/{name}/build` | 임베딩 빌드 시작 → `job_id` 반환 |
| `GET` | `/api/v1/jobs/{job_id}` | 빌드 상태 (`queued` / `running` / `succeeded` / `failed`) |
| `POST` | `/api/v1/infer` | 추론. multipart `images`(layout role 순서) + `parameters`(JSON) |
| `POST` | `/api/v1/mask` | 생성 없이 마스크만 확인. multipart `images` + `parameters` |

추론 `parameters` 예시:

```json
{
  "style_name": "smile_v1",
  "layout": "patient_only",
  "mode": "preset",
  "preset": "strong_whitening",
  "seed": 0,
  "mask": {"regions": ["lips", "teeth"], "dilate": 8, "feather": 12}
}
```

- `mode`: `preset` / `anchor` / `grid` / `delta` (`control_values`, `grid`, `attribute`, `delta_strengths`는 모드에 맞게)
- 첫 번째 이미지는 항상 환자(`patient`) 사진이다. 출력 크기는 첫 번째 이미지 크기와 같다.
- 응답은 PNG 바이너리이고, 사용한 설정은 `X-Design-Tune-Metadata` 헤더(JSON)에 들어 있다.

## 5. 얼굴 마스크 옵션

마스크를 켜면 전체 이미지를 생성한 뒤, **선택한 영역만** 원본 환자 사진 위에 합성한다.
마스크 밖은 원본 픽셀이 그대로 유지된다 (테스트에서 마스크 밖 픽셀 차이 0).

요청별 지정 (`parameters.mask`):

| 값 | 동작 |
| --- | --- |
| 생략 | 서버 기본값 (`--mask`, `--mask-regions` 등) |
| `true` / `false` | 기본 영역으로 켜기 / 끄기 |
| `{"regions": [...], "dilate": n, "feather": f}` | 켜고 영역 / 크기 지정 (`"enabled": false`로 끌 수도 있음) |

영역 (여러 개 지정하면 합집합):

| 영역 | 내용 |
| --- | --- |
| `upper_lip` / `lower_lip` | 윗입술 / 아랫입술 |
| `lips` | 입술 전체 |
| `inner_mouth` | 안쪽 입술 사이 (치아 + 혀 + 구강) |
| `teeth` | `inner_mouth` 중 밝고 붉지 않은 픽셀 (LAB 밝기 Otsu 후처리) |
| `mouth` | `lips` + `inner_mouth` |
| `face` | 얼굴 윤곽 |

동작 방식과 한계:

- MediaPipe Face Landmarker(478점)의 입술 / 얼굴 윤곽 점을 다각형으로 채운 근사 마스크다 (픽셀 단위 분할 모델이 아님).
- 얼굴이 검출되지 않으면 GPU 추론 전에 400 에러를 반환한다.
- 마스크는 **원본 사진 기준**이다. `broad_smile`, `mouth_opening`처럼 입 모양을 바꾸는 편집은
  `lips` / `teeth`만 쓰면 새 입 모양이 잘린다 → `mouth` 또는 `face`에 `dilate`를 크게 준다.
- 아주 가까운 클로즈업에서는 윗입술 다각형이 치아 끝을 살짝 덮을 수 있다.
- `teeth`는 밝기 규칙이라 누렇거나 그늘진 치아는 놓칠 수 있다.
- Face Landmarker는 CPU로 동작한다 (이미지당 약 10~30ms).

## 6. 클라이언트 (`my/design_tune_client.py`)

서버 주소는 `--server` 또는 환경변수 `DIFFUSION_TUNE_SERVER_URL` (기본 `http://127.0.0.1:5000`).

```bash
C="python my/design_tune_client.py --server http://127.0.0.1:5000"

$C health                                   # 서버 상태
$C styles                                   # 스타일 목록 / 빌드 여부
$C build smile_v1 --layout patient_only --wait   # 빌드 요청 후 완료까지 대기

# 마스크만 확인 (overlay: 원본 위에 빨간색으로 표시)
$C mask patient.png --regions lips,teeth --overlay -o mask.png

# 단일 추론
$C infer smile_v1 patient.png --preset strong_whitening --mask lips,teeth -o out.png
$C infer smile_v1 patient.png --mode grid --grid whitening_smile \
   --controls '{"whitening": 70, "smile_intensity": 50}' --no-mask -o out.png

# 빌드 → 완료 대기 → 폴더 이미지를 순서대로 추론 (이미지마다 프리셋 전부)
$C run smile_v1 <입력폴더> <출력폴더> \
   --build --layout patient_only --preset natural,strong_whitening --mask lips,teeth
```

`run` 결과:

- `<출력폴더>/<이미지이름>__<프리셋>.png`
- `<출력폴더>/results.json`: 이미지별 상태, 소요 시간, 메타데이터 (실패 시 에러 메시지)
- 한 장이 실패해도 (예: 얼굴 미검출) 나머지는 계속 처리한다.
- `patient_tooth` 등 참조 이미지가 필요한 layout은 `--reference`로 layout 순서대로 추가한다.

## 7. 튜닝 흐름

1. 스타일 YAML 수정 → `PUT /api/v1/styles/{name}`
2. `POST /api/v1/styles/{name}/build` → `GET /api/v1/jobs/{job_id}`로 완료 확인
3. `POST /api/v1/infer`로 결과 확인 (마스크 영역은 `POST /api/v1/mask`로 먼저 확인)

서버 재시작은 필요 없다.

- 라이브러리 경로는 `<libraries-dir>/<style 이름>/<YAML 내용 SHA256>`이다. 프롬프트를 한 글자만 바꿔도 새 경로가 된다.
- 빌드가 끝나기 전에는 해당 스타일 추론이 409 (`Embedding library is not built`)를 반환한다. 이전 라이브러리를 자동으로 쓰지 않는다.
- 빌드가 끝나면 다음 요청부터 바로 새 임베딩이 쓰인다 (서버는 요청마다 라이브러리를 새로 연다).
- YAML을 이전 내용으로 되돌리면 이전 라이브러리를 재빌드 없이 다시 쓴다.
- 스타일 이름은 빌드와 추론에서 **같은 표기**를 쓴다. 경로에 요청한 이름이 그대로 들어가므로
  `smile_v1`로 빌드하고 `smile-v1`로 추론하면 409가 난다.
- 빌드 중에는 GPU 락 때문에 **다른 스타일의 추론도 빌드가 끝날 때까지 대기**한다.

## 8. 성능 / 용량 (RTX A6000 기준 측정값)

| 항목 | 값 |
| --- | --- |
| 빌드 (`patient_only`, 항목 57개 / 고유 프롬프트 38개) | 약 12초 (text encoder 로드 약 8초 + 인코딩 개당 0.07초) |
| 빌드 (`smile_v1` 전체, 고유 프롬프트 228개) | 약 25~30초 추정 (미측정) |
| 빌드 (`smile_v1` 전체, 6 layout, 고유 프롬프트 228개) | 26초 (실측), 1.8GB |
| 임베딩 1개 | `[1, 512, 7680]` bf16, 약 7.9MB |
| 라이브러리 (`patient_only`) | 약 293MB |
| 임베딩 로드 (preset 1개) | 약 6~9ms (요청마다 디스크에서 로드, 메모리 캐시 없음) |
| 추론 (4 step, 약 1MP) | `patient_only` 약 4.0초, 참조 이미지 1장 추가마다 약 +1~1.5초 (`patient_tooth_mouth_pose` 6.9초) |

- 서버는 Qwen text encoder를 로드하지 않는다. 임베딩을 미리 빌드해 두는 이유가 이것이다 (text encoder bf16 약 8.1GB 절약).
- 참조 이미지(치아 / 입 / 포즈)는 요청마다 VAE로 인코딩된다 (latent를 미리 저장하지 않음). 참조 이미지가 늘면 시간은 늘지만
  최대 VRAM은 1MP에서 거의 같다 (최대치는 VAE decode 단계에서 결정됨).
- step-distilled 모델(`is_distilled: true`)이라 `guidance_scale > 1`은 무시되고, 이 경우 negative 임베딩은 로드하지 않는다.

알려진 문제: `build_embedding_library.py`를 `--batch-size 2` 이상으로 빌드하면 파일마다 배치 전체가 저장되어
용량이 배수로 커진다 (`.contiguous()` → `.clone()` 수정 필요). 서버 기본 빌드는 batch 1이라 해당 없음.

## 9. 정밀도(FP) 옵션별 VRAM

RTX A6000에서 모듈별로 측정한 값 (1024×1024, 4 step). fp16 대신 **bf16** 기준이다
(메모리는 같고, FLUX는 bf16으로 학습되어 fp16은 activation overflow 위험이 있다).
"fp8"은 별도 표기가 없으면 **fp8 저장 + bf16 연산**(`enable_layerwise_casting`)이다.

### 9.1 모듈별

| 모듈 | 정밀도 | 가중치 | 실행 중 추가 (activation) | 비고 |
| --- | --- | --- | --- | --- |
| Text encoder (Qwen3 4B) | bf16 | 8.10GB | +0.22GB (512 토큰) | 서버에서는 로드하지 않음 |
| | fp8 저장 | 4.44GB | +0.25GB | embedding(0.78GB)은 bf16 유지 |
| | bf16, 27층까지만 | 6.27GB | +0.16GB | 임베딩은 9 / 18 / 27층 출력만 쓰므로 28~36층과 `lm_head`는 불필요 |
| | fp8 + 27층 | 약 3.5GB (추정) | 약 +0.2GB | |
| DiT (transformer) | bf16 | 7.75GB | +1.6~1.8GB (참조 1~2장) | |
| | **fp8 저장** | **3.93GB** | +1.6~1.8GB | **서버 기본값** |
| VAE | bf16 | 0.17GB | encode +1.6GB / decode **+2.6GB** | VRAM은 activation이 차지 |
| | fp8 | 약 0.09GB | 동일 | 이득이 0.08GB뿐이라 **bf16 유지** |

### 9.2 배포 구성별 최대 VRAM

| 구성 | 상주 가중치 | 최대 VRAM |
| --- | --- | --- |
| Qwen bf16 + DiT bf16 + VAE bf16 | 16.0GB | 약 18.6GB (추정) |
| Qwen fp8 + DiT fp8 + VAE bf16 | 8.5GB | 약 11.1GB (추정) |
| Qwen fp8 (27층) + DiT fp8 + VAE bf16 | 약 7.6GB | 약 10.2GB (추정) |
| Qwen 없음 + DiT bf16 + VAE bf16 (`--no-fp8-weights`) | 7.9GB | **10.5GB (실측)** |
| **Qwen 없음 + DiT fp8 + VAE bf16 (서버 기본값)** | **4.1GB** | **6.7GB (실측)** |
| 위 + VAE tiling (타일 512) | 4.1GB | 약 5.9GB (추정, 미적용) |
| 위 + 얼굴 크롭 512px 생성 | 4.1GB | 4.8GB (실측, 결과가 달라짐 / 미적용) |

- 최대 VRAM = 상주 가중치 + 가장 큰 단계의 activation. 1MP에서는 VAE decode(+2.6GB)가 가장 크다.
- fp8 저장 품질: bf16 대비 입 영역 평균 픽셀 차이 2.4/255 (눈으로 구분 불가), 속도 +2% ([비교 이미지](assets/fp8_storage_quality.jpg)).

### 9.3 VAE decode VRAM과 tiling

decoder 마지막 단계가 `256 × 1024 × 1024` 텐서(bf16 537MB)를 다루고, ResNet 블록 안에서 이런 텐서 4~5개가 동시에 살아 있어
1MP에서 +2.6GB가 된다. 해상도 제곱에 비례한다 (512: 0.65GB, 768: 1.45GB, 1024: 2.56GB, 1536: 5.75GB).

`vae.enable_tiling()`은 기본 타일이 1024라 **1024 이하에서는 적용되지 않는다**. 타일 크기를 줄인 측정값 (1024 decode):

| 타일 | decode activation | decode 시간 | 차이 (랜덤 latent 기준) |
| --- | --- | --- | --- |
| 없음 | 2.56GB | 208ms | – |
| 512 | 0.66GB | 358ms | 0.68/255 |
| 384 | 0.39GB | 403ms | 0.99/255 |
| 256 | 0.19GB | 499ms | 1.61/255 |

서버에는 아직 옵션이 없다. 실제 사진에서 타일 경계가 보이는지 확인 후 `--vae-tile-size`(예: 기본 512)로 추가할 예정.

## 10. fp8 설정 시 소프트웨어 유의사항

1. **저장과 연산을 구분한다.**
   - fp8 **연산**(텐서코어)은 Ada(RTX 40, L4, L40S, RTX 6000 **Ada**), Hopper(H100), Blackwell(RTX 50, B200)부터 지원한다.
   - **RTX A6000 / A100 / RTX 30 (Ampere)은 fp8 저장만 가능**하다. RTX A6000(Ampere)과 RTX 6000 Ada를 혼동하지 않는다.
2. **현재 서버 방식** (`--fp8-weights`, `enable_layerwise_casting(storage_dtype=float8_e4m3fn, compute_dtype=bfloat16)`)
   - 어떤 GPU에서도 **항상 bf16으로 연산**한다. Ada/Hopper로 옮겨도 이 방식으로는 속도 이득이 없다.
   - bf16 가중치를 **scale 없이** 그대로 fp8로 변환한다. 이 모델에서는 차이가 작았지만, 다른 체크포인트(LoRA 병합 등)에서는 다시 확인한다.
   - 레이어마다 bf16 복사본을 만들었다 버리므로 약간 느려진다 (A6000 +2%).
3. **공식 fp8 체크포인트** (`black-forest-labs/FLUX.2-klein-4b-fp8`, 4.07GB, Apache 2.0)
   - transformer **만** 들어 있다. VAE / text encoder는 bf16 저장소의 것을 쓴다.
   - Linear 가중치 80개가 `F8_E4M3`이고, 각각 `weight_scale` / `input_scale`(F32)이 있는 **W8A8 static 양자화** 형식이다. norm / embedder / 출력층은 bf16.
   - Ada 이상: `input_scale`로 activation도 fp8로 바꿔 fp8 행렬곱 (torchao float8, `torch._scaled_mm`, ComfyUI fp8 fast 등 fp8 경로가 필요).
   - Ampere: `weight × weight_scale`로 bf16 복원 후 연산 (`input_scale` 미사용). scale이 보정되어 있어 현재 방식보다 정밀도가 안전하다.
   - diffusers 형식 변환본: `Photoroom/FLUX.2-klein-4b-fp8-diffusers` (torchao `AffineQuantizedTensor` state dict, 로드 스크립트 포함).
4. **품질 수치는 공개되어 있지 않다.** BFL은 속도 / VRAM(RTX 5080/5090에서 최대 1.6배 빠름, VRAM 최대 −40%)만 공개했다.
   배포 GPU에서 bf16 / fp8 저장 / fp8 연산을 같은 seed로 비교한 뒤 결정한다.
5. **지포스 Ada는 FP8 FP32-누산 성능이 절반**이라 데이터센터 GPU(L40S, H100)보다 이득이 작을 수 있다.
6. **양자화가 맞지 않았던 방법**: bitsandbytes 8bit / NF4는 single-file transformer 로드와 맞지 않아 실패했다.
   현재 환경에는 torchao가 설치되어 있지 않다 (fp8 연산 경로를 쓰려면 설치 필요).
7. **Text encoder를 fp8로 저장할 때** `nn.Embedding`과 tied `lm_head`는 제외해야 한다 (fp8 `index_select` 미지원 에러).
