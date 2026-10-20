# 배포 검토: Docker 이미지 / ONNX Runtime vs Torch

[deployment_plan.md](deployment_plan.md) 1장(Docker)의 결과와, 추론 런타임을 ONNX Runtime으로 바꿀지에 대한 검토.
API 문서는 [api.md](api.md) (Swagger: 서버의 `/docs`).

## 요약

- 배포용 Dockerfile을 새로 작성했다: `docker/design-tune-server/` (Dockerfile, requirements.txt, Dockerfile.dockerignore).
- **실제 `docker build`는 아직 못 했다.** 작업 환경(로컬, 10.100.1.45 모두 컨테이너 안)에 docker가 없고, 권한상 컨테이너 안에서
  빌드할 수도 없다(아래 1.3절). 대신 Dockerfile 단계를 빈 venv에서 그대로 재현해 서버 기동 → 전체 API 호출까지 확인했다.
- **ONNX Runtime은 쓰지 않는 것을 권장한다.** diffusers에 FLUX.2 ONNX 파이프라인이 없고, 우리 서버 기능(inpaint, MICE hook,
  fp8 연산)이 PyTorch에 묶여 있다. 더 빠르게 하려면 다음 후보는 ORT가 아니라 TensorRT(transformer만)다.

## 1. Docker

### 1.1 기존 `docker/` 이미지 검토

| 이미지 | 용도 | 배포에 쓸 수 있나 |
| --- | --- | --- |
| `diffusers-pytorch-cuda` | diffusers CI (CUDA 12.9, torch 2.10) | 아니오 — diffusers를 GitHub main에서 설치(우리 코드가 아님), 테스트 의존성 포함, `numpy==1.26.4` 고정(mediapipe 1.1.1 환경은 numpy 2.x), 서버 의존성 없음, `CMD bash` |
| `diffusers-onnxruntime-cuda` | 레거시 ONNX Stable Diffusion CI | 아니오 — CUDA 12.1/Ubuntu 20.04, torch 버전 미고정(cu117 인덱스), uv 0.1.11 |
| 나머지 (cpu, minimum-cuda, xformers, doc-builder) | CI / 문서 빌드 | 아니오 |

uv + CUDA runtime 베이스 구조만 `diffusers-pytorch-cuda`에서 가져왔다.

### 1.2 새 이미지 `docker/design-tune-server/`

| 항목 | 결정 | 이유 |
| --- | --- | --- |
| 베이스 | `nvidia/cuda:12.8.1-runtime-ubuntu24.04` | torch 2.9.0 cu128 휠과 CUDA 버전을 맞춤 (CUDA 라이브러리는 torch 휠이 포함) |
| Python | 3.11 (uv 관리) | 실험 환경과 동일. uv Python은 `Python.h`를 포함해 triton(compile)이 동작 |
| 패키지 | `requirements.txt`에 전부 고정 | 4090 실험 환경과 같은 버전 ([fp8_4090_report.md](fp8_4090_report.md)) |
| diffusers | 이 레포의 `src/`를 설치 | 테스트한 코드와 동일하게. 이 브랜치는 `src/`를 수정하지 않았다(main과 동일) |
| apt | `libgl1 libegl1 libgles2 libglib2.0-0 build-essential curl` | mediapipe/opencv 네이티브 라이브러리, compile 시 C 컴파일러 |
| 코드 | 서버 실행에 필요한 파일만 COPY (`Dockerfile.dockerignore`) | 실험 스크립트, 샘플, outputs 제외 |
| 볼륨 | 모델 가중치, `weights/`, 임베딩 라이브러리, compile 캐시 | 이미지를 가볍게 유지 ([deployment_plan.md](deployment_plan.md) 1장 권장) |
| 엔트리포인트 | `python my/design_tune_server.py --host 0.0.0.0 --port 5000` + 추가 인자 | `docker run ... design-tune-server --fp8-compute --compile regional` |

```bash
# 빌드 (레포 루트, BuildKit 필요 — Dockerfile.dockerignore를 쓰기 때문)
DOCKER_BUILDKIT=1 docker build -f docker/design-tune-server/Dockerfile -t design-tune-server .

# 실행
docker run --gpus '"device=0"' -p 5000:5000 \
  -v /data1/jooyonglee/pretrained/flux2:/data1/jooyonglee/pretrained/flux2:ro \
  -v $PWD/weights:/app/weights:ro \
  -v $PWD/outputs/design_tune_embedding_libraries:/app/outputs/design_tune_embedding_libraries \
  -v design-tune-inductor-cache:/cache/inductor \
  design-tune-server --fp8-compute --compile regional
```

**볼륨 주의**

- **모델은 임베딩을 빌드했을 때와 같은 경로로 마운트해야 한다.** 서버는 임베딩 manifest의 `model` 경로와 `--model`이
  문자열로 같아야 추론을 허용한다(다르면 `409`). 위 예시는 호스트와 같은 `/data1/...` 경로로 마운트한다.
- 컨테이너 안에서 임베딩을 빌드하려면(`POST /styles/{name}/build`) 텍스트 인코더 모델(klein-base-4B)도 같은 경로에 있어야 한다.
  위 예시처럼 `flux2` 폴더 전체를 마운트하면 둘 다 들어간다.
- `weights/`는 `face_landmarker.task`만 쓴다(나머지는 레거시, [deployment_plan.md](deployment_plan.md) 4장).
  이 파일이 없으면 마스크 요청이 `404`로 실패한다 — "스타일 없음"과 같은 코드라 헷갈리기 쉽다.
- **compile 캐시 볼륨**: `--compile`은 첫 요청에서(MICE를 쓰면 MICE 첫 요청에서 한 번 더) GPU 커널을 컴파일하고 결과를
  `TORCHINDUCTOR_CACHE_DIR`에 저장한다(해상도가 바뀌어도 다시 컴파일하지 않는다). 기본 위치(`/tmp`)는 컨테이너를 새로
  만들 때마다 지워지므로 볼륨에 둔다. 측정(첫 요청): full compile 251초 → 캐시가 남아 있으면 92초, `--compile regional`은
  98초 → 34초(모델 로드 + 그래프 추적은 매번 다시 함). 캐시 크기 약 270MB. GPU 종류, torch 버전, 코드가 바뀌면 다시 컴파일한다.
  상세는 [robustness_report.md](robustness_report.md) 3장.

### 1.3 검증한 것 / 못 한 것

작업 환경은 docker socket이 없고, `CAP_SYS_ADMIN`이 없으며 user namespace도 막혀 있어 Docker-in-Docker,
buildah, podman 모두 쓸 수 없다. 그래서 **Dockerfile 단계를 손으로 재현**했다.

| 단계 | 방법 | 결과 |
| --- | --- | --- |
| Python + 패키지 설치 | 빈 uv venv (Python 3.11)에 Dockerfile과 같은 순서로 torch → requirements.txt → `src/` 설치 | 성공, `pip check` 통과. **`importlib_metadata` 누락을 발견해 requirements.txt에 추가** (diffusers를 `--no-deps`로 설치하기 때문) |
| COPY 범위 | Dockerfile이 COPY하는 파일만 별도 디렉터리에 복사 | 이 파일들만으로 서버 기동 성공 |
| 볼륨 | 심링크로 대체 | |
| API | `/health`, `/docs`, `/openapi.json`, styles, build, mask, infer, 에러(400/404) | 모두 정상. infer 2.0초(1024×1024, fp8 연산 + compile) |
| compile 캐시 | 같은 ENV로 재시작 | 254초 → 83초 |
| Swagger UI | headless Chromium으로 `/docs` → Try it out → Execute | 200, 결과 이미지 표시 |

**아직 확인 안 된 것** (docker가 있는 호스트에서 확인 필요):

- 베이스 이미지 위에서 apt 패키지 목록이 충분한지 (작업 환경은 Ubuntu 22.04에 GL 라이브러리를 따로 설치한 상태라 조건이 다르다)
- uv 설치 스크립트, `Dockerfile.dockerignore` 적용, `HEALTHCHECK`, `--gpus` 전달
- 이미지 크기: venv만 7.3GB(이 중 CUDA 라이브러리 4.3GB, torch 1.7GB, triton 0.6GB) + 베이스 + apt → **약 9~10GB 예상**

## 2. ONNX Runtime vs Torch

| 항목 | Torch (현재) | ONNX Runtime |
| --- | --- | --- |
| FLUX.2 Klein 파이프라인 | diffusers에 있음 (`Flux2KleinPipeline`, `Flux2KleinInpaintPipeline`) | **없음.** diffusers의 ONNX 파이프라인은 레거시 Stable Diffusion 1.x용(`OnnxStableDiffusion*`)뿐 |
| 변환 작업 | 없음 | transformer(4B)·VAE를 직접 export. 2GB 넘는 모델이라 external data 형식, 가변 해상도는 dynamic axes 필요. 스케줄러, latent packing, inpaint 마스크 블렌딩, 참조 이미지 처리를 파이프라인 밖에서 다시 구현해야 함 |
| 우리 서버 기능 | 그대로 동작 | MICE(`mask.mode: mice`)는 transformer에 PyTorch forward hook을 걸어 attention bias를 넣는 방식이라 ONNX 그래프에서는 못 쓴다. 사전 임베딩 주입(`prompt_embeds`)은 가능 |
| fp8 연산 (4090) | torchao + compile로 확인 (1.74배, [fp8_4090_report.md](fp8_4090_report.md)) | ORT CUDA EP에는 fp8 GEMM contrib op(`GemmFloat8`)가 있지만, 모델 그래프를 직접 그 op로 바꾸고 스케일을 넣어야 한다 — FLUX 같은 모델을 fp8 ORT 모델로 바꾸는 표준 도구가 없다. 실용적인 fp8 경로는 TensorRT(EP 또는 단독) |
| 얼굴 마스크 | mediapipe (TFLite) — 런타임 무관 | 동일 (ORT로 바꿀 대상이 아님) |
| 이미지 크기 | 9~10GB 예상 | torch를 완전히 빼면 줄어들지만, 전처리/후처리를 다시 짜야 하고 이득은 검증되지 않음 |

**결론: Torch를 유지한다.** ORT로 바꾸면 변환·재구현 비용이 크고, 4090에서 가장 효과가 컸던 fp8 연산을 ORT에서 쓰려면
그래프를 직접 고치거나 결국 TensorRT를 써야 한다. 추가 속도가 필요하면 다음 후보는 **transformer만 TensorRT로 바꾸는 것**(Torch-TensorRT 또는 NVIDIA ModelOpt
fp8 → TensorRT 엔진)이고, 파이프라인 나머지는 Torch로 둔다. 이건 아직 측정하지 않았다 — 하게 되면 현재 기준선
(fp8 연산 + compile, transformer forward 346ms, 요청당 약 2초 @1024×1024)과 비교한다.

## 3. Swagger / API 문서

- 스펙: `my/openapi.yaml` (OpenAPI 3.0.3, `openapi-spec-validator` 통과). 서버가 `/openapi.yaml`, `/openapi.json`으로 제공.
- UI: 서버의 `/docs` — Swagger UI 5.17.14를 jsdelivr CDN에서 불러온다. 브라우저가 인터넷에 못 나가는 환경이면
  swagger-ui-dist 파일을 이미지에 넣어 서버가 직접 제공하도록 바꿔야 한다.
- 스펙은 손으로 쓴 파일이라 **서버 코드를 바꾸면 같이 고쳐야 한다**(파라미터, 에러 코드).
- `parameters`는 multipart 안의 JSON 문자열이라 Swagger UI에서는 문자열 입력칸으로 보인다. 스키마는 Schemas의
  `InferParameters` / `MaskOptions`에 정리했다.

## 4. 배포 전 남은 일

1. docker가 있는 호스트에서 실제 빌드·실행 (1.3절 미확인 항목).
2. **인증/접근 제한이 없다.** 스타일 수정(`PUT`/`DELETE`)과 빌드 API까지 네트워크에 그대로 열린다. 배포에서는 이 API를
   막거나(옵션으로 끄기) 앞단 프록시에서 제한해야 한다 ([deployment_plan.md](deployment_plan.md) 3장).
3. 웹 브라우저 클라이언트가 다른 도메인에서 호출하면 CORS 설정이 필요하고, 메타데이터를 읽으려면
   `Access-Control-Expose-Headers: X-Design-Tune-Metadata`도 필요하다(Swagger UI는 같은 도메인이라 해당 없음).
4. 서버가 Flask 개발 서버(Werkzeug)로 뜬다. GPU 추론은 어차피 락으로 한 번에 하나씩 처리되지만, 운영용으로는
   WSGI 서버(gunicorn 1 worker + 스레드)로 바꾸는 것을 검토.
5. `--compile` warmup: 서버 시작 직후 대표 해상도로 한 번씩 호출하는 스크립트, 또는 입력 해상도를 몇 가지로 고정.
