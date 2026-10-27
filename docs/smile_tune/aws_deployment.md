# EC2 + S3 배포 초안

구조는 `Git 소스 → Docker 이미지 → EC2 GPU 서버`와 `S3 → EC2 데이터 디스크`로 나눈다.
S3에서 직접 추론하지 않는다. 시작 스크립트가 파일을 내려받은 다음 기존 Torch 서버를 실행한다.
이미지에 AWS 키, 가중치, 임베딩은 넣지 않는다.

## 설정 파일

- `docker/design-tune-server/Dockerfile`: 기존 로컬 이미지. uv 설치 실패를 감지하도록 수정하고 의존성·네이티브 import 검사를 추가했다.
- `Dockerfile.aws`: 로컬 이미지를 상속하고 AWS CLI와 S3 시작 스크립트를 추가한다.
- `compose.aws.yaml`: EC2의 GPU, 데이터 디스크, 포트, AWS 환경변수를 지정한다.
- `deployment.yaml`: S3 하위 경로와 컨테이너 경로, 필수 파일, 서버 실행 인자를 지정한다.

기본값은 **fp8 가중치 저장 + bf16 연산**, compile 없음이다. `--fp8-compute`와 AOT는 활성화하지 않았다.
fp8 연산 실험은 `deployment.yaml`의 `server_args`에 `--fp8-compute`를 추가한다.
regional compile까지 실험할 때만 `--compile`, `regional`을 별도 항목으로 추가한다.
가중치 저장과 fp8 연산은 서로 다른 옵션이다.

## S3에 올릴 파일

테스트·운영 전환 시 버킷 이름을 반드시 바꿀 필요는 없다. 다른 버킷을 받으면 `S3_BUCKET`을 바꾼다.
파일 묶음은 `S3_PREFIX`로 구분한다. 사용 중인 묶음을 덮어쓰지 말고 새 release 경로에 올린다.

```text
s3://<bucket>/design-tune/releases/v1/
  models/klein-4b/                  전체 모델 폴더, model_index.json 포함
  tokenizers/klein-base-4b/         임베딩 빌드에 사용한 base 모델의 tokenizer/ 내용
  libraries/smile_final/<hash>/    manifest.json, negative.pt, embeddings/ 전체
  mediapipe/face_landmarker.task
```

모델의 Hugging Face snapshot 폴더에는 실제 파일 대신 blobs를 가리키는 심볼릭 링크가 있을 수 있다.
업로드는 링크가 가리키는 실제 파일을 따라가야 한다. 아래 명령의 `--follow-symlinks`는 그 용도다.
[AWS CLI sync 문서](https://docs.aws.amazon.com/cli/latest/reference/s3/sync.html)를 참고한다.

아래는 **파일이 있는 로컬 환경에서 직접 실행할 업로드 명령**이다. 버킷·리전은 실제 값으로 바꾼다.

```bash
export S3_BUCKET=YOUR_TEST_BUCKET
export AWS_DEFAULT_REGION=ap-northeast-2
export S3_PREFIX=design-tune/releases/v1
export MODEL=/data1/jooyonglee/pretrained/flux2/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294
export ENCODER=/data1/jooyonglee/pretrained/flux2/models--black-forest-labs--FLUX.2-klein-base-4B/snapshots/a3b4f4849157f664bdbc776fd7453c2783562f4d
export LIBRARY_HASH=ac177319ac4a25f8cd09fbbdc17a82d1a5dd5212f5801dd2a5992ce58254c467

aws s3 sync "$MODEL/" "s3://$S3_BUCKET/$S3_PREFIX/models/klein-4b/" --follow-symlinks
aws s3 sync "$ENCODER/tokenizer/" "s3://$S3_BUCKET/$S3_PREFIX/tokenizers/klein-base-4b/" --follow-symlinks
aws s3 sync "outputs/design_tune_embedding_libraries/smile_final/$LIBRARY_HASH/" \
  "s3://$S3_BUCKET/$S3_PREFIX/libraries/smile_final/$LIBRARY_HASH/"
aws s3 cp weights/face_landmarker.task "s3://$S3_BUCKET/$S3_PREFIX/mediapipe/face_landmarker.task"
```

`LIBRARY_HASH`는 현재 `smile_final.yaml`에 대응한다. YAML을 바꾸면 임베딩을 다시 만들고
`deployment.yaml`의 필수 manifest 경로도 바꾼다. 파일명·해시 디렉터리를 유지해야 서버가 찾는다.
기존 `264b10...` 임베딩은 `/root/.cache/...` 모델 경로를 기록하므로 이 기본 설정과 맞지 않는다.
서버의 `--model`과 선택한 manifest의 `model` 문자열이 같아야 한다. S3 다운로드 경로도 여기에 맞춘다.
manifest만 임의로 고쳐서 다른 모델의 임베딩을 재사용하지 않는다.

MICE는 사전 임베딩이 있어도 토큰 길이 계산용 토크나이저가 필요하다.
기본 배포에는 base 모델의 텍스트 인코더 가중치를 올리지 않는다. 따라서 서버에서 임베딩을 새로 빌드하는
`POST /styles/{name}/build`는 이 구성의 대상이 아니다. 새 스타일은 로컬에서 빌드하여 새 release로 올린다.
서버 내 빌드가 필요하면 전체 base 모델을 추가로 내려받고 `--text-encoder-model` 경로를 맞춰야 한다.
MediaPipe를 사용하지 않기로 결정하면 `assets`의 mediapipe 항목과 해당 `required_files`를 함께 제거한다.

## EC2 준비

x86_64 NVIDIA GPU 호스트에 Docker, Docker Compose 플러그인, NVIDIA 드라이버와 Container Toolkit이 필요하다.
호스트의 `nvidia-smi`와 컨테이너 GPU 접근부터 확인한다.
데이터 디스크에는 모델·임베딩·이미지·다운로드 중 임시 파일을 담을 여유 공간을 확보한다.

EC2에는 S3 읽기용 IAM role을 연결한다. 이미지나 Git에 access key를 기록하지 않는다.
해당 버킷의 `s3:ListBucket`(release prefix 범위), 파일의 `s3:GetObject` 권한이 필요하다.
SSE-KMS를 사용하면 해당 키의 `kms:Decrypt` 권한도 필요하다.
[AWS 자격 증명 문서](https://docs.aws.amazon.com/boto3/latest/guide/credentials.html)에 설명된 기본 자격 증명 체인을 사용한다.
컨테이너에서 EC2 role을 쓰도록 IMDSv2를 활성화하고 metadata response hop limit은 2로 설정한다.
[EC2 metadata 설정 문서](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/configuring-IMDS-new-instances.html)를 참고한다.
네트워크에서 S3 및 필요한 AWS 서비스에 접근할 수 있어야 한다.

## 빌드와 실행

레포 루트에서 실행한다. 실제 EC2에서는 버킷·리전·데이터 디스크 위치를 지정한다.
`ASSET_DIR`은 절대 경로로 지정하고 release마다 다른 디렉터리를 사용한다.

```bash
export S3_BUCKET=YOUR_TEST_BUCKET
export AWS_DEFAULT_REGION=ap-northeast-2
export S3_PREFIX=design-tune/releases/v1
export ASSET_DIR=/srv/design-tune/releases/v1
mkdir -p "$ASSET_DIR/models" "$ASSET_DIR/assets"

DOCKER_BUILDKIT=1 docker build \
  -f docker/design-tune-server/Dockerfile -t design-tune-server:local .
docker run --rm --gpus all --entrypoint python design-tune-server:local \
  -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0))'

docker compose -f docker/design-tune-server/compose.aws.yaml config --quiet
docker compose -f docker/design-tune-server/compose.aws.yaml build
docker compose -f docker/design-tune-server/compose.aws.yaml up -d
docker compose -f docker/design-tune-server/compose.aws.yaml logs -f server
```

Compose는 로컬 이미지가 먼저 빌드되어 있어야 AWS 이미지를 만들 수 있다.
ECR을 사용할 때는 사용자가 검토한 이미지를 직접 게시하고 `DESIGN_TUNE_IMAGE`에 이미지 주소를 지정한다.
그 경우 EC2에서 이미지를 pull한 뒤 `up -d --no-build`로 실행할 수 있다.

시작 시 `aws s3 sync`는 변경된 파일을 내려받고 로컬의 기존 파일을 삭제하지 않는다.
S3 접근 실패나 필수 파일 누락 시 서버를 시작하지 않는다. release마다 새 `ASSET_DIR`을 사용하는 이유는
다른 release의 잔여 파일이 섞이는 것을 막기 위해서다. 같은 데이터 디렉터리를 여러 서버가 동시에 쓰지 않는다.
초기 다운로드를 고려해 Compose의 healthcheck start period는 30분이다. 다운로드 시간 제한은 아니다.

```bash
curl -fsS http://127.0.0.1:5000/health
curl -fsS http://127.0.0.1:5000/api/v1/styles
```

`/health`의 200은 HTTP 서버 기동 확인이다. 모델은 첫 추론 때 로드되므로 GPU 추론 성공을 의미하지 않는다.
두 번째 명령의 경로는 `/api/v1/styles`다(`/styles`가 아니다).
`/docs` 또는 [기존 API 사용법](api.md)으로 `smile_final`을 선택하여 실제 이미지 추론도 확인한다.
앱이 실제로 쓰는 엔드포인트는 **`POST /api/v1/design`**이므로 이쪽도 함께 확인한다([tunning_guide.md](tunning_guide.md)).
일반 추론과 MediaPipe 마스크를 사용하는 요청을 각각 실행해야 한다.
이후 `docker compose ... down`과 `up -d`로 재시작 후 데이터 재사용도 확인한다.

포트는 기본적으로 EC2의 `127.0.0.1:5000`에만 바인딩한다.
외부 서비스로 연결하려면 앞단 인증·TLS 프록시를 구성하고 스타일 수정·삭제·빌드 API 접근을 제한한다.
현재 실행 방식은 Flask 개발 서버다. 운영용 WSGI 전환은 별도 작업이며,
이 구성만으로 공개 운영 배포가 완료되는 것은 아니다.

## 최종 의존성 정리

### 이미지에 들어가는 것 / 들어가지 않는 것

`Dockerfile.dockerignore`는 **허용 목록 방식**이다. 첫 줄 `*`로 전부 막고 필요한 코드만 `!`로 통과시킨다.
그래서 문서·샘플 이미지·실험 결과·모델 가중치는 **애초에 빌드 컨텍스트에 들어가지 않는다.**

| 들어간다 | 크기 |
| --- | --- |
| `src/` (diffusers 본체) | 26.1MB — **설치 후 `/app/src`는 Dockerfile에서 삭제한다** |
| `my/design_tune_server.py`, `my/smile_design/`, `my/openapi.yaml`, `my/__init__.py`, `my/build_smile_embedding_library.py` | 0.15MB |
| `my/configs/smile_design/*.yaml` (스타일 9개) | 0.06MB |
| `setup.py`, `pyproject.toml`, `README.md` | 0.02MB — `setup.py`가 README를 `long_description`으로 읽는다 |
| `docker/design-tune-server/requirements.txt` | — |

**빌드 컨텍스트 합계 26.4MB, 파일 1,068개.**

| 들어가지 않는다 | 이유 |
| --- | --- |
| `docs/`, `*.md`, 보고서 그림 | 실행에 불필요 |
| `my/samples/`, `outputs/` | 샘플·실험 결과 이미지 (수백 MB) |
| `weights/` | `face_landmarker.task`는 S3에서 받아 `/assets/mediapipe`에 둔다 |
| `/data1/...` 모델 가중치 | S3에서 받아 볼륨에 둔다 |
| 임베딩 라이브러리 | S3에서 받아 `/assets/libraries`에 둔다 |
| `examples/`, `tests/`, `benchmarks/`, `scripts/` | 라이브러리 예제·테스트 |
| `__pycache__`, `*.pyc`, `*.egg-info` | 빌드 부산물 |

컨텍스트에 무엇이 들어가는지는 Docker 없이도 확인할 수 있다. `Dockerfile.dockerignore`의 규칙을
순서대로 적용해 파일 목록을 뽑아 보면 된다 (마지막에 일치한 규칙이 이긴다).

### 런타임 패키지 15개

`requirements.txt`에 **왜 필요한지를 줄마다 주석으로 적어 두었다.** 전부 버전을 고정한다 —
추론 결과가 라이브러리 버전에 따라 달라질 수 있기 때문이다.

| 분류 | 패키지 | 용도 |
| --- | --- | --- |
| 모델 로드/추론 | `transformers`, `accelerate`, `safetensors`, `huggingface_hub`, `importlib_metadata`, `regex`, `requests` | 가중치 로딩, 설정 클래스, 토크나이저 |
| | `torchao` | **`--fp8-compute` 전용**(지연 import). 배포 기본값 `--fp8-weights`만 쓸 거면 빼도 동작한다 (약 41MB 절약) |
| 얼굴 마스크 | `mediapipe`, `opencv-contrib-python` | inpaint 영역 검출. 합계 약 220MB로 가장 크다 |
| HTTP | `flask`, `werkzeug` | |
| 데이터 | `pyyaml`, `pillow`, `numpy` | 스타일 YAML, 이미지 입출력, 마스크 연산 |

- `torch`는 `requirements.txt`가 아니라 Dockerfile에서 **CUDA 12.8 인덱스**로 따로 설치한다.
- `diffusers`는 PyPI가 아니라 **이 레포의 `src/`**를 설치한다 (`uv pip install --no-deps .`).
- `mediapipe`가 `opencv-contrib-python`(GUI 빌드)을 강제하므로 headless 버전으로 바꿀 수 없다.
  그래서 apt로 `libgl1`, `libegl1`, `libgles2`, `libglib2.0-0`을 설치한다.
- `build-essential`은 `torch.compile`(triton)이 런타임에 C 런처를 컴파일할 때 필요하다. `--compile`을 쓰지 않으면 빼도 된다.

### 배포 전에 정리할 수 있는 것

| 항목 | 효과 | 주의 |
| --- | --- | --- |
| `torchao` 제거 | 약 41MB | `--fp8-compute`를 못 쓴다 |
| `build-essential` 제거 | 수백 MB | `--compile`을 못 쓴다 |
| 스타일 YAML을 배포용 1개로 좁히기 | 용량은 미미 | 쓰지 않는 스타일이 `/api/v1/styles`에 `library_ready: false`로 보이는 것을 막는다 |
| `my/build_smile_embedding_library.py` 제거 | 미미 | 서버 내 임베딩 빌드가 불가능해진다 (이 배포 구성은 어차피 텍스트 인코더를 올리지 않는다) |

## 검증 범위

이 변경 환경에는 Docker 실행 파일과 소켓이 없어 이미지 빌드, EC2/S3 연결, 컨테이너 GPU 추론은 검증하지 못했다.
Python 시작 스크립트의 다운로드 실패·파일 누락 처리와 서버 실행 인자 전달 테스트 4개는 통과했다.
Compose 설정 파싱, 현재 YAML의 임베딩 해시·모델 경로 일치, Ruff 검사도 통과했다.
실제 Docker 이미지의 apt 라이브러리 충족 여부와 MediaPipe 동작은 위 빌드·추론 단계에서 확인해야 한다.
