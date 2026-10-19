# 배포 준비 다음 작업 계획 (Docker / 4090 fp8·fp16 / 임베딩·weights 관리 / AWS)

> 초안. 세부 사항(특히 AWS 구성)은 실험 전에 확인이 필요한 부분을 "확인 필요"로 표시했다.

## 0. 현재 상태 요약

- 추론 서버: `my/design_tune_server.py` (Flask, `mask.mode` 기본값 `inpaint` — [final_report.md 0.5절](final_report.md#05-기본값을-composite에서-inpaint로-변경--mice-다중-영역-조합을-진짜-inpaint로-재구현)).
- 검증된 스타일 YAML 2종 (`my/configs/smile_design/`):
  - **`smile_final.yaml`** — 치아 전용(미백/정렬/템플릿 교체) + **head_yaw/head_pitch(고개 방향) 추가함**(이번 변경, 미백/정렬은 기존처럼 preset 옵션). 치아 축은 5명 × 실제 API로 가장 많이 검증됨 (0~5장, [final_report.md](final_report.md)). YAML 내용이 바뀌었으니 **임베딩 라이브러리 재빌드 필요**(pose 축 자체는 아직 미검증 — `smile_pose.yaml`에서 검증된 것과 같은 축이라 anchor 문구는 재사용했지만 `smile_final`에서 직접 돌려본 적은 없음).
  - **`pose-makeup-v2.yaml`** — 치아 + 메이크업 + 포즈 + (`mask.edits`로) 치아/입술/눈 동시 조합까지 이미 포함(변경 없음, 확인만 함). 범위는 더 넓지만 검증량은 `smile_final`보다 적다(포즈/메이크업 축은 [experiments_preset.md](experiments_preset.md) 수준, 다중 영역 inpaint 조합은 이번에 sample17 1건만 확인).
  - **배포 1순위 후보는 `smile_final.yaml`**(치아+포즈만 필요하면 이걸로). 메이크업까지 필요하면 `pose-makeup-v2.yaml`. 둘 다 배포 전 pose 축/다중 영역 조합의 추가 검증(여러 환자)이 필요하다.
- 런타임 의존성(`flask`, `mediapipe`, `opencv-python`/`opencv-contrib-python`)이 **`setup.py`/`pyproject.toml`에 기록되어 있지 않다** — 지금은 세션 중 임시로 pip install한 상태라, Docker 이미지 작업 전에 `my/requirements.txt` 같은 고정 목록이 필요하다.
- GPU: 지금까지 모든 실험은 **A6000**(RTX A6000, 48GB) 기준이다. 4090 관련 수치는 아직 없다.

## 1. 배포용 Docker 이미지

- `docker/diffusers-pytorch-cuda/Dockerfile`(uv 기반, CUDA 12.9 + Python)을 베이스로 재사용 가능 — 지금은 diffusers 코어 개발/테스트용이라, 배포용으로는 다음이 추가로 필요하다.
  - `my/` 런타임 의존성 고정 (`flask`, `mediapipe`, `opencv-python-headless` 권장 — 서버 환경엔 GUI용 `opencv-python` 불필요, 이미지 더 작음).
  - 모델 가중치(FLUX.2 Klein 4B 등, 17~23GB/모델)와 `weights/face_landmarker.task`를 이미지에 **포함할지 vs 런타임에 마운트/다운로드할지** 결정 필요 (이미지 크기 vs 배포 시 다운로드 시간 트레이드오프). 당장은 외부 볼륨 마운트를 권장(이미지가 가벼워야 반복 빌드/배포가 빠름).
  - 임베딩 라이브러리(`outputs/design_tune_embedding_libraries/`, 스타일당 0.1~1.8GB)도 마찬가지로 볼륨 마운트 권장.
- 체크리스트:
  1. `my/` 의존성 고정 목록 작성 (`requirements-my.txt` 또는 `pyproject.toml` extras).
  2. Dockerfile 작성 (베이스 이미지 + 의존성 설치 + `my/design_tune_server.py` 엔트리포인트).
  3. 가중치/임베딩 볼륨 마운트 규약 정리 (`/weights`, `/libraries` 같은 고정 경로).
  4. 컨테이너 기동 시 `/health`로 스모크 테스트.
  5. 이미지 크기 측정, `.dockerignore`로 불필요 파일(실험 스크립트, `my/samples/` 등) 제외.

## 2. 4090에서 DiT fp8/fp16 실험

목적: A6000(48GB, 지금까지의 모든 측정 기준) 대신 **RTX 4090(24GB)에서도 같은 속도/품질이 나오는지** 확인 —
VRAM이 절반이라 fp8 캐스팅(현재 `--fp8-weights` 기본 on, [README.md 9~10장](README.md)) 효과가 더 중요해진다.

- 비교 대상: transformer(DiT) 가중치 **fp8 저장 + bf16 연산**(현재 기본, `enable_layerwise_casting`) vs **fp16/bf16 그대로**.
- 측정 항목: 최대 VRAM(모델 로드 시 / 추론 중 피크), 추론 시간(4 step, inpaint 기준), 품질(기존 테스트 이미지로 육안 비교 — 4090은 Ampere가 아니라 Ada라 fp8 텐서코어 네이티브 연산 지원 여부도 확인 필요, A6000은 연산은 bf16으로 캐스팅해서 돌렸었다).
- 확인 필요: 4090 1장으로 모델(17~23GB) + 활성화 메모리가 24GB 안에 들어가는지(OOM 여유 확인), `--cpu-offload` 필요 여부.
- 결과는 이 리포트와 같은 형식(표 + 비교 이미지)으로 별도 문서나 `final_report.md` 추가 절에 정리.

## 3. 사전 임베딩(embedding library) 관리

- 현재는 `<libraries-dir>/<YAML 파일 이름>/<YAML 내용 SHA256>`에 로컬 디스크로만 쌓인다([README.md 7.1절](README.md)) — **이전 해시의 라이브러리가 자동 삭제되지 않는다**(스타일당 0.1~1.8GB, 지금 2개 스타일만으로 500MB+).
- 배포 전 정리할 것:
  - 배포에 쓸 스타일(YAML) 고정 → 해당 라이브러리만 이미지/볼륨에 포함, 나머지(실험용 해시)는 제외.
  - 버전 관리 방식 결정: 라이브러리를 Docker 이미지에 굽는지, 아니면 배포 때마다 빌드 스텝으로 생성하는지(후자가 "YAML=소스, 임베딩=빌드 산출물" 원칙에 더 맞음).
  - `PUT /api/v1/styles/{name}` / `build` API를 배포 환경에서 막을지 결정 (README 7.1절 마지막 항목에 이미 언급됨 — 운영 중 YAML이 바뀌는 걸 막으려면 필요).
  - 임베딩 저장소를 S3 같은 외부 스토리지로 옮길지(AWS 사용 계획과 연결, 3장 참고).

## 4. Weights 관리

- 지금 `weights/`에 섞여 있는 것:
  - `face_landmarker.task` (MediaPipe, Apache 2.0) — 실제 사용 중, 문제 없음.
  - `face_detection_yunet_2023mar.onnx` — 초기 face_detector.py 실험용, 현재 서버 경로(MediaPipe 기반)에서는 미사용으로 보임 — 배포 이미지에 포함할지 확인 필요.
  - `79999_iter.pth` (BiSeNet) — **라이선스 문제로 MediaPipe로 교체된 레거시** 자산, 배포에는 불필요.
  - `bisenet_celebA.pth` — 9바이트짜리 깨진 파일(다운로드 실패 잔재)로 보임, 사용되지 않으면 삭제 대상.
  - → 배포 전에 실제 런타임이 참조하는 파일만 남기고 정리 필요.
- FLUX.2 모델 가중치(HF 캐시, 17~23GB)는 라이선스가 Apache 2.0으로 상업 이용 가능([final_report.md 0.1절](final_report.md#01-distilled는-스텝-설정이-아니라-완전히-다른-가중치다)) — 배포 시 HF 허브에서 매번 받을지, 내부 저장소(S3 등)에 미러링해 둘지 결정 필요.
- 체크리스트: 실제 사용 weights 목록 확정 → 각 파일의 출처/라이선스/버전(revision) 기록 → 재현 가능한 다운로드 스크립트(지금 흩어져 있는 `gdown`/`huggingface_hub` 호출들을 한 곳에) → 배포용 저장소에 업로드.

## 5. AWS 사용 — 확인 필요한 부분

사용자가 "AWS 사용 예정"이라고 했는데, 구체적으로 어떤 용도인지 다음을 확인해야 계획을 더 구체화할 수 있다:

- **GPU 인스턴스**: AWS EC2에는 **RTX 4090 인스턴스가 없다** (일반적으로 A10G/L4/L40S/A100/H100 계열만 제공). 4090 실험은 온프레미스/베어메탈(또는 Lambda Labs, RunPod 등 4090을 제공하는 다른 클라우드)에서 하고, **AWS는 배포(추론 서빙) 쪽**에 쓰는 것인지 확인 필요.
  - fp8 텐서코어를 **하드웨어로 지원**하는 건 Ada(L4/L40S, `g6`/`g6e` 인스턴스)와 Hopper(H100, `p5` 계열) 세대부터다. **A10G(`g5`)·A100(`p4`, Ampere)은 fp8 하드웨어 연산을 지원하지 않는다** — 우리가 지금까지 쓴 A6000(Ampere, compute capability 8.6)도 마찬가지라 지금까지는 "fp8로 저장만 하고 연산은 bf16으로 캐스팅"해서 돌렸다([README.md 10장](README.md)). 받은 인스턴스에서 `nvidia-smi --query-gpu=name,compute_cap --format=csv`를 돌려서 GPU 이름을 확인하면, L4/L40S/H100이면 네이티브 fp8 연산 테스트가 가능하고 A10G/A100이면 지금 A6000과 같은 "저장만 fp8" 방식이 된다.
- **용도 후보**: (a) 모델 가중치/임베딩 라이브러리 저장용 S3, (b) Docker 이미지 레지스트리(ECR), (c) 실제 서빙 인스턴스(EC2 GPU 또는 SageMaker), (d) CI/빌드 파이프라인. 용도에 따라 다음 단계가 달라진다.
- 확정되면 이 문서에 AWS 아키텍처(스토리지/네트워크/GPU 인스턴스 타입/비용) 절을 추가.

## 6. 진행 순서 제안

1. `my/` 런타임 의존성 고정 + weights 정리 (3, 4장) — Docker 작업의 전제조건.
2. Docker 이미지 1차 빌드 (1장), 로컬에서 볼륨 마운트로 스모크 테스트.
3. 4090 fp8/fp16 실험 (2장) — 이미지가 준비되면 4090 장비에서 같은 이미지로 바로 실행 가능.
4. AWS 용도 확인(5장) 후 배포 아키텍처 확정, 임베딩/weights 저장소 위치 결정(3, 4장과 연결).
