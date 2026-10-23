# design_tune_server.py 배포 이미지 (docs/smile_tune/deployment_review.md)
#
# 빌드 (레포 루트에서):
#   DOCKER_BUILDKIT=1 docker build -f docker/design-tune-server/Dockerfile -t design-tune-server .
#
# 실행: 모델 가중치 / face_landmarker / 임베딩 라이브러리 / compile 캐시는 볼륨으로 마운트한다.
#   모델은 임베딩을 빌드했을 때와 같은 경로로 마운트해야 한다 (manifest의 model 경로와 --model이 같아야 추론 허용).
#   docker run --gpus '"device=0"' -p 5000:5000 \
#     -v /data1/jooyonglee/pretrained/flux2:/data1/jooyonglee/pretrained/flux2:ro \
#     -v $PWD/weights:/app/weights:ro \
#     -v $PWD/outputs/design_tune_embedding_libraries:/app/outputs/design_tune_embedding_libraries \
#     -v design-tune-inductor-cache:/cache/inductor \
#     design-tune-server --fp8-compute --compile regional

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ARG PYTHON_VERSION=3.11
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# libgl1/libegl1/libgles2/libglib2.0-0: mediapipe, opencv-contrib-python 네이티브 의존성
# build-essential: torch.compile(triton)이 런타임에 C 런처를 컴파일할 때 필요
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        libegl1 \
        libgl1 \
        libgles2 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# uv가 관리하는 Python에는 Python.h가 포함되어 있어 triton 컴파일이 동작한다
RUN curl -fsSL https://astral.sh/uv/0.9.5/install.sh -o /tmp/uv-install.sh \
    && sh /tmp/uv-install.sh \
    && rm /tmp/uv-install.sh
ENV PATH="/root/.local/bin:$PATH" \
    VIRTUAL_ENV=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python
RUN uv venv --python ${PYTHON_VERSION} --seed ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN uv pip install --no-cache-dir torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128

COPY docker/design-tune-server/requirements.txt /tmp/requirements.txt
RUN uv pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
COPY setup.py pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --no-cache-dir --no-deps . && rm -rf build
RUN uv pip check \
    && python -c "import cv2, mediapipe, torch; from diffusers import Flux2KleinPipeline, Flux2KleinInpaintPipeline; assert torch.__version__.split('+')[0] == '2.9.0'; assert torch.version.cuda == '12.8'"

COPY my/design_tune_server.py my/build_smile_embedding_library.py my/__init__.py my/openapi.yaml ./my/
COPY my/smile_design ./my/smile_design
COPY my/configs/smile_design ./my/configs/smile_design

# torch.compile 결과를 볼륨에 남겨 재시작 후 첫 요청의 컴파일 시간을 줄인다
ENV TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
    TRITON_CACHE_DIR=/cache/inductor/triton

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD curl -fsS http://127.0.0.1:5000/health || exit 1
ENTRYPOINT ["python", "my/design_tune_server.py", "--host", "0.0.0.0", "--port", "5000"]
