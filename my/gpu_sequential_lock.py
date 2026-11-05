"""여러 프로세스(여기서는 design_tune_server와 img2vid backend_server)가 같은 GPU를 쓸 때,
동시에 GPU 작업이 겹치지 않도록 순서를 맞추는 선택적 장치.

기본은 꺼져 있다 (ENABLED = False, 각 서버가 지금처럼 독립적으로 동작). unified_server.py가 두 서버를
함께 띄울 때만 --serialize-gpu로 켠다. 켜져 있으면 fcntl 파일 락으로 프로세스 간에도 "요청 하나 끝난 뒤
다음 요청 처리"를 보장한다 (각 서버 내부의 자체 락은 같은 프로세스 안에서만 유효해 다른 프로세스의
GPU 사용과는 겹칠 수 있었다).
"""

from __future__ import annotations

import contextlib
import fcntl
import os

# 전역 변수로 켜고 끈다 (unified_server.py가 기동 시 set_enabled로 설정)
ENABLED = os.environ.get("UNIFIED_GPU_SEQUENTIAL", "") == "1"
LOCK_FILE = os.environ.get("UNIFIED_GPU_LOCK_FILE", "/tmp/unified_dental_gpu.lock")


def set_enabled(enabled: bool) -> None:
    global ENABLED
    ENABLED = enabled


@contextlib.contextmanager
def gpu_sequential_guard():
    """ENABLED가 아니면 아무 일도 하지 않는다. ENABLED면 다른 프로세스의 같은 guard와 상호 배제한다."""
    if not ENABLED:
        yield
        return
    fd = os.open(LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
