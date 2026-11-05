# FLUX.2 + Wan2.2 통합 서버 — 핵심 요약

## 구조
- 서버 2개를 각자 그대로 유지, **합치지 않음** (프로세스/포트 분리: design_tune_server:5000, backend_server:8000)
- `my/unified_server.py`: 둘을 한 번에 띄우는 런처 (subprocess, SIGTERM 한 번에 같이 종료)
- `my/unified_client.py`: FLUX.2 편집 → 그 결과 이미지로 Wan2.2 영상 생성, 순서대로 호출

## 동시 GPU 사용 방지
- `gpu_sequential_lock.py` (양쪽 레포에 동일 파일): fcntl 기반 **프로세스 간** 락
- `--serialize-gpu`(flux) / `--serialize_gpu`(wan) 옵션으로 켜고 끔, 전역 변수(`ENABLED`)로 관리, 기본 off
- 켜면 한쪽이 GPU 쓰는 동안 다른 쪽 요청은 끝날 때까지 대기 (동시 실행 안 함) — 실측으로 검증됨
- `unified_server.py`는 기본으로 이 옵션을 켜서 둘을 띄움

## VRAM 테스트 결과 (24GB GPU)
- 순차 실행이면 OOM 없음 (여러 번 테스트 완료)
- **기본값(121프레임, wan-turbo 4스텝, 704×1280)에서 피크 24057MiB / 24564MiB — 여유 507MB(2%)뿐, 매우 빠듯함**
- 짧은 테스트(17프레임)는 여유 1.3GB로 그보다는 안전

## 속도 (wan-turbo 1회 생성, 121프레임 기준 실측)
- T5 텍스트 인코딩(CPU): 58.7s ← 가장 큰 병목
- VAE 인코딩: 9.1s / 디노이징 4스텝: 24.0s / VAE 디코딩: 81.8s
- 총 ~174s + FLUX.2 편집 8.5s ≈ 182s(3분)/건

## 후속 개선 (iSmileNet 쪽 커밋에서 반영됨)
- `backend_server..py`에 `--prompt_embeddings` 캐시 추가 → `style=` 파라미터로 호출 시 T5 인코딩(58.7s) 스킵 가능
- 캐시 파일: `smiledesign/img2vid/prompt_embeddings/wan22_turbo_smile_styles.pt`

## 구동/종료 명령
→ `my/README.md`의 "통합 서버 구동" 절 참고
