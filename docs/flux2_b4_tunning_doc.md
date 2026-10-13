# client flux2 튜닝 데모

# 초기(현재) apps 상태 
 * 기존 apps\smiledesign 폴더 통째로 apps\diffusion_tune 폴더로 복사
 * apps\smiledesign 는 flux2 4b 모델을 테스트하기 위한 데모앱 prompt입력창이 있다.
 * 해당 앱을 수정하여 flux2 스타일을 사전에 세팅하고 ui 조절바를 통해 편하게 편집하는 기능을 제공하도록한다.


# diffusion 계열의 실제 앱 스타일을 수정세팅, 그리고 실제 스타일 가중치를 변경하는 튜닝앱
  * 해당 apps은 client 로 동작
  * 서버에 해당하는 기능은(serveㄱ아직 미구현 / 추론 기능만 구현)  다음 저장소에 정의
    * tools\aimodule\mydiffuser
  * server / 서버 주요 기능은 크게 두가지이다.
     
     * 스타일 text2encoder embedding 빌드
     * 지정 스타일 가중치 설정해서 입력 이미지 조합후 추론
     * 문서는 여기를 참고 \tools\aimodule\mydiffusers\docs\flux2_klein_smile_design_cached_embedding_implementation.md
  * client 영역은
     * 해당 스타일 지정을 자유롭게 변경가능해야하며 (추가, 수정, 삭제, 업데이트), 이를 서버에 업데이트 요청한다.(서버에서는 ebedding을 재 빌드해야함)
     * 그리고 입력 스타일에 대해 가중치를 자유롭게 설정할수 있는 슬라이더를 통해 이미지를 선택해서 editting한다


## 구현된 client UI

`apps/diffusion_tune/main.py`를 실행한다.

```powershell
python apps/diffusion_tune/main.py
```

- `Tune Image`: reference 이미지 선택, layout/mode 선택, YAML에서 동적으로 생성된 style slider 조절, 추론 및 결과 비교
- `Style Definitions`: YAML 원문 편집, 문법/schema 검증, control/anchor/preset 편집 helper
- control, anchor, preset 개수는 고정하지 않는다. 유효한 YAML에 정의된 개수만큼 UI가 동적으로 생성된다.
- YAML이 실제 source of truth이다. helper는 YAML을 안전하게 생성하는 보조 UI이며 고급 layout/grid/template 편집은 YAML에서 직접 처리한다.
- YAML 검증은 문법 위치, 필수 key, template placeholder, default/anchor, preset 값, grid 참조를 확인한다.
- `Build Embeddings`는 YAML을 서버에 먼저 업데이트한 뒤 embedding build를 요청한다. 서버가 `job_id`를 반환하면 완료될 때까지 상태를 polling한다.

기본 설정은 `apps/diffusion_tune/styles/smile_v1.yaml`에 있다.

## REST API 계약

서버 구현은 아직 별도 작업이다. client는 다음 계약을 사용한다. API path는
`apps/diffusion_tune/flux2_backend.py` 한 파일에 모여 있으므로 서버 계약이 달라지면 이 파일만 수정한다.

| Method | Path | 용도 |
| --- | --- | --- |
| `GET` | `/health` | 서버 상태 확인 |
| `GET` | `/api/v1/styles` | style 목록 조회 |
| `GET` | `/api/v1/styles/{name}` | YAML 조회 (`text/yaml` 또는 `{ "yaml": "..." }`) |
| `PUT` | `/api/v1/styles/{name}` | YAML 추가/업데이트 (`{ "yaml": "..." }`) |
| `DELETE` | `/api/v1/styles/{name}` | style 삭제 |
| `POST` | `/api/v1/styles/{name}/build` | embedding library build 시작 |
| `GET` | `/api/v1/jobs/{job_id}` | build 상태 조회 |
| `POST` | `/api/v1/infer` | multipart reference 이미지와 inference parameter로 추론 |

추론 요청은 `images` multipart part를 layout role 순서대로 보내며 `parameters` form field에
다음 JSON을 보낸다.

```json
{
  "style_name": "smile-v1",
  "layout": "patient_tooth",
  "mode": "grid",
  "grid": "whitening_smile",
  "control_values": {
    "whitening": 63,
    "smile_intensity": 82
  },
  "seed": 42
}
```

추론 응답은 이미지 binary, `{ "image_base64": "..." }`, `{ "image_url": "..." }` 중 하나를 지원한다.

`width`, `height`는 REST 추론 요청에 보내지 않는다. 서버는 첫 번째 입력 이미지 크기를 최종 출력 크기로 사용한다.
모델에 전달할 내부 크기는 각 축을 16의 배수로 조정하고, 추론이 끝난 이미지는 원본 입력 크기로 resize해서 반환한다.
client도 이전 버전 서버와의 호환을 위해 응답 크기가 첫 번째 입력 이미지와 다르면 최종 resize를 수행한다.
    

