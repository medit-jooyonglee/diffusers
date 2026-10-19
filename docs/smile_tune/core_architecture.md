# Core 아키텍처: 추론 서버 ↔ diffusers 코어 연결 구조

> 디버깅용 기술 참조 문서. "무슨 실험을 했는지"는 [final_report.md](final_report.md),
> "API를 어떻게 쓰는지"는 [README.md](README.md) 참고. 여기는 **코드가 어디서 분기되고,
> 파라미터가 어디로 흘러가는지**만 다룬다.

## 1. 요청 1건의 전체 흐름

```
HTTP POST /api/v1/infer (multipart: parameters=JSON, images=파일들)
  → parse_parameters()                          [server.py:850]
  → StyleStore.read(style_name)                 [server.py:216]  — YAML 로드 + SHA256 해시로 라이브러리 경로 결정
  → EmbeddingLibrary(library_path)               [embedding_library.py:13]  — manifest.json만 읽음, 모델 없음
  → load_uploaded_images()                       [server.py:865]
  → InferenceEngine.infer(library, config, parameters, images)   [server.py:526]
      ├─ resolve_mask_options()                  [server.py:76]   — mask.mode 등 파라미터 정규화
      ├─ (mask.mode별 분기, 4장 참고)
      │    prompt_embeds 하나를 만든다 (텍스트 인코더 호출 없음, 디스크에서 로드/보간만)
      ├─ self.load() 또는 self.load_inpaint()    [server.py:465 / 507]  — diffusers 파이프라인 인스턴스 선택
      └─ pipeline(**call_kwargs)                 ← **여기서 diffusers 코어로 넘어간다**
  → PNG 인코딩 + 메타데이터(PngInfo, X-Design-Tune-Metadata 헤더)
```

**핵심 원리**: 서버(`my/`) 쪽은 "어떤 프롬프트 임베딩을 쓸지, 어떤 마스크를 만들지, 어떤 파이프라인 객체를
쓸지"만 결정한다. 실제 디노이징 루프·transformer forward·VAE encode/decode는 전부 `diffusers` 코어
(`src/diffusers/pipelines/flux2/*`, `src/diffusers/models/transformers/transformer_flux2.py`)가 한다.
**텍스트 인코더(Qwen3)는 서버 프로세스에 아예 로드되지 않는다** — `text_encoder=None, tokenizer=None`으로
파이프라인을 만들고, 프롬프트는 전부 빌드 타임에 미리 인코딩해 둔 텐서를 쓴다 (6장).

## 2. 파일 맵

| 경로 | 역할 |
| --- | --- |
| `my/design_tune_server.py` | Flask 서버 전체: 라우팅, `InferenceEngine`(451), `BuildManager`(308), `StyleStore`(161) |
| `my/smile_design/prompt_templates.py` | YAML 파싱(`load_experiment_config`), 슬라이더 값→영문 지시문 치환(`render_prompt`), `reference_roles` |
| `my/smile_design/embedding_library.py` | `EmbeddingLibrary` — 미리 인코딩된 `prompt_embeds`를 디스크에서 로드/보간 (텍스트 인코더 없음) |
| `my/smile_design/build_embedding_library.py` | **별도 프로세스**. `Flux2KleinPipeline`을 텍스트 인코더용으로 로드해 YAML의 모든 프롬프트 조합을 실제로 인코딩, `.pt` 파일 + `manifest.json`으로 저장 |
| `my/smile_design/face_mask.py` | `FaceMasker` — MediaPipe Face Landmarker 기반 영역 마스크(`region_masks`, `union`, `build`) |
| `my/smile_design/mice.py` | `MiceAttention` — transformer에 forward pre-hook을 걸어 attention bias 주입 (5장) |
| `my/smile_design/inference.py` | `load_transformer_override` — 단일 safetensors 체크포인트(klein-4B) 로딩 분기 (7장) |
| `src/diffusers/pipelines/flux2/pipeline_flux2_klein.py` | `Flux2KleinPipeline` — 메인 edit 파이프라인 (diffusers 코어) |
| `src/diffusers/pipelines/flux2/pipeline_flux2_klein_inpaint.py` | `Flux2KleinInpaintPipeline` — latent 단위 inpaint 파이프라인 (diffusers 코어) |
| `src/diffusers/models/transformers/transformer_flux2.py` | `Flux2Transformer2DModel`(1059) — 실제 DiT. MICE 훅이 걸리는 대상 |

## 3. diffusers 코어와 my/ 코드의 경계

경계는 **두 곳**뿐이다.

1. **파이프라인 생성** — `InferenceEngine.load()`(465)/`load_inpaint()`(507)에서 `Flux2KleinPipeline`/
   `Flux2KleinInpaintPipeline` 인스턴스를 만든다. 이때 `transformer`/`vae` 모듈 객체를 두 파이프라인이
   **그대로 공유**한다 (`load_inpaint()`가 `load()`가 만든 파이프라인의 `.transformer`/`.vae`를 그대로 넘겨받음
   — 추가 VRAM 없음, fp8 캐스팅도 같은 객체라 자동 적용).
2. **파이프라인 호출** — `infer()`(526)의 `pipeline(**call_kwargs)` 한 줄, 그리고 `_inpaint_sequential()`(641)
   안의 같은 호출. **이 줄 안쪽은 100% diffusers 코어**(스케줄러 스텝, DiT forward, VAE decode)다.

이 두 지점을 빼면 `my/` 코드는 diffusers 소스를 전혀 수정하지 않는다. MICE만 예외(5장) — 코드를 수정하는 게
아니라 **런타임에 PyTorch 모듈 훅을 거는 방식**으로 개입한다.

## 4. API 파라미터 → 코드 연결

| API 파라미터 | 처리 함수 | 비고 |
| --- | --- | --- |
| `style_name`, `layout` | `StyleStore.read` → `reference_roles` | layout이 요구하는 이미지 수와 `images` 개수가 다르면 `infer()`(536) 초반에 바로 거부 |
| `mode`(`preset`/`anchor`/`grid`/`delta`), `preset`/`attribute`/`value`/`grid`/`control_values`/`delta_strengths` | `InferenceEngine._embedding()`(790) | `EmbeddingLibrary.preset/interpolate_anchor/interpolate_grid/compose_deltas` 중 하나를 호출해 **디스크의 캐시 텐서만** 조합. 텍스트 인코더 호출 없음 |
| `mask`(`enabled`/`mode`/`regions`/`dilate`/`feather`/`strength`/`edits`/...) | `resolve_mask_options()`(76) | 기본값은 `--mask-mode`(기본 `inpaint`). 여기서 만들어진 `dict`가 `infer()` 분기를 결정 |
| `mask.mode: "composite"` | `infer()`(526) 내 `self.masker.build()` 호출 후, 파이프라인 결과를 RGB로 원본에 합성 | 레거시 |
| `mask.mode: "mice"` | `InferenceEngine._mice()`(735) → `MiceAttention` 생성 → `infer()`에서 `mice.install(pipeline.transformer)`로 hook 등록 | 레거시, 5장 |
| `mask.mode: "inpaint"`, `edits` 없음 | `infer()`(526)의 `is_inpaint` 분기, `self.load_inpaint()` 사용, `call_kwargs`에 `mask_image`/`image_reference`/`strength` 전달 | 현재 기본값 |
| `mask.mode: "inpaint"`, `edits` 있음 | `InferenceEngine._inpaint_sequential()`(641) | 영역마다 `Flux2KleinInpaintPipeline` 호출을 체이닝 (MICE 대체) |
| `num_steps`/`steps`, `guidance_scale`, `seed` | `infer()`/`_inpaint_sequential()` 안에서 바로 `pipeline(...)`에 전달 | distilled 모델이 아니면서 step이 낮으면 `step_warning` 메타데이터에 경고 노출 |
| `images`(multipart 파일들) | `load_uploaded_images()`(865) → `call_kwargs["image"]`(또는 `image_reference`) | layout의 두 번째 이미지는 inpaint 모드에서 자동으로 `image_reference`가 됨 |

## 5. MICE가 diffusers 코어에 개입하는 정확한 지점

`MiceAttention.install(transformer)`(mice.py:75)가 `transformer.register_forward_pre_hook(self._pre_hook,
with_kwargs=True)`를 건다. 대상은 `Flux2Transformer2DModel.forward`(transformer_flux2.py:1225)이고, 이 hook이
가로채는 kwarg는 forward 시그니처 그대로다: `hidden_states`, `encoder_hidden_states`, `img_ids`,
`joint_attention_kwargs`. hook은 `joint_attention_kwargs["attention_mask"]`에 직접 계산한 bias 텐서를 끼워
넣고 `(args, kwargs)`를 돌려준다 — diffusers 쪽은 이 `attention_mask`를 내부 attention processor에
그대로 전달할 뿐, MICE의 존재를 모른다. 요청이 끝나면 `hook.remove()`로 제거되어 다음 요청(mice 아닌 요청
포함)에는 전혀 영향이 없다 (`infer()`의 `try/finally`, 526 근처).

디버깅 포인트: MICE 결과가 이상하면 `_bias()`(mice.py:111)가 만드는 bias 행렬의 영역 배정(`_grid`,
mice.py:96, `coverage` 임계값)과 `_image_groups()`(mice.py:155, `img_ids`에서 이미지별 경계를 역산)를 먼저
의심한다 — attention processor 자체는 건드리지 않는다.

## 6. 임베딩: 빌드 타임 vs 런타임이 완전히 분리되어 있다

- **빌드 타임**(`build_embedding_library.py`, 서버와 별도 프로세스): YAML의 `base_template`/`controls`/
  `presets`/`mice` 조합마다 `render_prompt()`(prompt_templates.py:69)로 영문 문장을 만들고,
  `Flux2KleinPipeline.encode_prompt(...)`(diffusers 코어 메서드)를 호출해 `[1, L, D]` 텐서로 인코딩,
  `<library>/<entry_id>.pt`로 저장한다. **텍스트 인코더(Qwen3, bf16 약 8GB)가 메모리에 올라가는 유일한 순간.**
- **런타임**(서버 프로세스): 텍스트 인코더를 아예 로드하지 않는다. `EmbeddingLibrary.load/preset/
  interpolate_anchor/interpolate_grid/compose_deltas`(embedding_library.py)는 전부 `torch.load()` +
  `torch.lerp`/`add_` 같은 텐서 연산일 뿐, 프롬프트 문자열을 다시 인코딩하지 않는다.

**디버깅 포인트**: "프롬프트를 바꿨는데 결과가 그대로다" → 거의 항상 **라이브러리 재빌드를 안 한 것**이다.
`library_path`(StyleStore, server.py:216 부근)는 YAML 내용의 SHA256이라, 텍스트 한 글자만 바꿔도 해시가
바뀌어 새 경로가 되고, 재빌드 전까지는 `manifest.json`이 없어 `/api/v1/infer`가 409를 반환한다(`build_endpoint`
필드로 빌드 요청 경로를 알려줌). "YAML은 바뀌었는데 예전 임베딩이 쓰인다"는 상황 자체가 구조적으로 불가능하다.

## 7. Weights 위치

| 가중치 | 위치 | 비고 |
| --- | --- | --- |
| FLUX.2 Klein 4B / Klein base-4B / Flux2-dev | `~/.cache/huggingface/hub/models--black-forest-labs--*` (17~23GB) | Apache 2.0. `--model` 인자로 경로/repo id 지정 |
| MediaPipe Face Landmarker | `weights/face_landmarker.task` | 현재 마스크 생성의 유일한 모델 (Apache 2.0) |
| BiSeNet(`79999_iter.pth`), `bisenet_celebA.pth`, YuNet(`face_detection_yunet_*.onnx`) | `weights/` | **레거시, 미사용** — BiSeNet은 라이선스 문제로 MediaPipe로 교체됨. `bisenet_celebA.pth`는 9바이트 깨진 파일(다운로드 실패 잔재) |

**모델 로딩 분기** (`load_transformer_override`, `my/smile_design/inference.py:202`): `--model` 경로 안에
`transformer/diffusion_pytorch_model.safetensors`(표준 diffusers 폴더 레이아웃)가 있으면 `None`을 반환해
`from_pretrained`의 기본 로딩을 쓴다(klein-base-4B가 이 케이스). 루트에 단일 `.safetensors` 파일만 있으면
(klein-4B가 이 케이스) `Flux2Transformer2DModel.from_single_file(...)`로 직접 로딩한다. **이 함수가 `None`을
반환할 때 `InferenceEngine.load()`(465)가 `transformer=None`을 그대로 `from_pretrained`에 넘기면 안 된다**
(과거 실제 버그였음, `final_report.md` 0.3절) — kwarg 자체를 생략해야 정상적으로 로드된다.

## 8. 디버깅 체크리스트 (증상 → 먼저 볼 곳)

| 증상 | 먼저 확인 |
| --- | --- |
| 컨트롤 값을 바꿨는데 결과가 그대로 | 라이브러리 재빌드했는지 (`library_ready`, `/api/v1/styles`) — 6장 |
| 409 "Embedding library is not built" | 응답의 `build_endpoint`로 빌드 요청, YAML 해시가 바뀐 것 — `StyleStore.read`(216) |
| 편집 영역 밖이 같이 바뀜 | `mask.mode`가 `inpaint`인지 확인(기본값), `composite`/`mice` 레거시 모드를 쓰고 있으면 전체 재생성 특성상 원래 그럴 수 있음 |
| 여러 영역 조합 시 번짐/간섭 | `mask.mode: mice`를 쓰고 있다면 `inpaint` + `edits`(`_inpaint_sequential`, 641)로 바꿔보기 — 영역이 완전히 독립적인 호출이라 구조적으로 번짐이 없음 |
| OOM | `--cpu-offload`, `--fp8-weights`(기본 on) 상태 확인, GPU 세대(Ampere는 저장만 fp8, 연산은 bf16 캐스팅) |
| base-4B 등 비증류 모델에서 결과가 뭉갬 | `step_warning` 메타데이터 확인 — `--min-steps-non-distilled`(기본 20) 미만이면 경고 |
| transformer가 로드 안 된 것 같음(`AttributeError`) | `--model` 경로가 diffusers 폴더 형식인지 단일 safetensors인지 확인 — 7장 |
