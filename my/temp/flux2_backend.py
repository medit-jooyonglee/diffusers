"""REST client for the cached-embedding diffusion tuning server.

The server is intentionally kept out of this desktop package. All endpoint paths
live here so the server contract does not leak into the Qt widgets.
"""

from __future__ import annotations

import base64
import io
import json
import os
from typing import Any, Optional
from urllib.parse import quote

import numpy as np
import requests
from PIL import Image


DEFAULT_SERVER_URL = os.environ.get("DIFFUSION_TUNE_SERVER_URL", "http://10.100.1.45:8603")
API_PREFIX = "/api/v1"


class DiffusionTuneApiError(RuntimeError):
    pass


def check_server(server_url: str = DEFAULT_SERVER_URL, timeout: float = 5.0) -> dict[str, Any]:
    response = requests.get(_url(server_url, "/health", api=False), timeout=timeout)
    _raise_for_status(response)
    try:
        return response.json()
    except ValueError:
        return {"status": response.text.strip() or "ok"}


def list_styles(server_url: str = DEFAULT_SERVER_URL, timeout: float = 10.0) -> list[dict[str, Any]]:
    response = requests.get(_url(server_url, "/styles"), timeout=timeout)
    _raise_for_status(response)
    payload = response.json()
    styles = payload.get("styles", payload) if isinstance(payload, dict) else payload
    if not isinstance(styles, list):
        raise DiffusionTuneApiError("서버의 style 목록 응답 형식이 올바르지 않습니다.")
    return [item if isinstance(item, dict) else {"name": str(item)} for item in styles]


def get_style(style_name: str, server_url: str = DEFAULT_SERVER_URL, timeout: float = 10.0) -> str:
    response = requests.get(_url(server_url, f"/styles/{quote(style_name, safe='')}"), timeout=timeout)
    _raise_for_status(response)
    content_type = response.headers.get("Content-Type", "")
    if "yaml" in content_type or "text/plain" in content_type:
        return response.text
    payload = response.json()
    if isinstance(payload, dict) and isinstance(payload.get("yaml"), str):
        return payload["yaml"]
    raise DiffusionTuneApiError("서버 style 응답에 yaml 문자열이 없습니다.")


def update_style(
    style_name: str,
    yaml_text: str,
    server_url: str = DEFAULT_SERVER_URL,
    timeout: float = 30.0,
) -> dict[str, Any]:
    response = requests.put(
        _url(server_url, f"/styles/{quote(style_name, safe='')}"),
        json={"yaml": yaml_text},
        timeout=timeout,
    )
    _raise_for_status(response)
    return _json_or_message(response)


def delete_style(style_name: str, server_url: str = DEFAULT_SERVER_URL, timeout: float = 30.0) -> dict[str, Any]:
    response = requests.delete(
        _url(server_url, f"/styles/{quote(style_name, safe='')}"), timeout=timeout
    )
    _raise_for_status(response)
    return _json_or_message(response)


def build_style_embeddings(
    style_name: str,
    layouts: Optional[list[str]] = None,
    kinds: Optional[list[str]] = None,
    server_url: str = DEFAULT_SERVER_URL,
    timeout: float = 30.0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if layouts:
        payload["layouts"] = layouts
    if kinds:
        payload["kinds"] = kinds
    response = requests.post(
        _url(server_url, f"/styles/{quote(style_name, safe='')}/build"),
        json=payload,
        timeout=timeout,
    )
    _raise_for_status(response)
    return _json_or_message(response)


def get_job(job_id: str, server_url: str = DEFAULT_SERVER_URL, timeout: float = 10.0) -> dict[str, Any]:
    response = requests.get(_url(server_url, f"/jobs/{quote(job_id, safe='')}"), timeout=timeout)
    _raise_for_status(response)
    payload = response.json()
    if not isinstance(payload, dict):
        raise DiffusionTuneApiError("서버의 job 응답 형식이 올바르지 않습니다.")
    return payload


def run_inference(
    style_name: str,
    images: list[np.ndarray],
    layout: str,
    mode: str,
    control_values: dict[str, float],
    preset: Optional[str] = None,
    grid: Optional[str] = None,
    anchor_attribute: Optional[str] = None,
    num_steps: Optional[int] = None,
    guidance: Optional[float] = None,
    seed: int = 0,
    server_url: str = DEFAULT_SERVER_URL,
    timeout: float = 600.0,
) -> np.ndarray:
    if not style_name.strip():
        raise ValueError("style 이름이 필요합니다.")
    if not images:
        raise ValueError("추론할 입력 이미지를 선택하세요.")
    parameters: dict[str, Any] = {
        "style_name": style_name,
        "layout": layout,
        "mode": mode,
        "control_values": control_values,
        "seed": seed,
    }
    if preset:
        parameters["preset"] = preset
    if grid:
        parameters["grid"] = grid
    if anchor_attribute:
        parameters["attribute"] = anchor_attribute
    if num_steps is not None:
        parameters["num_steps"] = num_steps
    if guidance is not None:
        parameters["guidance_scale"] = guidance

    files = [
        ("images", (f"reference_{index + 1}.png", _encode_png(image), "image/png"))
        for index, image in enumerate(images)
    ]
    response = requests.post(
        _url(server_url, "/infer"),
        data={"parameters": json.dumps(parameters, ensure_ascii=False)},
        files=files,
        timeout=timeout,
    )
    _raise_for_status(response)
    result = _decode_image_response(response, server_url, timeout)
    return _resize_to_source(result, images[0])


def _decode_image_response(response: requests.Response, server_url: str, timeout: float) -> np.ndarray:
    if response.headers.get("Content-Type", "").lower().startswith("image/"):
        return np.array(Image.open(io.BytesIO(response.content)).convert("RGB"))
    payload = response.json()
    if not isinstance(payload, dict):
        raise DiffusionTuneApiError("추론 응답에 이미지가 없습니다.")
    encoded = payload.get("image_base64")
    if isinstance(encoded, str):
        if encoded.startswith("data:"):
            encoded = encoded.split(",", 1)[-1]
        return np.array(Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB"))
    image_url = payload.get("image_url")
    if isinstance(image_url, str):
        if image_url.startswith("/"):
            image_url = f"{server_url.rstrip('/')}{image_url}"
        image_response = requests.get(image_url, timeout=timeout)
        _raise_for_status(image_response)
        return np.array(Image.open(io.BytesIO(image_response.content)).convert("RGB"))
    raise DiffusionTuneApiError("추론 응답에 image binary, image_base64 또는 image_url이 없습니다.")


def _encode_png(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def _resize_to_source(result: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Guarantee that the client receives an image with the source image dimensions.

    The server should run the model at dimensions rounded to a multiple of 16 and
    resize back itself. This fallback also supports older servers that return the
    internal inference dimensions directly.
    """
    source_height, source_width = source.shape[:2]
    if result.shape[:2] == (source_height, source_width):
        return result
    resized = Image.fromarray(result).resize((source_width, source_height), Image.Resampling.LANCZOS)
    return np.array(resized.convert("RGB"))


def _url(server_url: str, path: str, api: bool = True) -> str:
    prefix = API_PREFIX if api else ""
    return f"{server_url.rstrip('/')}{prefix}{path}"


def _raise_for_status(response: requests.Response) -> None:
    if response.ok:
        return
    detail = response.text.strip()
    try:
        payload = response.json()
        detail = payload.get("detail") or payload.get("error") or detail
    except ValueError:
        pass
    raise DiffusionTuneApiError(f"서버 오류 {response.status_code}: {detail or response.reason}")


def _json_or_message(response: requests.Response) -> dict[str, Any]:
    if not response.content:
        return {"status": "ok"}
    try:
        payload = response.json()
    except ValueError:
        return {"status": "ok", "message": response.text.strip()}
    return payload if isinstance(payload, dict) else {"status": "ok", "data": payload}
