from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageFilter


# MediaPipe Face Mesh 478점 인덱스
UPPER_LIP_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
UPPER_LIP_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LOWER_LIP_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# mouth = lips + inner_mouth, teeth = inner_mouth 중 밝은 픽셀 (Otsu)
MASK_REGIONS = ("upper_lip", "lower_lip", "lips", "inner_mouth", "teeth", "mouth", "face")


def parse_regions(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",")]
    if not isinstance(value, list) or not value:
        raise ValueError(f"mask regions must be a non-empty list or comma string from {MASK_REGIONS}")
    regions = [str(region) for region in value if str(region)]
    unknown = sorted(set(regions) - set(MASK_REGIONS))
    if unknown:
        raise ValueError(f"Unknown mask regions {unknown}; choose from {MASK_REGIONS}")
    return regions


def _polygon(mask: np.ndarray, points: np.ndarray, indices: list[int]) -> np.ndarray:
    cv2.fillPoly(mask, [points[indices].reshape(-1, 1, 2)], 255)
    return mask


def _teeth(rgb: np.ndarray, inner_mouth: np.ndarray, min_pixels: int = 20) -> np.ndarray:
    # 입 안에서 밝고 붉지 않은 픽셀을 치아로 분리 (혀/구강은 어둡거나 붉음)
    teeth = np.zeros_like(inner_mouth)
    inside = inner_mouth > 0
    if inside.sum() < min_pixels:
        return teeth
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness, red = lab[..., 0][inside], lab[..., 1][inside]
    threshold, _ = cv2.threshold(lightness.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.nonzero(inside)
    selected = (lightness > threshold) & (red < 150)
    teeth[ys[selected], xs[selected]] = 255
    return teeth


class FaceMasker:
    """MediaPipe Face Landmarker 랜드마크를 다각형으로 채워 얼굴 영역 마스크를 만든다."""

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.landmarker = None
        self.lock = threading.Lock()

    def _load(self):
        if self.landmarker is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"Face landmarker model not found: {self.model_path}")
            from mediapipe.tasks.python import BaseOptions, vision

            self.landmarker = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(self.model_path)), num_faces=1
                )
            )
        return self.landmarker

    def close(self) -> None:
        with self.lock:
            if self.landmarker is not None:
                self.landmarker.close()
                self.landmarker = None

    def landmarks(self, image: Image.Image) -> np.ndarray | None:
        import mediapipe as mp

        rgb = np.ascontiguousarray(np.asarray(image.convert("RGB")))
        with self.lock:
            result = self._load().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.face_landmarks:
            return None
        height, width = rgb.shape[:2]
        return np.array([[point.x * width, point.y * height] for point in result.face_landmarks[0]], np.int32)

    def build(
        self, image: Image.Image, regions: list[str], dilate: int = 0, feather: float = 0.0
    ) -> tuple[Image.Image, dict[str, Any]]:
        if dilate < 0 or feather < 0:
            raise ValueError("mask dilate and feather must be non-negative")
        points = self.landmarks(image)
        if points is None:
            raise ValueError("Mask requested but no face was detected in the patient image")

        rgb = np.asarray(image.convert("RGB"))
        empty = np.zeros(rgb.shape[:2], np.uint8)
        upper_lip = _polygon(empty.copy(), points, UPPER_LIP_OUTER + UPPER_LIP_INNER[::-1])
        lower_lip = _polygon(empty.copy(), points, LOWER_LIP_OUTER + LOWER_LIP_INNER[::-1])
        inner_mouth = _polygon(empty.copy(), points, UPPER_LIP_INNER + LOWER_LIP_INNER[::-1])
        masks = {
            "upper_lip": upper_lip,
            "lower_lip": lower_lip,
            "lips": upper_lip | lower_lip,
            "inner_mouth": inner_mouth,
            "teeth": _teeth(rgb, inner_mouth),
            "mouth": upper_lip | lower_lip | inner_mouth,
            "face": _polygon(empty.copy(), points, FACE_OVAL),
        }

        mask = empty
        for region in regions:
            mask = mask | masks[region]
        if dilate:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
            mask = cv2.dilate(mask, kernel)
        mask_image = Image.fromarray(mask, mode="L")
        if feather:
            mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=feather))

        face_x = points[FACE_OVAL, 0]
        metadata = {
            "enabled": True,
            "regions": regions,
            "dilate": dilate,
            "feather": feather,
            "face_width": int(face_x.max() - face_x.min()),
            "mask_pixels": int((mask > 0).sum()),
            "region_pixels": {region: int((masks[region] > 0).sum()) for region in regions},
        }
        return mask_image, metadata
