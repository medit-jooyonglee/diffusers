from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from .prompt_templates import format_slider_value


class EmbeddingLibrary:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Embedding library manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 1:
            raise ValueError(f"Unsupported embedding library schema: {self.manifest.get('schema_version')!r}")
        entries = self.manifest.get("entries")
        if not isinstance(entries, dict) or not entries:
            raise ValueError(f"Embedding library contains no entries: {manifest_path}")
        self.entries: dict[str, dict[str, Any]] = entries
        self.controls: dict[str, dict[str, Any]] = self.manifest["controls"]

    def load(
        self,
        entry_id: str,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        record = self._record(entry_id)
        path = self.root / record["file"]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        tensor = payload["prompt_embeds"] if isinstance(payload, dict) else payload
        self._validate_tensor(tensor, entry_id)
        return tensor.to(device=device, dtype=dtype or tensor.dtype)

    def load_negative(
        self, device: str | torch.device = "cpu", dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        relative_path = self.manifest.get("negative_embedding")
        if not relative_path:
            raise ValueError("Embedding library does not contain a negative embedding")
        payload = torch.load(self.root / relative_path, map_location="cpu", weights_only=True)
        tensor = payload["prompt_embeds"] if isinstance(payload, dict) else payload
        self._validate_tensor(tensor, "negative")
        return tensor.to(device=device, dtype=dtype or tensor.dtype)

    def preset(
        self,
        layout: str,
        name: str,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return self.load(f"{layout}/preset/{name}", device=device, dtype=dtype)

    def base(
        self,
        layout: str,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return self.load(f"{layout}/base", device=device, dtype=dtype)

    def trimmed(
        self,
        entry_id: str,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """패딩을 뺀 실제 문장 토큰만 돌려준다 (manifest의 tokens 사용)."""
        tensor = self.load(entry_id, device=device, dtype=dtype)
        tokens = self._record(entry_id).get("tokens")
        return tensor if not tokens else tensor[:, : min(int(tokens), tensor.shape[1])]

    def compose_concat(
        self,
        layout: str,
        values: dict[str, float],
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """공통 서문 + 요청한 control의 지시문을 토큰 축으로 이어 붙인다.

        delta(덧셈)처럼 문장이 밀려 어긋나는 일이 없고, 조합을 미리 등록하지 않아도 된다.
        순서는 manifest의 control_order(스타일 템플릿에 나오는 순서)를 따른다.
        """
        unknown = set(values) - set(self.controls)
        if unknown:
            raise ValueError(f"Unknown concat controls: {sorted(unknown)}")
        order = self.manifest.get("control_order") or list(self.controls)
        parts = [self.trimmed(f"{layout}/clause/identity", device=device, dtype=dtype)]
        for attribute in order:
            if attribute not in values:
                continue
            value = float(values[attribute])
            if value == float(self.controls[attribute]["default"]):
                continue
            entry_id = f"{layout}/clause/{attribute}/{format_slider_value(value)}"
            parts.append(self.trimmed(entry_id, device=device, dtype=dtype))
        return torch.cat(parts, dim=1)

    def interpolate_anchor(
        self,
        layout: str,
        attribute: str,
        value: float,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        low, high, weight = self.anchor_interval(attribute, value)
        low_tensor = self.load(
            f"{layout}/anchor/{attribute}/{format_slider_value(low)}", device=device, dtype=dtype
        )
        if low == high:
            return low_tensor
        high_tensor = self.load(
            f"{layout}/anchor/{attribute}/{format_slider_value(high)}", device=device, dtype=dtype
        )
        self._require_same_shape(low_tensor, high_tensor, f"anchor interpolation for {attribute}")
        return torch.lerp(low_tensor, high_tensor, weight)

    def interpolate_grid(
        self,
        layout: str,
        grid: str,
        values: dict[str, float],
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        grid_config = self.manifest["grids"].get(grid)
        if grid_config is None:
            raise ValueError(f"Unknown interpolation grid: {grid}")
        first, second = grid_config["attributes"]
        if set(values) != {first, second}:
            raise ValueError(f"Grid {grid!r} requires values for {first!r} and {second!r}")
        first_low, first_high, first_weight = self.anchor_interval(first, values[first])
        second_low, second_high, second_weight = self.anchor_interval(second, values[second])

        def corner(first_value: float, second_value: float) -> torch.Tensor:
            suffix = f"{format_slider_value(first_value)}_{format_slider_value(second_value)}"
            return self.load(f"{layout}/grid/{grid}/{suffix}", device=device, dtype=dtype)

        low_low = corner(first_low, second_low)
        high_low = corner(first_high, second_low) if first_high != first_low else low_low
        low_high = corner(first_low, second_high) if second_high != second_low else low_low
        if first_high == first_low and second_high == second_low:
            return low_low
        high_high = (
            corner(first_high, second_high)
            if first_high != first_low and second_high != second_low
            else high_low if second_high == second_low else low_high
        )
        for tensor in (high_low, low_high, high_high):
            self._require_same_shape(low_low, tensor, f"grid interpolation for {grid}")
        bottom = torch.lerp(low_low, high_low, first_weight)
        top = torch.lerp(low_high, high_high, first_weight)
        return torch.lerp(bottom, top, second_weight)

    def compose_deltas(
        self,
        layout: str,
        values: dict[str, float],
        strengths: dict[str, float] | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if not values:
            return self.base(layout, device=device, dtype=dtype)
        unknown = set(values) - set(self.controls)
        if unknown:
            raise ValueError(f"Unknown delta controls: {sorted(unknown)}")
        strengths = strengths or {}
        unknown_strengths = set(strengths) - set(values)
        if unknown_strengths:
            raise ValueError(f"Delta strengths have no matching values: {sorted(unknown_strengths)}")
        base = self.base(layout, device=device, dtype=dtype)
        result = base.clone()
        for attribute, value in values.items():
            attribute_embedding = self.interpolate_anchor(
                layout, attribute, value, device=device, dtype=dtype
            )
            self._require_same_shape(base, attribute_embedding, f"delta composition for {attribute}")
            result.add_(attribute_embedding - base, alpha=float(strengths.get(attribute, 1.0)))
        return result

    def anchor_interval(self, attribute: str, value: float) -> tuple[float, float, float]:
        if attribute not in self.controls:
            raise ValueError(f"Unknown control: {attribute}")
        anchors = sorted(float(anchor) for anchor in self.controls[attribute]["anchors"])
        value = float(value)
        if not anchors[0] <= value <= anchors[-1]:
            raise ValueError(f"{attribute} must be between {anchors[0]} and {anchors[-1]}, got {value}")
        for anchor in anchors:
            if math.isclose(value, anchor, rel_tol=0.0, abs_tol=1e-8):
                return anchor, anchor, 0.0
        for low, high in zip(anchors, anchors[1:]):
            if low < value < high:
                return low, high, (value - low) / (high - low)
        raise RuntimeError(f"Could not locate anchor interval for {attribute}={value}")

    def entry_metadata(self, entry_id: str) -> dict[str, Any]:
        return dict(self._record(entry_id))

    def prompt_for(self, entry_id: str) -> str:
        return str(self._record(entry_id)["prompt"])

    def _record(self, entry_id: str) -> dict[str, Any]:
        try:
            return self.entries[entry_id]
        except KeyError as error:
            raise ValueError(f"Embedding entry not found: {entry_id}") from error

    def _validate_tensor(self, tensor: Any, entry_id: str) -> None:
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3 or tensor.shape[0] != 1:
            shape = tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else type(tensor).__name__
            raise ValueError(f"Embedding {entry_id!r} must have shape (1,L,D), got {shape}")
        expected = self.manifest.get("embedding_shape")
        # clause 엔트리는 잘라서 이어 붙이는 용도라 길이가 고정이 아니다
        if expected is not None and "/clause/" not in entry_id and list(tensor.shape) != expected:
            raise ValueError(f"Embedding {entry_id!r} shape {list(tensor.shape)} does not match {expected}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Embedding {entry_id!r} contains non-finite values")

    @staticmethod
    def _require_same_shape(first: torch.Tensor, second: torch.Tensor, operation: str) -> None:
        if first.shape != second.shape:
            raise ValueError(f"Shape mismatch during {operation}: {tuple(first.shape)} vs {tuple(second.shape)}")
