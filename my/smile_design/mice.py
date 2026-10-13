"""MICE (Multi-Instance Concurrent Editing) attention bias for FLUX.2 Klein.

논문 "Editing Everything Everywhere All at Once" (arXiv 2606.31278) 설명을 바탕으로 직접 구현했다.
영역별 지시문 토큰이 자기 마스크 영역의 이미지 토큰과만 상호작용하도록 joint attention logits에 bias를 더한다
(Softmax(QK^T / sqrt(d) + B) V). 마스크 밖 영역은 전역 지시문과 원본 이미지만 참조하므로 그대로 유지된다.

토큰 순서: [전역 텍스트, 영역 1 텍스트, ..., 영역 N 텍스트, 생성 latent, 환자 참조 latent, 기타 참조 latent]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


BLOCKED = -1.0e4  # bf16에서 softmax 후 0이 되는 값 (-inf는 행 전체가 막히면 NaN이 됨)


@dataclass
class MiceEdit:
    mask: np.ndarray  # 원본 이미지 해상도의 uint8 마스크 (0 / 255)
    prompt_embeds: torch.Tensor  # [1, T_i, D], padding을 잘라낸 영역 지시문 임베딩


class PromptLengthCounter:
    """빌드할 때와 같은 chat template으로 프롬프트의 실제 토큰 수를 센다 (padding 제거용)."""

    def __init__(self, text_encoder_model: str | Path):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(Path(text_encoder_model) / "tokenizer")

    def __call__(self, prompt: str, max_sequence_length: int) -> int:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return min(len(self.tokenizer(text).input_ids), max_sequence_length)


class MiceAttention:
    def __init__(
        self,
        global_embeds: torch.Tensor,
        edits: list[MiceEdit],
        kernel_size: int = 11,
        temperature: float = 3.0,
        steps: int | None = None,
        coverage: float = 0.25,
        background_context: bool = True,
    ):
        if not edits:
            raise ValueError("MICE requires at least one edit")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.edits = edits
        self.kernel_size = kernel_size
        self.temperature = temperature
        self.steps = steps
        # coverage: 토큰을 영역에 배정하는 최소 마스크 비율. 0.5면 작은 영역(아랫니 끝 등)의 경계 토큰이 배경으로 빠져 뭉개짐
        # background_context: 배경 토큰도 영역 토큰을 약한 패널티(-1)로 참조 허용 (경계 이음새 완화)
        self.coverage = coverage
        self.background_context = background_context
        self.prompt_embeds = torch.cat([global_embeds] + [edit.prompt_embeds for edit in edits], dim=1)
        bounds = np.cumsum([0, global_embeds.shape[1]] + [edit.prompt_embeds.shape[1] for edit in edits])
        self.text_segments = list(zip(bounds[:-1], bounds[1:]))  # [0] = 전역, [1:] = 영역별
        self.calls = 0
        self.cache: tuple[tuple, torch.Tensor] | None = None

    def install(self, transformer: torch.nn.Module):
        self.calls = 0
        return transformer.register_forward_pre_hook(self._pre_hook, with_kwargs=True)

    def _pre_hook(self, module, args, kwargs):
        self.calls += 1
        if self.steps is not None and self.calls > self.steps:
            return None
        text_length = kwargs["encoder_hidden_states"].shape[1]
        if text_length != self.prompt_embeds.shape[1]:
            raise RuntimeError(f"MICE text length mismatch: {text_length} vs {self.prompt_embeds.shape[1]}")
        groups = _image_groups(kwargs["img_ids"][0])
        key = (text_length, tuple(groups))
        if self.cache is None or self.cache[0] != key:
            hidden = kwargs["hidden_states"]
            self.cache = (key, self._bias(text_length, groups, hidden.device, hidden.dtype))
        joint_attention_kwargs = dict(kwargs.get("joint_attention_kwargs") or {})
        joint_attention_kwargs["attention_mask"] = self.cache[1]
        kwargs["joint_attention_kwargs"] = joint_attention_kwargs
        return args, kwargs

    def _grid(self, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
        """토큰 격자에서 영역 소속(-1 = 배경)과 영역별 Gaussian log 패널티 [N, H*W]를 만든다."""
        coverage = [cv2.resize(e.mask.astype(np.float32) / 255.0, (width, height), interpolation=cv2.INTER_AREA) for e in self.edits]
        owner = np.full(height * width, -1, dtype=np.int64)
        for index, c in enumerate(coverage):
            owner[(c.reshape(-1) > self.coverage) & (owner < 0)] = index
        weights = []
        for index, c in enumerate(coverage):
            s = cv2.GaussianBlur((owner == index).reshape(height, width).astype(np.float32), (self.kernel_size, self.kernel_size), 0)
            w = np.maximum(np.log(s.reshape(-1) + 1e-6) / self.temperature, BLOCKED)
            w[(owner >= 0) & (owner != index)] = BLOCKED  # 다른 영역은 차단
            w[owner == index] = 0.0
            weights.append(w)
        return owner, np.stack(weights)

    def _bias(self, text_length: int, groups: list[tuple[int, int, int, int]], device, dtype) -> torch.Tensor:
        # 행 = query, 열 = key. 기본은 차단, 아래 규칙만 연다.
        # - 전역 지시문 <-> 배경 이미지 토큰(생성 latent / 환자 참조의 마스크 밖)
        # - 영역 n 지시문 <-> 영역 n 이미지 토큰
        # - 영역 n 이미지 토큰 -> 이미지 토큰: 영역 n 안 0, 멀어질수록 Gaussian 감쇠, 다른 영역 차단 (전역 지시문은 차단)
        # - 기타 참조 이미지(치아 템플릿 등)는 모든 토큰이 참조 가능
        total = text_length + sum(h * w for _, _, h, w in groups)
        bias = torch.full((total, total), BLOCKED, device=device, dtype=torch.float32)
        owners, weights, start = [], [], text_length
        bound = []  # 환자 사진과 같은 좌표계로 묶이는 이미지(생성 latent + 첫 번째 참조)
        free = []
        for order, (time, _, height, width) in enumerate(groups):
            index = torch.arange(start, start + height * width, device=device)
            if time == 0 or order == 1:
                owner, weight = self._grid(height, width)
                bound.append((index, torch.from_numpy(owner).to(device), torch.from_numpy(weight).to(device)))
            else:
                free.append(index)
            start += height * width
        g = torch.arange(*self.text_segments[0], device=device)
        bias[g[:, None], g] = 0.0
        for index, owner, _ in bound:
            background = index[owner < 0]
            bias[g[:, None], background] = 0.0
            bias[background[:, None], g] = 0.0
            for other_index, other_owner, other_weight in bound:
                bias[background[:, None], other_index[other_owner < 0]] = 0.0
                if self.background_context:
                    bias[background[:, None], other_index[other_owner >= 0]] = -1.0
        for n, (t0, t1) in enumerate(self.text_segments[1:]):
            text = torch.arange(t0, t1, device=device)
            bias[text[:, None], text] = 0.0
            for index, owner, weight in bound:
                inside = index[owner == n]
                bias[text[:, None], inside] = 0.0
                bias[inside[:, None], text] = 0.0
                for other_index, _, other_weight in bound:
                    bias[inside[:, None], other_index] = other_weight[n][None, :].float()
        for index in free:
            bias[:, index] = 0.0
            bias[index[:, None], index] = 0.0
        return bias.to(dtype)[None, None]


def _image_groups(img_ids: torch.Tensor) -> list[tuple[int, int, int, int]]:
    """img_ids (N, 4) = (t, h, w, l)에서 이미지별 (t, 시작 인덱스, 높이, 너비)를 순서대로 구한다."""
    times = img_ids[:, 0].tolist()
    groups, start = [], 0
    while start < len(times):
        end = start
        while end < len(times) and times[end] == times[start]:
            end += 1
        ids = img_ids[start:end]
        groups.append((int(times[start]), start, int(ids[:, 1].max()) + 1, int(ids[:, 2].max()) + 1))
        start = end
    return groups
