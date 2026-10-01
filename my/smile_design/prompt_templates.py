from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PromptEntry:
    entry_id: str
    kind: str
    layout: str
    prompt: str
    values: dict[str, float]
    preset: str | None = None
    grid: str | None = None


def load_experiment_config(path: str | Path) -> tuple[dict[str, Any], str]:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Smile Design config not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("version") != 1:
        raise ValueError(f"Smile Design config must be a version 1 mapping: {config_path}")
    required = {"template_version", "base_template", "layouts", "controls", "presets", "grids"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Smile Design config is missing keys: {sorted(missing)}")
    if not isinstance(config["layouts"], dict) or not config["layouts"]:
        raise ValueError("layouts must be a non-empty mapping")
    if not isinstance(config["controls"], dict) or not config["controls"]:
        raise ValueError("controls must be a non-empty mapping")
    for name, control in config["controls"].items():
        if not isinstance(control, dict) or "default" not in control or "anchors" not in control:
            raise ValueError(f"Control {name!r} requires default and anchors")
        anchors = {_as_float(value): text for value, text in control["anchors"].items()}
        if len(anchors) < 2:
            raise ValueError(f"Control {name!r} requires at least two anchors")
        if float(control["default"]) not in anchors:
            raise ValueError(f"Control {name!r} default must be one of its anchors")
        control["default"] = float(control["default"])
        control["anchors"] = dict(sorted(anchors.items()))
    for name, preset in config["presets"].items():
        if not isinstance(preset, dict):
            raise ValueError(f"Preset {name!r} must be a mapping")
        normalized = _validate_values(config, preset)
        for attribute, value in normalized.items():
            if value not in config["controls"][attribute]["anchors"]:
                raise ValueError(f"Preset {name!r} value for {attribute!r} must be an anchor")
    for name, grid in config["grids"].items():
        attributes = grid.get("attributes") if isinstance(grid, dict) else None
        if not isinstance(attributes, list) or len(attributes) != 2 or len(set(attributes)) != 2:
            raise ValueError(f"Grid {name!r} must contain exactly two unique attributes")
        unknown = set(attributes) - set(config["controls"])
        if unknown:
            raise ValueError(f"Grid {name!r} references unknown controls: {sorted(unknown)}")
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return config, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def render_prompt(config: dict[str, Any], layout: str, values: dict[str, float]) -> str:
    if layout not in config["layouts"]:
        raise ValueError(f"Unknown reference layout: {layout}")
    normalized = default_values(config)
    normalized.update(_validate_values(config, values))
    descriptions = {
        name: _anchor_text(config["controls"][name]["anchors"], value)
        for name, value in normalized.items()
    }
    return config["base_template"].format(
        reference_instructions=config["layouts"][layout]["instruction"].strip(),
        **descriptions,
    ).strip()


def default_values(config: dict[str, Any]) -> dict[str, float]:
    return {name: float(control["default"]) for name, control in config["controls"].items()}


def build_prompt_entries(
    config: dict[str, Any],
    layouts: list[str] | None = None,
    kinds: set[str] | None = None,
) -> list[PromptEntry]:
    selected_layouts = layouts or list(config["layouts"])
    unknown_layouts = set(selected_layouts) - set(config["layouts"])
    if unknown_layouts:
        raise ValueError(f"Unknown reference layouts: {sorted(unknown_layouts)}")
    selected_kinds = kinds or {"base", "preset", "anchor", "grid"}
    unknown_kinds = selected_kinds - {"base", "preset", "anchor", "grid"}
    if unknown_kinds:
        raise ValueError(f"Unknown entry kinds: {sorted(unknown_kinds)}")
    entries: list[PromptEntry] = []
    defaults = default_values(config)
    for layout in selected_layouts:
        if "base" in selected_kinds:
            entries.append(_entry(config, layout, "base", f"{layout}/base", defaults))
        if "preset" in selected_kinds:
            for preset_name, overrides in config["presets"].items():
                values = defaults | _validate_values(config, overrides)
                entries.append(
                    _entry(
                        config,
                        layout,
                        "preset",
                        f"{layout}/preset/{preset_name}",
                        values,
                        preset=preset_name,
                    )
                )
        if "anchor" in selected_kinds:
            for attribute, control in config["controls"].items():
                for value in control["anchors"]:
                    values = defaults | {attribute: float(value)}
                    entry_id = f"{layout}/anchor/{attribute}/{format_slider_value(value)}"
                    entries.append(_entry(config, layout, "anchor", entry_id, values))
        if "grid" in selected_kinds:
            for grid_name, grid in config["grids"].items():
                first, second = grid["attributes"]
                for first_value, second_value in itertools.product(
                    config["controls"][first]["anchors"], config["controls"][second]["anchors"]
                ):
                    values = defaults | {first: float(first_value), second: float(second_value)}
                    suffix = f"{format_slider_value(first_value)}_{format_slider_value(second_value)}"
                    entry_id = f"{layout}/grid/{grid_name}/{suffix}"
                    entries.append(_entry(config, layout, "grid", entry_id, values, grid=grid_name))
    ids = [entry.entry_id for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Generated prompt entry IDs are not unique")
    return entries


def format_slider_value(value: float) -> str:
    value = float(value)
    if value.is_integer():
        return f"{int(value):03d}"
    return f"{value:07.3f}".replace(".", "p")


def reference_roles(config: dict[str, Any], layout: str) -> list[str]:
    if layout not in config["layouts"]:
        raise ValueError(f"Unknown reference layout: {layout}")
    roles = config["layouts"][layout].get("roles")
    if not isinstance(roles, list) or not roles or roles[0] != "patient":
        raise ValueError(f"Layout {layout!r} must define roles beginning with patient")
    return [str(role) for role in roles]


def _entry(
    config: dict[str, Any],
    layout: str,
    kind: str,
    entry_id: str,
    values: dict[str, float],
    preset: str | None = None,
    grid: str | None = None,
) -> PromptEntry:
    return PromptEntry(
        entry_id=entry_id,
        kind=kind,
        layout=layout,
        prompt=render_prompt(config, layout, values),
        values={name: float(value) for name, value in values.items()},
        preset=preset,
        grid=grid,
    )


def _validate_values(config: dict[str, Any], values: dict[str, Any]) -> dict[str, float]:
    unknown = set(values) - set(config["controls"])
    if unknown:
        raise ValueError(f"Unknown controls: {sorted(unknown)}")
    normalized = {name: float(value) for name, value in values.items()}
    for name, value in normalized.items():
        anchors = config["controls"][name]["anchors"]
        low, high = min(anchors), max(anchors)
        if not low <= value <= high:
            raise ValueError(f"Control {name!r} must be between {low} and {high}, got {value}")
    return normalized


def _anchor_text(anchors: dict[float, str], value: float) -> str:
    if value not in anchors:
        raise ValueError(f"Prompt generation requires an exact anchor value, got {value}")
    return str(anchors[value]).strip()


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Anchor value must be numeric, got {value!r}") from error
