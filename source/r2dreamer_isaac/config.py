from __future__ import annotations

import copy
import pathlib
import re
from typing import Any

import yaml


_INTERPOLATION_RE = re.compile(r"^\$\{([^}]+)\}$")
_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\d*\.\d+|\d+[eE][+-]?\d+|\d+\.\d*[eE][+-]?\d+|\d*\.\d+[eE][+-]?\d+)$")


class Config(dict):
    """Dictionary with attribute-style access for nested config trees."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

    def clone(self) -> "Config":
        return to_config(to_plain_dict(self))


def to_config(value: Any) -> Any:
    if isinstance(value, Config):
        return value
    if isinstance(value, dict):
        return Config({key: to_config(val) for key, val in value.items()})
    if isinstance(value, list):
        return [to_config(val) for val in value]
    return value


def to_plain_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_plain_dict(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_plain_dict(val) for val in value]
    return value


def load_yaml(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping in {path}, got: {type(data)!r}")
    return data


def deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def get_by_dotted_path(data: dict[str, Any], dotted_path: str) -> Any:
    node: Any = data
    for part in dotted_path.split("."):
        if not isinstance(node, dict):
            raise KeyError(f"Cannot resolve '{dotted_path}': '{part}' is not inside a mapping.")
        node = node[part]
    return node


def set_by_dotted_path(data: dict[str, Any], dotted_path: str, value: Any) -> None:
    node = data
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        current = node.get(part)
        if not isinstance(current, dict):
            current = {}
            node[part] = current
        node = current
    node[parts[-1]] = value


def _resolve_node(node: Any, root: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        return {key: _resolve_node(val, root) for key, val in node.items()}
    if isinstance(node, list):
        return [_resolve_node(val, root) for val in node]
    if isinstance(node, str):
        match = _INTERPOLATION_RE.fullmatch(node)
        if match is not None:
            return copy.deepcopy(get_by_dotted_path(root, match.group(1)))
    return node


def resolve_interpolations(data: dict[str, Any], max_passes: int = 8) -> dict[str, Any]:
    resolved = copy.deepcopy(data)
    for _ in range(max_passes):
        next_resolved = _resolve_node(resolved, resolved)
        if next_resolved == resolved:
            return next_resolved
        resolved = next_resolved
    return resolved


def normalize_scalars(node: Any) -> Any:
    if isinstance(node, dict):
        return {key: normalize_scalars(value) for key, value in node.items()}
    if isinstance(node, list):
        return [normalize_scalars(value) for value in node]
    if isinstance(node, str):
        if _INT_RE.fullmatch(node):
            return int(node)
        if _FLOAT_RE.fullmatch(node):
            return float(node)
    return node


def parse_dotlist(overrides: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected the form key=value.")
        key, raw_value = override.split("=", 1)
        value = yaml.safe_load(raw_value)
        set_by_dotted_path(parsed, key, value)
    return parsed


def vendor_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent / "vendor" / "r2dreamer"


def load_model_config(
    preset_name: str,
    env_cfg: dict[str, Any],
    device: str,
    model_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_dir = vendor_root() / "configs" / "model"
    base_cfg = load_yaml(model_dir / "_base_.yaml")
    preset_cfg = load_yaml(model_dir / f"{preset_name}.yaml")
    preset_cfg.pop("defaults", None)

    merged_model = deep_merge(base_cfg, preset_cfg)
    if model_overrides:
        merged_model = deep_merge(merged_model, model_overrides)

    root = {"device": device, "env": copy.deepcopy(env_cfg), "model": merged_model}
    resolved = normalize_scalars(resolve_interpolations(root))
    return resolved["model"]


def build_runtime_config(
    task_cfg: dict[str, Any],
    cli_updates: dict[str, Any],
    dotlist_overrides: list[str],
) -> Config:
    merged = deep_merge(task_cfg, cli_updates)
    if dotlist_overrides:
        merged = deep_merge(merged, parse_dotlist(dotlist_overrides))

    if "env" not in merged:
        raise KeyError("Task R2-Dreamer config must define an 'env' section.")
    if "model_preset" not in merged:
        merged["model_preset"] = "size12M"
    if "device" not in merged:
        merged["device"] = "cuda:0"

    merged.setdefault("agent_device", merged["device"])
    merged.setdefault("env_device", merged["device"])
    # Keep the legacy top-level "device" as the learner device for backward compatibility.
    merged["device"] = merged["agent_device"]

    merged["model"] = load_model_config(
        preset_name=str(merged["model_preset"]),
        env_cfg=merged["env"],
        device=str(merged["agent_device"]),
        model_overrides=merged.get("model"),
    )
    merged["buffer"]["device"] = str(merged["agent_device"])
    merged = normalize_scalars(resolve_interpolations(merged))
    return to_config(merged)
