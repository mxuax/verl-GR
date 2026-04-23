"""Shared beam-search settings and compatibility helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

BEAM_WIDTH_KEY = "beam_width"
BEAM_INDEX_KEY = "beam_index"
BEAM_GROUP_ID_KEY = "beam_group_id"
BEAM_RETURN_MODE_KEY = "beam_return_mode"
BEAM_SEARCH_PARAMS_KEY = "beam_search_params"
DECODE_CONFIG_KEY = "decode_config"

LEGACY_BEAM_WIDTH_KEYS = ("stage2_beam_size",)
LEGACY_BEAM_INDEX_KEYS = ("beam_idx",)
LEGACY_BEAM_GROUP_ID_KEYS = ("two_stage_group_id",)
LEGACY_BEAM_MAX_TOKENS_KEYS = ("stage2_num_tokens", "stage2_max_tokens")
LEGACY_REASONING_MAX_TOKENS_KEYS = ("stage1_max_tokens",)

DEFAULT_ITEM_PREFIX_TEXT = "\n<|sid_begin|>"
DEFAULT_REASONING_STOP = ["</think>"]


@dataclass(slots=True)
class DecodePhaseConfig:
    max_tokens: int
    stop: list[str] | None = None
    include_stop_str_in_output: bool = False
    prefix_text: str = ""


@dataclass(slots=True)
class BeamSearchConfig:
    width: int
    index: int
    group_id: str
    return_mode: str
    max_tokens: int
    length_penalty: float = 1.0
    ignore_eos: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1


@dataclass(slots=True)
class TwoStageDecodeConfig:
    reasoning: DecodePhaseConfig
    item_generation: DecodePhaseConfig


def get_rollout_custom_value(config, key: str, default: Any) -> Any:
    custom = getattr(config, "custom", None)
    if isinstance(custom, dict):
        return custom.get(key, default)
    if custom is None:
        return default
    try:
        return custom.get(key, default)
    except AttributeError:
        return default


def get_rollout_custom_nested_value(config, path: tuple[str, ...], default: Any) -> Any:
    custom = getattr(config, "custom", None)
    value = _get_nested(_as_mapping(custom), path, default)
    return default if value is None else value


def normalize_beam_return_mode(value: Any, *, return_all_beams: Any = None) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"all_beams", "all"}:
            return "all_beams"
        if normalized in {"best_only", "best"}:
            return "best_only"
    if return_all_beams is not None:
        return "all_beams" if bool(return_all_beams) else "best_only"
    return "best_only"


def resolve_beam_search_config(
    source: MutableMapping[str, Any],
    *,
    config,
    request_id: str,
    default_width: int = 32,
    default_max_tokens: int = 16,
) -> BeamSearchConfig:
    beam_search_params = _as_dict(_pop_first(source, (BEAM_SEARCH_PARAMS_KEY,), {}))
    beam_width = int(
        _pop_first(
            source,
            (BEAM_WIDTH_KEY, *LEGACY_BEAM_WIDTH_KEYS),
            get_rollout_custom_value(
                config,
                BEAM_WIDTH_KEY,
                get_rollout_custom_value(config, LEGACY_BEAM_WIDTH_KEYS[0], default_width),
            ),
        )
    )
    beam_index = int(_pop_first(source, (BEAM_INDEX_KEY, *LEGACY_BEAM_INDEX_KEYS), 0))
    beam_group_id = str(_pop_first(source, (BEAM_GROUP_ID_KEY, *LEGACY_BEAM_GROUP_ID_KEYS), request_id))
    beam_return_mode = normalize_beam_return_mode(
        _pop_first(source, (BEAM_RETURN_MODE_KEY,), None),
        return_all_beams=_pop_first(source, ("return_all_beams",), None),
    )
    max_tokens = int(
        _pop_first(
            beam_search_params,
            ("max_tokens",),
            _pop_first(
                source,
                LEGACY_BEAM_MAX_TOKENS_KEYS,
                get_rollout_custom_nested_value(
                    config,
                    (BEAM_SEARCH_PARAMS_KEY, "max_tokens"),
                    get_rollout_custom_nested_value(
                        config,
                        (DECODE_CONFIG_KEY, "item_generation", "max_tokens"),
                        get_rollout_custom_value(config, LEGACY_BEAM_MAX_TOKENS_KEYS[0], default_max_tokens),
                    ),
                ),
            ),
        )
    )
    length_penalty = float(_pop_first(beam_search_params, ("length_penalty",), 1.0))
    ignore_eos = bool(_pop_first(beam_search_params, ("ignore_eos",), False))
    temperature = float(
        _pop_first(
            beam_search_params,
            ("temperature",),
            _pop_first(source, ("stage2_temperature",), 0.0),
        )
    )
    top_p = float(_pop_first(beam_search_params, ("top_p",), _pop_first(source, ("stage2_top_p",), 1.0)))
    top_k = int(_pop_first(beam_search_params, ("top_k",), _pop_first(source, ("stage2_top_k",), -1)))

    return BeamSearchConfig(
        width=max(1, beam_width),
        index=beam_index,
        group_id=beam_group_id,
        return_mode=beam_return_mode,
        max_tokens=max_tokens,
        length_penalty=length_penalty,
        ignore_eos=ignore_eos,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )


def resolve_two_stage_decode_config(
    source: MutableMapping[str, Any],
    *,
    config,
    response_length: int,
) -> TwoStageDecodeConfig:
    decode_config = _as_dict(_pop_first(source, (DECODE_CONFIG_KEY,), {}))
    reasoning_cfg = _as_dict(decode_config.get("reasoning"))
    item_cfg = _as_dict(decode_config.get("item_generation"))

    reasoning_max_tokens = int(
        _pop_first(
            reasoning_cfg,
            ("max_tokens",),
            _pop_first(
                source,
                LEGACY_REASONING_MAX_TOKENS_KEYS,
                get_rollout_custom_nested_value(
                    config,
                    (DECODE_CONFIG_KEY, "reasoning", "max_tokens"),
                    get_rollout_custom_value(config, LEGACY_REASONING_MAX_TOKENS_KEYS[0], response_length),
                ),
            ),
        )
    )
    reasoning_stop = _as_str_list(
        _pop_first(
            reasoning_cfg,
            ("stop",),
            get_rollout_custom_nested_value(config, (DECODE_CONFIG_KEY, "reasoning", "stop"), DEFAULT_REASONING_STOP),
        )
    )
    include_stop_str_in_output = bool(
        _pop_first(
            reasoning_cfg,
            ("include_stop_str_in_output",),
            get_rollout_custom_nested_value(
                config,
                (DECODE_CONFIG_KEY, "reasoning", "include_stop_str_in_output"),
                True,
            ),
        )
    )
    item_prefix_text = str(
        _pop_first(
            item_cfg,
            ("prefix_text",),
            get_rollout_custom_nested_value(
                config,
                (DECODE_CONFIG_KEY, "item_generation", "prefix_text"),
                DEFAULT_ITEM_PREFIX_TEXT,
            ),
        )
    )
    item_max_tokens = int(
        _pop_first(
            item_cfg,
            ("max_tokens",),
            get_rollout_custom_nested_value(
                config,
                (DECODE_CONFIG_KEY, "item_generation", "max_tokens"),
                get_rollout_custom_nested_value(
                    config,
                    (BEAM_SEARCH_PARAMS_KEY, "max_tokens"),
                    get_rollout_custom_value(config, LEGACY_BEAM_MAX_TOKENS_KEYS[0], 16),
                ),
            ),
        )
    )

    return TwoStageDecodeConfig(
        reasoning=DecodePhaseConfig(
            max_tokens=reasoning_max_tokens,
            stop=reasoning_stop,
            include_stop_str_in_output=include_stop_str_in_output,
        ),
        item_generation=DecodePhaseConfig(
            max_tokens=item_max_tokens,
            prefix_text=item_prefix_text,
        ),
    )


def build_two_stage_sampling_params(
    *,
    reasoning_max_tokens: int,
    item_max_tokens: int,
    beam_width: int,
    return_all_beams: bool = False,
) -> dict[str, Any]:
    return {
        BEAM_WIDTH_KEY: beam_width,
        BEAM_RETURN_MODE_KEY: "all_beams" if return_all_beams else "best_only",
        BEAM_SEARCH_PARAMS_KEY: {"max_tokens": item_max_tokens},
        DECODE_CONFIG_KEY: {
            "reasoning": {"max_tokens": reasoning_max_tokens},
            "item_generation": {"max_tokens": item_max_tokens},
        },
    }


def _as_dict(value: Any) -> dict[str, Any]:
    mapping = _as_mapping(value)
    if mapping is None:
        return {}
    return dict(mapping)


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if hasattr(value, "items"):
        return dict(value.items())
    if hasattr(value, "keys") and hasattr(value, "get"):
        return value
    return None


def _as_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _get_nested(mapping: Mapping[str, Any] | None, path: tuple[str, ...], default: Any) -> Any:
    current: Any = mapping
    for key in path:
        next_mapping = _as_mapping(current)
        if next_mapping is None or key not in next_mapping:
            return default
        current = next_mapping[key]
    return current


def _pop_first(mapping: MutableMapping[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    for key in keys:
        if key in mapping:
            return mapping.pop(key)
    return default
