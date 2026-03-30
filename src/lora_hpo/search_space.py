from __future__ import annotations

import random
from dataclasses import asdict
from typing import Any

from .config import BaselineConfig, SearchSpaceConfig


SEARCH_KEYS = [
    "rank",
    "alpha",
    "dropout",
    "learning_rate",
    "batch_size",
    "steps",
    "scheduler",
    "target_modules",
]


def sample_config(space: SearchSpaceConfig, rng: random.Random) -> BaselineConfig:
    payload = {}
    raw = asdict(space)
    for key in SEARCH_KEYS:
        payload[key] = rng.choice(raw[key])
    return BaselineConfig(**payload)


def mutate_config(
    config: BaselineConfig,
    space: SearchSpaceConfig,
    rng: random.Random,
    mutation_rate: float,
) -> BaselineConfig:
    payload = asdict(config)
    raw = asdict(space)
    for key in SEARCH_KEYS:
        if rng.random() < mutation_rate:
            payload[key] = rng.choice(raw[key])
    return BaselineConfig(**payload)


def config_to_key(config: BaselineConfig) -> tuple[Any, ...]:
    payload = asdict(config)
    return tuple(tuple(v) if isinstance(v, list) else v for v in payload.values())


def config_to_dict(config: BaselineConfig) -> dict[str, Any]:
    return asdict(config)


def index_bounds(space: SearchSpaceConfig) -> dict[str, int]:
    raw = asdict(space)
    return {key: len(raw[key]) - 1 for key in SEARCH_KEYS}


def position_to_config(position: dict[str, float], space: SearchSpaceConfig) -> BaselineConfig:
    raw = asdict(space)
    payload = {}
    for key in SEARCH_KEYS:
        idx = int(round(position[key]))
        idx = max(0, min(idx, len(raw[key]) - 1))
        payload[key] = raw[key][idx]
    return BaselineConfig(**payload)
