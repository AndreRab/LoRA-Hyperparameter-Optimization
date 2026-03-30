from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DatasetConfig:
    gsm8k_enabled: bool = True
    gsm8k_name: str = "openai/gsm8k"
    gsm8k_config: str = "main"
    superni_enabled: bool = False
    superni_name: str = "Muennighoff/natural-instructions"
    max_gsm8k_train: int | None = 512
    max_gsm8k_validation: int | None = 128
    max_gsm8k_test: int | None = 256
    max_superni_train: int | None = 0
    max_superni_validation: int | None = 0
    max_superni_test: int | None = 0
    seed: int = 42


@dataclass
class TrainingConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    output_root: str = "outputs"
    data_root: str = "artifacts/data"
    max_input_length: int = 512
    max_target_length: int = 128
    eval_max_new_tokens: int = 96
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 10
    save_total_limit: int = 2
    seed: int = 42
    use_bf16: bool = True
    load_in_8bit: bool = False
    trust_remote_code: bool = False
    train_on_validation_too: bool = False


@dataclass
class BaselineConfig:
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    rank: int = 4
    alpha: int = 16
    dropout: float = 0.05
    learning_rate: float = 2e-4
    batch_size: int = 8
    steps: int = 300
    scheduler: str = "cosine"


@dataclass
class SearchSpaceConfig:
    rank: list[int] = field(default_factory=lambda: [4, 8, 16, 32, 64])
    alpha: list[int] = field(default_factory=lambda: [8, 16, 32, 64, 128])
    dropout: list[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.15])
    learning_rate: list[float] = field(
        default_factory=lambda: [5e-6, 1e-5, 5e-5, 1e-4, 2e-4, 5e-4]
    )
    batch_size: list[int] = field(default_factory=lambda: [2, 4, 8, 16])
    steps: list[int] = field(default_factory=lambda: [100, 200, 300, 500])
    scheduler: list[str] = field(default_factory=lambda: ["linear", "cosine", "constant"])
    target_modules: list[list[str]] = field(
        default_factory=lambda: [["q_proj", "v_proj"]]
    )


@dataclass
class HPOConfig:
    budget: int = 12
    population_size: int = 4
    elite_size: int = 2
    mutation_rate: float = 0.35
    inertia: float = 0.6
    cognitive: float = 1.4
    social: float = 1.4
    seed: int = 42


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def dataclass_from_json(cls: type[Any], path: str | Path) -> Any:
    payload = load_json(path)
    return cls(**payload)


def dataclass_to_json(instance: Any, path: str | Path) -> None:
    save_json(path, asdict(instance))
