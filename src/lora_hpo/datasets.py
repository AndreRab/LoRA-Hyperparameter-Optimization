from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from .config import DatasetConfig


def _take_split(ds: Dataset, limit: int | None) -> Dataset:
    if limit is None:
        return ds
    return ds.select(range(min(limit, len(ds))))


def _standardize_gsm8k(split: Dataset) -> Dataset:
    def convert(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "prompt": row["question"],
            "target": row["answer"],
            "source": "gsm8k",
            "task_name": "gsm8k",
        }

    return split.map(convert, remove_columns=split.column_names)


def _standardize_superni(split: Dataset) -> Dataset:
    def convert(row: dict[str, Any]) -> dict[str, Any]:
        definition = row["definition"][0] if isinstance(row["definition"], list) else row["definition"]
        prompt = f"{definition}\n\nInput:\n{row['inputs']}\n\nAnswer:"
        targets = row["targets"] or [""]
        target = targets[0] if isinstance(targets, list) else targets
        return {
            "prompt": prompt,
            "target": target,
            "source": "superni",
            "task_name": row.get("task_name", "superni"),
        }

    return split.map(convert, remove_columns=split.column_names)


def build_mixed_dataset(config: DatasetConfig) -> DatasetDict:
    train_parts: list[Dataset] = []
    validation_parts: list[Dataset] = []
    test_parts: list[Dataset] = []

    if config.gsm8k_enabled:
        gsm8k = load_dataset(config.gsm8k_name, config.gsm8k_config)
        train_parts.append(_standardize_gsm8k(_take_split(gsm8k["train"], config.max_gsm8k_train)))
        validation_source = gsm8k["test"].shuffle(seed=config.seed)
        validation_parts.append(
            _standardize_gsm8k(_take_split(validation_source, config.max_gsm8k_validation))
        )
        test_source = gsm8k["test"].shuffle(seed=config.seed + 1)
        test_parts.append(_standardize_gsm8k(_take_split(test_source, config.max_gsm8k_test)))

    if config.superni_enabled:
        superni = load_dataset(config.superni_name)
        train_parts.append(
            _standardize_superni(_take_split(superni["train"], config.max_superni_train))
        )
        validation_parts.append(
            _standardize_superni(
                _take_split(superni["validation"], config.max_superni_validation)
            )
        )
        test_parts.append(_standardize_superni(_take_split(superni["test"], config.max_superni_test)))

    if not train_parts:
        raise ValueError("At least one dataset must be enabled.")

    return DatasetDict(
        {
            "train": concatenate_datasets(train_parts).shuffle(seed=config.seed),
            "validation": concatenate_datasets(validation_parts).shuffle(seed=config.seed),
            "test": concatenate_datasets(test_parts).shuffle(seed=config.seed),
        }
    )


def save_dataset_dict(dataset_dict: DatasetDict, output_dir: str | Path, config: DatasetConfig) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))
    (output_path / "dataset_config.json").write_text(
        __import__("json").dumps(asdict(config), indent=2, sort_keys=True) + "\n"
    )


def load_prepared_dataset(data_dir: str | Path) -> DatasetDict:
    return DatasetDict.load_from_disk(str(data_dir))
