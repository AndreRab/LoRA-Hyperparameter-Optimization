from __future__ import annotations

import argparse
from pathlib import Path

from .config import (
    BaselineConfig,
    DatasetConfig,
    HPOConfig,
    SearchSpaceConfig,
    TrainingConfig,
    dataclass_from_json,
)


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset-config", default="configs/dataset.json")
    parser.add_argument("--training-config", default="configs/training.json")
    parser.add_argument("--baseline-config", default="configs/baseline.json")
    parser.add_argument("--search-space-config", default="configs/search_space.json")
    parser.add_argument("--hpo-config", default="configs/hpo.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    return parser


def load_runtime_configs(args) -> tuple[DatasetConfig, TrainingConfig, BaselineConfig, SearchSpaceConfig, HPOConfig]:
    dataset = dataclass_from_json(DatasetConfig, args.dataset_config)
    training = dataclass_from_json(TrainingConfig, args.training_config)
    baseline = dataclass_from_json(BaselineConfig, args.baseline_config)
    search_space = dataclass_from_json(SearchSpaceConfig, args.search_space_config)
    hpo = dataclass_from_json(HPOConfig, args.hpo_config)

    if args.output_dir:
        training.output_root = args.output_dir
    if args.data_dir:
        training.data_root = args.data_dir

    Path(training.output_root).mkdir(parents=True, exist_ok=True)
    Path(training.data_root).mkdir(parents=True, exist_ok=True)
    return dataset, training, baseline, search_space, hpo
