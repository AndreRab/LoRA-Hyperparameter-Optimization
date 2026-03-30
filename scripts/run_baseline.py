#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lora_hpo.cli import common_parser, load_runtime_configs
from lora_hpo.datasets import load_prepared_dataset
from lora_hpo.training import train_short_run


def main() -> None:
    parser = common_parser("Run the standard LoRA baseline on the validation split.")
    args = parser.parse_args()
    _dataset_cfg, training, baseline, _search_space, _hpo = load_runtime_configs(args)

    dataset = load_prepared_dataset(training.data_root)
    output_dir = Path(training.output_root) / "baseline_validation"
    result = train_short_run(dataset, training, baseline, output_dir)
    print(f"Baseline validation score: {result['validation_score']:.4f}")


if __name__ == "__main__":
    main()
