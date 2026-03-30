#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lora_hpo.cli import common_parser, load_runtime_configs
from lora_hpo.datasets import load_prepared_dataset
from lora_hpo.hpo import run_es


def main() -> None:
    parser = common_parser("Run Evolution Strategies over the LoRA search space.")
    args = parser.parse_args()
    _dataset_cfg, training, _baseline, search_space, hpo = load_runtime_configs(args)

    dataset = load_prepared_dataset(training.data_root)
    best = run_es(dataset, training, search_space, hpo, Path(training.output_root) / "es")
    print(f"Best ES validation score: {best['validation_score']:.4f}")


if __name__ == "__main__":
    main()
