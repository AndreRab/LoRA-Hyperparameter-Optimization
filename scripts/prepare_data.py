#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lora_hpo.cli import common_parser, load_runtime_configs
from lora_hpo.datasets import build_mixed_dataset, save_dataset_dict


def main() -> None:
    parser = common_parser("Download, standardize, and save the experiment dataset.")
    args = parser.parse_args()
    dataset_config, training_config, _baseline, _search_space, _hpo = load_runtime_configs(args)

    dataset = build_mixed_dataset(dataset_config)
    save_dataset_dict(dataset, training_config.data_root, dataset_config)
    print(f"Prepared dataset saved to {training_config.data_root}")


if __name__ == "__main__":
    main()
