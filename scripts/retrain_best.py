#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lora_hpo.cli import common_parser, load_runtime_configs
from lora_hpo.config import BaselineConfig


def main() -> None:
    parser = common_parser("Retrain a selected config and evaluate it on the test split.")
    parser.add_argument("--config-json", required=True, help="Path to a best_config.json or compatible config file.")
    parser.add_argument("--run-name", default="retrain", help="Output subdirectory name.")
    args = parser.parse_args()
    from lora_hpo.datasets import load_prepared_dataset
    from lora_hpo.training import retrain_and_test

    _dataset_cfg, training, _baseline, _search_space, _hpo = load_runtime_configs(args)

    payload = json.loads(Path(args.config_json).read_text())
    config_payload = payload["config"] if "config" in payload else payload
    baseline = BaselineConfig(**config_payload)

    dataset = load_prepared_dataset(training.data_root)
    output_dir = Path(training.output_root) / args.run_name
    result = retrain_and_test(dataset, training, baseline, output_dir)
    print(f"Test score: {result['test_score']:.4f}")


if __name__ == "__main__":
    main()
