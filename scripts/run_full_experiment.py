#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lora_hpo.cli import common_parser, load_runtime_configs
from lora_hpo.config import BaselineConfig
from lora_hpo.datasets import build_mixed_dataset, save_dataset_dict
from lora_hpo.hpo import run_es, run_pso
from lora_hpo.training import compare_results, retrain_and_test, train_short_run


def main() -> None:
    parser = common_parser("Run the entire LoRA HPO experiment end to end.")
    args = parser.parse_args()
    dataset_config, training, baseline, search_space, hpo = load_runtime_configs(args)

    dataset = build_mixed_dataset(dataset_config)
    save_dataset_dict(dataset, training.data_root, dataset_config)

    baseline_validation = train_short_run(
        dataset,
        training,
        baseline,
        Path(training.output_root) / "baseline_validation",
    )
    print(f"Baseline validation score: {baseline_validation['validation_score']:.4f}")

    best_es = run_es(dataset, training, search_space, hpo, Path(training.output_root) / "es")
    print(f"Best ES validation score: {best_es['validation_score']:.4f}")

    best_pso = run_pso(dataset, training, search_space, hpo, Path(training.output_root) / "pso")
    print(f"Best PSO validation score: {best_pso['validation_score']:.4f}")

    baseline_test = retrain_and_test(
        dataset,
        training,
        baseline,
        Path(training.output_root) / "baseline_test",
    )

    es_test = retrain_and_test(
        dataset,
        training,
        BaselineConfig(**best_es["config"]),
        Path(training.output_root) / "es_test",
    )

    pso_test = retrain_and_test(
        dataset,
        training,
        BaselineConfig(**best_pso["config"]),
        Path(training.output_root) / "pso_test",
    )

    compare_results(
        Path(training.output_root) / "final_comparison.json",
        baseline_result=baseline_test,
        es_result=es_test,
        pso_result=pso_test,
    )

    summary = {
        "baseline_validation": baseline_validation,
        "best_es": best_es,
        "best_pso": best_pso,
        "baseline_test": baseline_test,
        "es_test": es_test,
        "pso_test": pso_test,
    }
    (Path(training.output_root) / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Finished full experiment. Outputs are in {training.output_root}")


if __name__ == "__main__":
    main()
