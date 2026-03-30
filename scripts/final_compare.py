#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lora_hpo.training import compare_results


def load_result(path: str) -> dict:
    return json.loads(Path(path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a final comparison report from test results.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--es", required=True)
    parser.add_argument("--pso", required=True)
    parser.add_argument("--output", default="outputs/final_comparison.json")
    args = parser.parse_args()

    compare_results(
        args.output,
        baseline_result=load_result(args.baseline),
        es_result=load_result(args.es),
        pso_result=load_result(args.pso),
    )
    print(f"Saved final comparison to {args.output}")


if __name__ == "__main__":
    main()
