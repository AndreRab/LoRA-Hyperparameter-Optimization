#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_result(path: str) -> dict:
    return json.loads(Path(path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a final comparison report from test results.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--es", required=True)
    parser.add_argument("--pso", required=True)
    parser.add_argument("--output", default="outputs/final_comparison.json")
    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline": load_result(args.baseline),
        "es": load_result(args.es),
        "pso": load_result(args.pso),
    }
    payload["best_method"] = max(
        ("baseline", "es", "pso"),
        key=lambda key: payload[key].get("test_score", float("-inf")),
    )
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved final comparison to {args.output}")


if __name__ == "__main__":
    main()
