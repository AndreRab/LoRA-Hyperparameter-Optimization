# Experiment Scripts

This repository now contains a runnable scaffold for the LoRA hyperparameter search experiment described in the README.

The code is organized into:

- `src/lora_hpo/` for shared Python logic
- `configs/` for editable JSON configuration files
- `scripts/` for command-line entrypoints
- `outputs/` for training runs and comparison reports
- `artifacts/data/` for the prepared dataset cache

## Install

Create a Python environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration files

- `configs/dataset.json`
  Controls which datasets are loaded and how many examples are used.
  The default setup uses GSM8K only so the experiment is easier to run first.

- `configs/training.json`
  Controls the base model, output paths, sequence lengths, mixed precision, and shared training settings.

- `configs/baseline.json`
  Defines the standard LoRA baseline.
  The default baseline follows the project note: `target_modules = [q_proj, v_proj]`, `rank = 4`.

- `configs/search_space.json`
  Defines the discrete search space used by both ES and PSO.

- `configs/hpo.json`
  Defines the search budget and the ES / PSO optimizer hyperparameters.

## Scripts

- `scripts/prepare_data.py`
  Downloads the configured datasets, converts them to a shared instruction format, and saves them to `artifacts/data/`.

- `scripts/run_baseline.py`
  Runs the standard LoRA baseline on the prepared training split and reports the validation score.

- `scripts/run_hpo_es.py`
  Runs Evolution Strategies over the LoRA search space.
  It writes per-trial outputs into `outputs/es/` and stores the best validation config in `outputs/es/best_config.json`.

- `scripts/run_hpo_pso.py`
  Runs Particle Swarm Optimization over the same search space.
  It writes per-trial outputs into `outputs/pso/` and stores the best validation config in `outputs/pso/best_config.json`.

- `scripts/retrain_best.py`
  Retrains a selected config and evaluates it on the test split.
  This is the script to use after baseline, ES, or PSO to get final test numbers.

- `scripts/final_compare.py`
  Combines the final test JSON files into one comparison report.

- `scripts/run_full_experiment.py`
  Runs the entire pipeline in one command:
  prepare data, run the baseline, run ES, run PSO, retrain the best ES and PSO configs, evaluate on test, and write a final comparison JSON.

## Recommended run order

### 1. Prepare the dataset

```bash
python3 scripts/prepare_data.py
```

### 2. Run the baseline

```bash
python3 scripts/run_baseline.py
```

### 3. Run ES and PSO

```bash
python3 scripts/run_hpo_es.py
python3 scripts/run_hpo_pso.py
```

### 4. Retrain the best ES and PSO configurations on the test setup

```bash
python3 scripts/retrain_best.py --config-json outputs/es/best_config.json --run-name es_test
python3 scripts/retrain_best.py --config-json outputs/pso/best_config.json --run-name pso_test
python3 scripts/retrain_best.py --config-json configs/baseline.json --run-name baseline_test
```

### 5. Build the final comparison report

```bash
python3 scripts/final_compare.py \
  --baseline outputs/baseline_test/test_result.json \
  --es outputs/es_test/test_result.json \
  --pso outputs/pso_test/test_result.json
```

## One-command run

```bash
python3 scripts/run_full_experiment.py
```

## Notes

- The provided metric implementation is intentionally simple so the experiment can run end to end without extra task-specific evaluation plumbing.
  GSM8K uses extracted numeric exact match.
  SuperNI examples use normalized text exact match.

- The default configs are chosen to make the scaffold practical to start with, not to guarantee the best final score.

- For 7B models or larger runs, you will likely need to lower `batch_size`, lower `max_*_examples`, or enable a quantized workflow later.
