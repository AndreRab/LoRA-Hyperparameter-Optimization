from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from datasets import DatasetDict

from .config import BaselineConfig, HPOConfig, SearchSpaceConfig, TrainingConfig, save_json
from .search_space import (
    SEARCH_KEYS,
    config_to_dict,
    config_to_key,
    index_bounds,
    mutate_config,
    position_to_config,
    sample_config,
)
from .training import train_short_run


def run_es(
    dataset_dict: DatasetDict,
    training: TrainingConfig,
    search_space: SearchSpaceConfig,
    hpo: HPOConfig,
    output_dir: str | Path,
) -> dict[str, Any]:
    rng = random.Random(hpo.seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    population = [sample_config(search_space, rng) for _ in range(hpo.population_size)]
    seen: set[tuple[Any, ...]] = set()
    history: list[dict[str, Any]] = []

    while len(history) < hpo.budget:
        evaluated = []
        for config in population:
            key = config_to_key(config)
            if key in seen:
                continue
            seen.add(key)
            run_id = f"trial_{len(history):03d}"
            result = train_short_run(dataset_dict, training, config, output_path / run_id)
            evaluated.append((config, result))
            history.append(
                {
                    "run_id": run_id,
                    "method": "es",
                    "config": config_to_dict(config),
                    "validation_score": result["validation_score"],
                }
            )
            save_json(output_path / "history.json", {"trials": history})
            if len(history) >= hpo.budget:
                break

        if not evaluated:
            population = [sample_config(search_space, rng) for _ in range(hpo.population_size)]
            continue

        evaluated.sort(key=lambda item: item[1]["validation_score"], reverse=True)
        elites = [config for config, _ in evaluated[: max(1, hpo.elite_size)]]
        next_population = list(elites)
        while len(next_population) < hpo.population_size:
            parent = rng.choice(elites)
            child = mutate_config(parent, search_space, rng, hpo.mutation_rate)
            next_population.append(child)
        population = next_population

    best = max(history, key=lambda row: row["validation_score"])
    save_json(output_path / "best_config.json", best)
    return best


def run_pso(
    dataset_dict: DatasetDict,
    training: TrainingConfig,
    search_space: SearchSpaceConfig,
    hpo: HPOConfig,
    output_dir: str | Path,
) -> dict[str, Any]:
    rng = random.Random(hpo.seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    bounds = index_bounds(search_space)
    particles = []
    for _ in range(hpo.population_size):
        position = {key: rng.uniform(0, bounds[key]) for key in SEARCH_KEYS}
        velocity = {key: 0.0 for key in SEARCH_KEYS}
        particles.append(
            {
                "position": position,
                "velocity": velocity,
                "best_position": dict(position),
                "best_score": float("-inf"),
            }
        )

    global_best = {"position": None, "score": float("-inf")}
    seen: set[tuple[Any, ...]] = set()
    history: list[dict[str, Any]] = []

    while len(history) < hpo.budget:
        for particle in particles:
            config = position_to_config(particle["position"], search_space)
            key = config_to_key(config)
            if key in seen:
                for field, upper in bounds.items():
                    particle["position"][field] = rng.uniform(0, upper)
                continue

            seen.add(key)
            run_id = f"trial_{len(history):03d}"
            result = train_short_run(dataset_dict, training, config, output_path / run_id)
            score = result["validation_score"]
            history.append(
                {
                    "run_id": run_id,
                    "method": "pso",
                    "config": asdict(config),
                    "validation_score": score,
                }
            )
            save_json(output_path / "history.json", {"trials": history})

            if score > particle["best_score"]:
                particle["best_score"] = score
                particle["best_position"] = dict(particle["position"])
            if score > global_best["score"]:
                global_best = {"position": dict(particle["position"]), "score": score}

            if len(history) >= hpo.budget:
                break

        if len(history) >= hpo.budget:
            break

        for particle in particles:
            for key in SEARCH_KEYS:
                r1 = rng.random()
                r2 = rng.random()
                cognitive = hpo.cognitive * r1 * (particle["best_position"][key] - particle["position"][key])
                social = 0.0
                if global_best["position"] is not None:
                    social = hpo.social * r2 * (global_best["position"][key] - particle["position"][key])
                particle["velocity"][key] = hpo.inertia * particle["velocity"][key] + cognitive + social
                particle["position"][key] += particle["velocity"][key]
                particle["position"][key] = max(0.0, min(bounds[key], particle["position"][key]))

    best = max(history, key=lambda row: row["validation_score"])
    save_json(output_path / "best_config.json", best)
    return best
