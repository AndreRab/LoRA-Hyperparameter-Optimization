from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .config import BaselineConfig, TrainingConfig, save_json
from .lora import attach_lora
from .metrics import score_prediction


PROMPT_TEMPLATE = "### Instruction:\n{prompt}\n\n### Response:\n"


def _dtype(use_bf16: bool):
    if not torch.cuda.is_available():
        return None
    return torch.bfloat16 if use_bf16 and torch.cuda.is_bf16_supported() else torch.float16


def load_tokenizer(model_name: str, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(training: TrainingConfig):
    kwargs: dict[str, Any] = {
        "trust_remote_code": training.trust_remote_code,
    }
    dtype = _dtype(training.use_bf16)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    if training.load_in_8bit:
        kwargs["load_in_8bit"] = True
    return AutoModelForCausalLM.from_pretrained(training.model_name, **kwargs)


def tokenize_supervised_dataset(
    dataset: Dataset,
    tokenizer,
    training: TrainingConfig,
) -> Dataset:
    def convert(row: dict[str, Any]) -> dict[str, Any]:
        prompt_text = PROMPT_TEMPLATE.format(prompt=row["prompt"])
        target_text = row["target"] + tokenizer.eos_token

        prompt_ids = tokenizer(
            prompt_text,
            truncation=True,
            max_length=training.max_input_length,
            add_special_tokens=False,
        )["input_ids"]
        target_ids = tokenizer(
            target_text,
            truncation=True,
            max_length=training.max_target_length,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = prompt_ids + target_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + target_ids

        max_total = training.max_input_length + training.max_target_length
        input_ids = input_ids[:max_total]
        attention_mask = attention_mask[:max_total]
        labels = labels[:max_total]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_text": prompt_text,
            "reference_text": row["target"],
            "source": row["source"],
            "task_name": row["task_name"],
        }

    return dataset.map(convert, remove_columns=dataset.column_names)


def _build_training_args(output_dir: str | Path, training: TrainingConfig, baseline: BaselineConfig):
    per_device_batch_size = max(1, baseline.batch_size)
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=max(1, min(4, per_device_batch_size)),
        max_steps=baseline.steps,
        learning_rate=baseline.learning_rate,
        lr_scheduler_type=baseline.scheduler,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        warmup_ratio=training.warmup_ratio,
        weight_decay=training.weight_decay,
        logging_steps=training.logging_steps,
        save_steps=max(50, training.logging_steps),
        evaluation_strategy="no",
        save_total_limit=training.save_total_limit,
        bf16=torch.cuda.is_available() and training.use_bf16 and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not (training.use_bf16 and torch.cuda.is_bf16_supported()),
        report_to=[],
        seed=training.seed,
        remove_unused_columns=False,
    )


def train_short_run(
    dataset_dict: DatasetDict,
    training: TrainingConfig,
    baseline: BaselineConfig,
    run_dir: str | Path,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    set_seed(training.seed)

    tokenizer = load_tokenizer(training.model_name, training.trust_remote_code)
    tokenized_train = tokenize_supervised_dataset(dataset_dict["train"], tokenizer, training)
    tokenized_validation = tokenize_supervised_dataset(dataset_dict["validation"], tokenizer, training)

    model = load_model(training)
    model, resolved_modules = attach_lora(model, baseline)
    model.print_trainable_parameters()

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    args = _build_training_args(run_path, training, baseline)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    train_result = trainer.train()
    trainer.save_model(str(run_path / "adapter"))
    tokenizer.save_pretrained(str(run_path / "adapter"))

    metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        dataset=tokenized_validation,
        training=training,
        sample_limit=min(len(tokenized_validation), 128),
    )
    result = {
        "config": asdict(baseline),
        "resolved_target_modules": resolved_modules,
        "train_runtime_seconds": train_result.metrics.get("train_runtime"),
        "train_loss": train_result.metrics.get("train_loss"),
        "validation_score": metrics["score"],
        "validation_examples": metrics["num_examples"],
    }
    save_json(run_path / "result.json", result)
    return result


def evaluate_model(model, tokenizer, dataset: Dataset, training: TrainingConfig, sample_limit: int) -> dict[str, Any]:
    model.eval()
    rows = dataset.select(range(min(sample_limit, len(dataset))))
    total = 0.0
    predictions = []
    device = model.device
    for row in rows:
        batch = tokenizer(
            row["prompt_text"],
            return_tensors="pt",
            truncation=True,
            max_length=training.max_input_length,
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            output = model.generate(
                **batch,
                max_new_tokens=training.eval_max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(output[0][batch["input_ids"].shape[1]:], skip_special_tokens=True)
        score = score_prediction(generated, row["reference_text"], row["source"])
        total += score
        predictions.append(
            {
                "source": row["source"],
                "task_name": row["task_name"],
                "prediction": generated,
                "reference": row["reference_text"],
                "score": score,
            }
        )

    score = total / max(1, len(predictions))
    return {"score": score, "num_examples": len(predictions), "predictions": predictions}


def retrain_and_test(
    dataset_dict: DatasetDict,
    training: TrainingConfig,
    baseline: BaselineConfig,
    run_dir: str | Path,
) -> dict[str, Any]:
    if training.train_on_validation_too:
        merged_train = Dataset.from_list(
            list(dataset_dict["train"]) + list(dataset_dict["validation"])
        )
        working = DatasetDict(
            {"train": merged_train, "validation": dataset_dict["validation"], "test": dataset_dict["test"]}
        )
    else:
        working = dataset_dict

    result = train_short_run(working, training, baseline, run_dir)

    tokenizer = load_tokenizer(training.model_name, training.trust_remote_code)
    tokenized_test = tokenize_supervised_dataset(dataset_dict["test"], tokenizer, training)
    model = load_model(training)
    model, _resolved = attach_lora(model, baseline)

    adapter_dir = Path(run_dir) / "adapter"
    if adapter_dir.exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))
    if torch.cuda.is_available():
        model = model.cuda()

    test_metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        dataset=tokenized_test,
        training=training,
        sample_limit=min(len(tokenized_test), 256),
    )
    result["test_score"] = test_metrics["score"]
    result["test_examples"] = test_metrics["num_examples"]
    save_json(Path(run_dir) / "test_result.json", result)
    (Path(run_dir) / "test_predictions.json").write_text(json.dumps(test_metrics, indent=2) + "\n")
    return result


def compare_results(output_path: str | Path, baseline_result: dict[str, Any], es_result: dict[str, Any], pso_result: dict[str, Any]) -> None:
    comparison = {
        "baseline": baseline_result,
        "es": es_result,
        "pso": pso_result,
        "best_method": max(
            [("baseline", baseline_result), ("es", es_result), ("pso", pso_result)],
            key=lambda item: item[1].get("test_score", -math.inf),
        )[0],
    }
    save_json(output_path, comparison)
