from __future__ import annotations

from typing import Iterable

from peft import LoraConfig, TaskType, get_peft_model

from .config import BaselineConfig


def resolve_target_modules(model, requested: Iterable[str]) -> list[str]:
    available = set()
    for name, _module in model.named_modules():
        suffix = name.split(".")[-1]
        available.add(suffix)

    resolved = [name for name in requested if name in available]
    if resolved:
        return resolved

    fallback_pairs = [
        ("q_proj", "v_proj"),
        ("wq", "wv"),
        ("query_key_value",),
    ]
    for pair in fallback_pairs:
        current = [name for name in pair if name in available]
        if current:
            return current

    raise ValueError(
        f"Could not resolve LoRA target modules from {list(requested)}. "
        f"Available module suffixes include: {sorted(list(available))[:30]}"
    )


def attach_lora(model, baseline: BaselineConfig):
    target_modules = resolve_target_modules(model, baseline.target_modules)
    lora_config = LoraConfig(
        r=baseline.rank,
        lora_alpha=baseline.alpha,
        lora_dropout=baseline.dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, lora_config), target_modules
