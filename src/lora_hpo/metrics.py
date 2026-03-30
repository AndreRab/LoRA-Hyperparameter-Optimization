from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def extract_numeric_answer(text: str) -> str:
    marker_match = re.search(r"####\s*([-+]?[\d,]+(?:\.\d+)?)", text)
    if marker_match:
        return marker_match.group(1).replace(",", "")

    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not matches:
        return normalize_text(text)
    return matches[-1].replace(",", "")


def score_prediction(prediction: str, reference: str, source: str) -> float:
    if source == "gsm8k":
        return float(extract_numeric_answer(prediction) == extract_numeric_answer(reference))
    return float(normalize_text(prediction) == normalize_text(reference))
