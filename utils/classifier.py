import asyncio
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MODEL_NAME = "ProtectAI/deberta-v3-base-prompt-injection-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = 0 if torch.cuda.is_available() else -1

classifier = pipeline(
    task="text-classification",
    model=model,
    tokenizer=tokenizer,
    truncation=True,
    max_length=512,
    device=device,
)

SAFE_LABELS = {"SAFE"}


def _normalize_output(result: list[dict[str, Any]] | dict[str, Any]) -> tuple[str, float]:
    if isinstance(result, list):
        result = result[0]

    label = result.get("label", "").upper()
    score = float(result.get("score", 0.0))

    is_safe = label in SAFE_LABELS
    return ("appropriate" if is_safe else "inappropriate"), score


def classify_sync(text: str):
    text = text.strip()
    if not text:
        return "appropriate", 0.0

    result = classifier(text)
    return _normalize_output(result)


async def classify_text(text: str):
    if torch.cuda.is_available():
        return classify_sync(text)
    else:
        return await asyncio.to_thread(classify_sync, text)
