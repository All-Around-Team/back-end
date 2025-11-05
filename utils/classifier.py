from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch, asyncio

tokenizer = AutoTokenizer.from_pretrained("ProtectAI/deberta-v3-base-prompt-injection-v2")
model = AutoModelForSequenceClassification.from_pretrained("ProtectAI/deberta-v3-base-prompt-injection-v2")

device = 0 if torch.cuda.is_available() else -1
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    truncation=True,
    max_length=512,
    device=device,
)

SAFE_LABELS = {"CLEAN", "NO_INJECTION", "BENIGN"}


def _normalize_pipeline_output(raw):
    if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
        raw = raw[0]

    if not raw or not isinstance(raw, list):
        return "error", 0.0

    best = max(raw, key=lambda x: x.get("score", 0.0))
    label = best.get("label", "").upper()
    score = float(best.get("score", 0.0))

    if label in SAFE_LABELS:
        return "appropriate", score
    return "inappropriate", score


def _sync_classify(text: str):
    text = text.strip()
    if not text:
        return "appropriate", 0.0

    raw = classifier(text)
    return _normalize_pipeline_output(raw)


async def classify_text(text: str):
    return await asyncio.to_thread(_sync_classify, text)
