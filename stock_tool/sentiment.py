import logging
from functools import lru_cache
from typing import Iterable, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import FINBERT_MODEL_NAME


@lru_cache(maxsize=1)
def _load_finbert():
    try:
        logging.info("Loading FinBERT model (%s)...", FINBERT_MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
        logging.info("FinBERT model loaded successfully.")
        return tokenizer, model
    except Exception as exc:
        logging.error(
            "Error loading FinBERT model: %s. Sentiment scores will default to 0.", exc
        )
        return None, None


def get_finbert_sentiment(text: str) -> float:
    if not text:
        return 0.0

    tokenizer, model = _load_finbert()
    if tokenizer is None or model is None:
        return 0.0

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    positive_score = predictions[:, 0].item()
    negative_score = predictions[:, 1].item()
    return positive_score - negative_score


def batch_sentiment_scores(texts: Iterable[str]) -> List[float]:
    scores: List[float] = []
    for text in texts:
        scores.append(get_finbert_sentiment(text))
    return scores

