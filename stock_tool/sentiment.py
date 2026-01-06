import logging
import os
from functools import lru_cache
from typing import Iterable, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import FINBERT_MODEL_NAME


DISABLE_FINBERT = os.getenv("DISABLE_FINBERT", "0") == "1"
SENTIMENT_METHOD = os.getenv("SENTIMENT_METHOD", "auto").lower()  # "finbert", "vader", or "auto"


@lru_cache(maxsize=1)
def _load_vader():
    """Load VADER sentiment analyzer (lightweight, no GPU needed)."""
    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
        
        # Download VADER lexicon if not present
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            logging.info("Downloading VADER lexicon...")
            nltk.download('vader_lexicon', quiet=True)
        
        analyzer = SentimentIntensityAnalyzer()
        logging.info("VADER sentiment analyzer loaded successfully.")
        return analyzer
    except Exception as exc:
        logging.error("Error loading VADER: %s. Sentiment will default to 0.", exc)
        return None


@lru_cache(maxsize=1)
def _load_finbert():
    if DISABLE_FINBERT or SENTIMENT_METHOD == "vader":
        logging.info(
            "FinBERT disabled or VADER selected; skipping FinBERT loading."
        )
        return None, None

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


def get_vader_sentiment(text: str) -> float:
    """Get sentiment using VADER (lightweight, no GPU needed)."""
    if not text:
        return 0.0
    
    analyzer = _load_vader()
    if analyzer is None:
        return 0.0
    
    try:
        scores = analyzer.polarity_scores(text)
        # Compound score ranges from -1 (most negative) to +1 (most positive)
        # Scale to roughly match FinBERT's range
        return scores['compound']
    except Exception as exc:
        logging.error("Error computing VADER sentiment: %s", exc)
        return 0.0


def get_finbert_sentiment(text: str) -> float:
    if not text:
        return 0.0

    tokenizer, model = _load_finbert()
    if tokenizer is None or model is None:
        return 0.0

    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        positive_score = predictions[:, 0].item()
        negative_score = predictions[:, 1].item()
        return positive_score - negative_score
    except RuntimeError as exc:
        # Catch CUDA/CPU OOM and fail soft.
        logging.error(
            "Runtime error during FinBERT inference (%s). Treating sentiment as 0.",
            exc,
        )
        return 0.0


def get_sentiment(text: str) -> float:
    """Get sentiment score using the configured method."""
    if SENTIMENT_METHOD == "vader":
        return get_vader_sentiment(text)
    elif SENTIMENT_METHOD == "finbert":
        return get_finbert_sentiment(text)
    else:  # "auto" - try FinBERT, fallback to VADER
        score = get_finbert_sentiment(text)
        if score == 0.0:
            # FinBERT failed or disabled, try VADER
            return get_vader_sentiment(text)
        return score


def batch_sentiment_scores(texts: Iterable[str]) -> List[float]:
    scores: List[float] = []
    for text in texts:
        scores.append(get_sentiment(text))
    return scores
