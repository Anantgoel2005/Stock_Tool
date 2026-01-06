"""
Core package for the desktop stock prediction tool.

Modules:
    config: Global configuration constants and helpers.
    features: Feature engineering routines for price data.
    news: NewsAPI integration helpers.
    sentiment: FinBERT sentiment scoring utilities.
    modeling: Model training and live prediction logic.
"""

from .config import (
    DEFAULT_HORIZON,
    DEFAULT_NEWS_LOOKBACK,
    DEFAULT_SYMBOL,
    MODEL_FEATURES,
    MODELS_DIR,
)
from .modeling import train_and_save_model, get_live_prediction_with_reasoning

__all__ = [
    "DEFAULT_HORIZON",
    "DEFAULT_NEWS_LOOKBACK",
    "DEFAULT_SYMBOL",
    "MODEL_FEATURES",
    "MODELS_DIR",
    "train_and_save_model",
    "get_live_prediction_with_reasoning",
]

