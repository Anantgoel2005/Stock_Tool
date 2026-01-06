import os
from pathlib import Path

# --- Model / Feature configuration ------------------------------------------------

MODEL_FEATURES = [
    "SMA_50",
    "SMA_200",
    "RSI",
    "sentiment_score",
    "Volume",
    "MACD",
    "Signal_Line",
    "Return_1D",
    "Return_3D",
    "Return_5D",
]

TRAINING_YEARS = 5

# --- External services ------------------------------------------------------------

FINBERT_MODEL_NAME = "ProsusAI/finbert"
YAHOO_RSS_URL = (
    "https://feeds.finance.yahoo.com/rss/2.0/headline"
    "?s={symbol}&region=US&lang=en-US"
)
GOOGLE_NEWS_RSS_URL = (
    "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
)

# --- Paths ------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- UI defaults ------------------------------------------------------------------

DEFAULT_SYMBOL = "HDFCBANK.NS"
DEFAULT_HORIZON = "short_term"
DEFAULT_NEWS_LOOKBACK = 1

