## Dynamic Stock Prediction Web App

This repository now ships a stylized FastAPI + vanilla JS experience with neon/glassmorphism styling. It wraps the reusable `stock_tool` package to train XGBoost models (5-year lookback) and run FinBERT-powered sentiment analysis blended with technical indicators.

### Features
- Train short-term (5d) or long-term (30d) classifiers per ticker.
- Fetch price history via `yfinance` and compute SMA/RSI/MACD/returns automatically.
- Pull top headlines via Yahoo + Google News RSS feeds (no API keys needed), score with FinBERT, and feed directly into predictions.
- View the exact news snippets powering the sentiment so you can validate the signal.
- Modern single-page UI with logging stream, gradients, and responsive layout.

### Setup
1. Create a virtual environment (recommended):
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
### Run the Web Server
```
uvicorn webapp.main:app --reload
```

Browse to http://127.0.0.1:8000 and use the interface to train or predict. Enter a ticker (e.g., `HDFCBANK.NS`), choose the horizon, tweak news lookback or queries, and view the generated reasoning directly in the app.

> **Disclaimer:** Model outputs are experimental signals and not financial advice.

### Model quality notes
- Training uses a holdout split for probability calibration (sigmoid) and auto-tunes BUY/HOLD/SELL thresholds to favor confident actions.
- If insufficient validation data is available, calibration falls back to default thresholds (BUY≥0.60, SELL≤0.40).

