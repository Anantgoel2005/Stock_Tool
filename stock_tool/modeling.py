import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from .config import MODEL_FEATURES, MODELS_DIR, MODEL_MAX_AGE_DAYS, TRAINING_YEARS
from .features import create_features
from .news import fetch_news_articles
from .sentiment import batch_sentiment_scores


def _format_rupee(value: float) -> str:
    return f"₹{value:,.2f}"


def _safe_symbol_name(symbol: str) -> str:
    return symbol.replace(".NS", "").replace(".", "_")


def _build_model_path(symbol: str, investment_horizon: str) -> Path:
    safe_symbol_name = _safe_symbol_name(symbol)
    return MODELS_DIR / f"model_{safe_symbol_name}_{investment_horizon}.joblib"


def _select_thresholds(probabilities: pd.Series, labels: pd.Series) -> Tuple[float, float]:
    """
    Choose BUY/SELL thresholds that favor confident predictions.
    Score = precision_on_actions * coverage_on_actions.
    """
    buy_candidates = [round(x, 2) for x in list(np.linspace(0.55, 0.75, 11))]
    sell_candidates = [round(x, 2) for x in list(np.linspace(0.25, 0.45, 11))]

    best_score = -1.0
    best_buy, best_sell = 0.6, 0.4

    for t_buy in buy_candidates:
        for t_sell in sell_candidates:
            actions = (probabilities >= t_buy) | (probabilities <= t_sell)
            if actions.sum() == 0:
                continue

            preds = probabilities.copy()
            preds.loc[probabilities >= t_buy] = 1
            preds.loc[probabilities <= t_sell] = 0
            correct = (preds[actions] == labels[actions]).sum()
            precision = correct / actions.sum()
            coverage = actions.mean()
            score = precision * coverage
            if score > best_score:
                best_score = score
                best_buy, best_sell = t_buy, t_sell

    return best_buy, best_sell


def train_and_save_model(
    symbol: str,
    model_filename: Optional[Union[str, os.PathLike]] = None,
    investment_horizon: str = "short_term",
) -> bool:
    logging.info(
        "Training new model for %s (%s horizon)", symbol, investment_horizon
    )
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=TRAINING_YEARS * 365)
        logging.info(
            "Fetching %s years of training data for %s...", TRAINING_YEARS, symbol
        )
        data = yf.download(
            symbol,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
        if data.empty:
            logging.error("Could not fetch training data for %s.", symbol)
            return False

        data["sentiment_score"] = 0.0
        data = create_features(data)

        if investment_horizon == "short_term":
            data["Target"] = (data["Close"].shift(-5) > data["Close"]).astype(int)
            logging.info("Target set for short-term (5-day prediction).")
        elif investment_horizon == "long_term":
            data["Target"] = (data["Close"].shift(-30) > data["Close"]).astype(int)
            logging.info("Target set for long-term (30-day prediction).")
        else:
            logging.warning(
                "Invalid investment horizon: %s. Defaulting to short-term.",
                investment_horizon,
            )
            investment_horizon = "short_term"
            data["Target"] = (data["Close"].shift(-5) > data["Close"]).astype(int)

        data.dropna(inplace=True)
        if data.empty:
            logging.error(
                "Not enough data to train after feature creation and target definition."
            )
            return False

        X = data[MODEL_FEATURES]
        y = data["Target"]
        split_index = int(len(X) * 0.8)
        X_train, y_train = X[:split_index], y[:split_index]
        X_val, y_val = X[split_index:], y[split_index:]

        if len(X_train) == 0:
            logging.error("Not enough data for a training set.")
            return False

        logging.info("Training XGBoost model on %s samples...", len(X_train))
        model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)

        # Calibrate probabilities with internal CV (sklearn >=1.4 disallows cv="prefit").
        if y_train.nunique() < 2:
            logging.warning(
                "Skipping calibration/threshold tuning (training labels lack class variety)."
            )
            calibrator = model
            buy_threshold, sell_threshold = 0.6, 0.4
        else:
            calibrator = CalibratedClassifierCV(model, cv=3, method="sigmoid")
            calibrator.fit(X_train, y_train)

            if len(X_val) == 0:
                logging.warning(
                    "No validation split available; using default thresholds."
                )
                buy_threshold, sell_threshold = 0.6, 0.4
            else:
                val_probs = pd.Series(
                    calibrator.predict_proba(X_val)[:, 1], index=y_val.index
                )
                buy_threshold, sell_threshold = _select_thresholds(val_probs, y_val)

        model_path = Path(model_filename) if model_filename else _build_model_path(
            symbol, investment_horizon
        )
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        bundle: Dict[str, Union[float, str, CalibratedClassifierCV]] = {
            "model": calibrator,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
            "trained_at": datetime.now().isoformat(),
        }
        joblib.dump(bundle, model_path)
        logging.info(
            "Model for %s (%s horizon) saved to %s | thresholds BUY>=%.2f SELL<=%.2f",
            symbol,
            investment_horizon,
            model_path,
            buy_threshold,
            sell_threshold,
        )
        return True
    except Exception as exc:
        logging.exception("Error during model training for %s: %s", symbol, exc)
        return False


def get_live_prediction_with_reasoning(
    symbol: str,
    news_query: str = "",
    investment_horizon: str = "short_term",
    days_back_for_news: int = 1,
    return_articles: bool = False,
) -> Union[str, Tuple[str, List[str]]]:
    model_path = _build_model_path(symbol, investment_horizon)
    needs_retrain = False

    if not model_path.exists():
        needs_retrain = True
        logging.info(
            "No model found for '%s' (%s horizon). Training new model...",
            symbol,
            investment_horizon,
        )
    else:
        try:
            mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
            age_days = (datetime.now() - mtime).days
            if age_days >= MODEL_MAX_AGE_DAYS:
                needs_retrain = True
                logging.info(
                    "Model for '%s' (%s) is %s days old; retraining.",
                    symbol,
                    investment_horizon,
                    age_days,
                )
        except Exception as exc:
            needs_retrain = True
            logging.warning(
                "Could not read model age for %s (%s): %s. Retraining.",
                symbol,
                investment_horizon,
                exc,
            )

    if needs_retrain:
        success = train_and_save_model(symbol, model_path, investment_horizon)
        if not success:
            return f"Failed to train a new model for '{symbol}' ({investment_horizon})."

    try:
        loaded = joblib.load(model_path)
        if isinstance(loaded, dict):
            model = loaded.get("model")
            buy_threshold = float(loaded.get("buy_threshold", 0.6))
            sell_threshold = float(loaded.get("sell_threshold", 0.4))
            trained_at = loaded.get("trained_at", "unknown")
        else:
            # Backward compatibility: model file is just the estimator.
            model = loaded
            buy_threshold, sell_threshold = 0.6, 0.4
            trained_at = "legacy"
        logging.info(
            "Loaded model bundle: %s (trained %s, BUY>=%.2f SELL<=%.2f)",
            model_path,
            trained_at,
            buy_threshold,
            sell_threshold,
        )
    except Exception as exc:
        logging.exception("Error loading model: %s", exc)
        return f"Error loading model: {exc}"

    start_date = (datetime.now() - timedelta(days=int(365 * 1.5))).strftime("%Y-%m-%d")
    logging.info("Fetching data for %s starting %s", symbol, start_date)
    try:
        data = yf.download(symbol, start=start_date, auto_adjust=True, progress=False)
        if data.empty:
            return "Failed to fetch price data."
    except Exception as exc:
        logging.exception("Error fetching data: %s", exc)
        return f"Error fetching data: {exc}"

    data = create_features(data)

    query_for_news = news_query or symbol
    news_articles = fetch_news_articles(
        symbol,
        query=query_for_news,
        days_back=days_back_for_news,
    )

    if news_articles:
        # Cap number of articles for faster sentiment while keeping variety.
        if len(news_articles) > 25:
            logging.info(
                "Truncating news articles from %s to 25 for sentiment scoring.",
                len(news_articles),
            )
            news_for_sentiment = news_articles[:25]
        else:
            news_for_sentiment = news_articles

        sentiment_scores = batch_sentiment_scores(news_for_sentiment)
        avg_sentiment = (
            sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
        )
        if not data.empty:
            data.at[data.index[-1], "sentiment_score"] = avg_sentiment
            logging.info(
                "Applied FinBERT sentiment score %.4f to latest datapoint.",
                avg_sentiment,
            )
    else:
        logging.info("No news articles fetched; sentiment score remains 0.")

    data.dropna(inplace=True)
    if data.empty:
        return "Not enough data to create features after dropping NaNs."

    latest_features = data[MODEL_FEATURES].iloc[-1:]

    try:
        probability = model.predict_proba(latest_features)[0]
        prob_up = float(probability[1])
        # Three-way decision from calibrated probability bands.
        if prob_up >= buy_threshold:
            signal = "BUY"
        elif prob_up <= sell_threshold:
            signal = "SELL"
        else:
            signal = "HOLD"
        if signal == "BUY":
            confidence = f"{prob_up:.0%}"
        elif signal == "SELL":
            confidence = f"{(1 - prob_up):.0%}"
        else:
            hold_conf = 1 - abs(prob_up - 0.5) * 2  # closeness to neutral
            confidence = f"{hold_conf:.0%}"
    except Exception as exc:
        logging.exception("Error during model prediction: %s", exc)
        return f"Error during model prediction: {exc}"

    reasoning = []
    try:
        sma_50 = latest_features["SMA_50"].item()
        sma_200 = latest_features["SMA_200"].item()
        sma_50_str = _format_rupee(sma_50)
        sma_200_str = _format_rupee(sma_200)
        if sma_50 > sma_200:
            reasoning.append(
                f"POSITIVE TREND: 50-day SMA ({sma_50_str}) above 200-day SMA ({sma_200_str})."
            )
        else:
            reasoning.append(
                f"NEGATIVE TREND: 50-day SMA ({sma_50_str}) below 200-day SMA ({sma_200_str})."
            )

        rsi = latest_features["RSI"].item()
        if rsi > 70:
            reasoning.append(f"MOMENTUM WARNING: Overbought RSI ({rsi:.2f}).")
        elif rsi < 30:
            reasoning.append(f"MOMENTUM ALERT: Oversold RSI ({rsi:.2f}).")
        else:
            reasoning.append(f"MOMENTUM NEUTRAL: RSI is {rsi:.2f}.")

        macd = latest_features["MACD"].item()
        signal_line = latest_features["Signal_Line"].item()
        if macd > signal_line:
            reasoning.append("MOMENTUM POSITIVE: MACD above signal line.")
        else:
            reasoning.append("MOMENTUM NEGATIVE: MACD below signal line.")

        return_5d = latest_features["Return_5D"].item()
        if return_5d > 0.02:
            reasoning.append(f"SHORT-TERM STRENGTH: 5-day return {return_5d:.1%}.")
        elif return_5d < -0.02:
            reasoning.append(f"SHORT-TERM WEAKNESS: 5-day return {return_5d:.1%}.")
        else:
            reasoning.append(f"SHORT-TERM NEUTRAL: 5-day return {return_5d:.1%}.")

        sentiment = latest_features["sentiment_score"].item()
        if sentiment > 0.05:
            reasoning.append(
                f"SENTIMENT POSITIVE: Average news sentiment {sentiment:.2f}."
            )
        elif sentiment < -0.05:
            reasoning.append(
                f"SENTIMENT NEGATIVE: Average news sentiment {sentiment:.2f}."
            )
        else:
            reasoning.append(
                f"SENTIMENT NEUTRAL: Average news sentiment {sentiment:.2f}."
            )
    except Exception as exc:
        logging.exception("Error generating reasoning: %s", exc)
        reasoning.append(f"Error generating reasoning: {exc}")

    output = (
        f"--- Prediction for {symbol} ---\n"
        f"SIGNAL:        {signal}\n"
        f"CONFIDENCE:    {confidence}\n"
        f"REASONING:\n  - " + "\n  - ".join(reasoning)
    )
    if return_articles:
        return output, news_articles
    return output

