import pandas as pd


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Creates technical analysis features from stock price data.

    Args:
        data: Price dataframe with columns including 'Close' and 'Volume'.
    """
    df = data.copy()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-9))))

    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["Return_1D"] = df["Close"].pct_change(1)
    df["Return_3D"] = df["Close"].pct_change(3)
    df["Return_5D"] = df["Close"].pct_change(5)

    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = 0.0

    return df

