import numpy as np
import pandas as pd


def add_technical_features(df):
    df = df.copy()

    close = df["Close"]
    volume = df["Volume"]

    df["return_1d"] = close.pct_change()
    df["return_3d"] = close.pct_change(3)
    df["return_5d"] = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)

    for window in [5, 10, 20, 50]:
        df[f"sma_{window}"] = close.rolling(window).mean()
        df[f"close_sma_{window}"] = close / df[f"sma_{window}"]

    df["volatility_5"] = df["return_1d"].rolling(5).std()
    df["volatility_10"] = df["return_1d"].rolling(10).std()
    df["volatility_20"] = df["return_1d"].rolling(20).std()
    df["volatility_30"] = df["return_1d"].rolling(30).std()

    df["hl_range"] = (df["High"] - df["Low"]) / close
    df["oc_change"] = (df["Close"] - df["Open"]) / df["Open"]

    df["volume_change"] = volume.pct_change()
    df["volume_ma_10"] = volume.rolling(10).mean()
    df["volume_ratio"] = volume / (df["volume_ma_10"] + 1e-12)

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger position
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    df["bb_position"] = (close - lower) / (upper - lower + 1e-12)

    return df


def add_target(df):
    df = df.copy()
    df["next_day_return"] = df["Close"].shift(-1) / df["Close"] - 1
    df["target"] = (df["next_day_return"] > 0).astype(int)
    return df
