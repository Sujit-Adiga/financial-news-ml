import numpy as np
import pandas as pd


TECHNICAL_FEATURES = [
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "close_sma_5",
    "close_sma_10",
    "close_sma_20",
    "close_sma_50",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "volatility_30",
    "hl_range",
    "oc_change",
    "volume_change",
    "volume_ratio",
    "rsi_14",
    "macd",
    "macd_signal",
    "bb_position",
]

SENTIMENT_FEATURES = [
    "sentiment_mean",
    "sentiment_std",
    "sentiment_max",
    "sentiment_min",
    "news_count",
    "positive_mean",
    "negative_mean",
    "neutral_mean",
]


def aggregate_news_to_market_dates(
    news,
    market,
    cutoff_hour=16,
    cutoff_minute=0,
):
    """
    For each trading date t, aggregate news published on/before the
    configured cutoff. This is deliberately conservative: news after
    the cutoff is assigned to the next available trading date.
    """
    news = news.copy()
    market = market.copy()

    news["published_at"] = pd.to_datetime(
        news["published_at"], utc=True, errors="coerce"
    )
    market["Date"] = pd.to_datetime(market["Date"])

    news = news.dropna(subset=["published_at"])

    # Convert to UTC-naive date/time for deterministic handling.
    news["published_naive"] = (
        news["published_at"].dt.tz_convert("UTC").dt.tz_localize(None)
    )

    news["calendar_date"] = news["published_naive"].dt.normalize()
    cutoff = news["calendar_date"] + pd.Timedelta(
        hours=cutoff_hour,
        minutes=cutoff_minute,
    )

    # News after today's cutoff cannot affect today's prediction.
    news["eligible_date"] = np.where(
        news["published_naive"] <= cutoff,
        news["calendar_date"],
        news["calendar_date"] + pd.Timedelta(days=1),
    )
    news["eligible_date"] = pd.to_datetime(news["eligible_date"])

    # Map each news item to the next available trading day.
    trading_dates = pd.DataFrame({
        "Date": market["Date"].drop_duplicates().sort_values()
    })

    news = pd.merge_asof(
        news.sort_values("eligible_date"),
        trading_dates.sort_values("Date"),
        left_on="eligible_date",
        right_on="Date",
        direction="forward",
    )

    daily = (
        news.dropna(subset=["Date"])
        .groupby("Date")
        .agg(
            sentiment_mean=("sentiment", "mean"),
            sentiment_std=("sentiment", "std"),
            sentiment_max=("sentiment", "max"),
            sentiment_min=("sentiment", "min"),
            news_count=("sentiment", "count"),
            positive_mean=("positive", "mean"),
            negative_mean=("negative", "mean"),
            neutral_mean=("neutral", "mean"),
        )
        .reset_index()
    )

    daily["sentiment_std"] = daily["sentiment_std"].fillna(0)

    result = market.merge(daily, on="Date", how="left")

    for col in SENTIMENT_FEATURES:
        result[col] = result[col].fillna(0)

    return result


def build_model_dataset(
    market,
    news_scored,
    cutoff_hour=16,
    cutoff_minute=0,
):
    from .technical import add_technical_features, add_target

    market = add_technical_features(market)
    market = add_target(market)

    merged = aggregate_news_to_market_dates(
        news_scored,
        market,
        cutoff_hour=cutoff_hour,
        cutoff_minute=cutoff_minute,
    )

    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(
        subset=TECHNICAL_FEATURES + ["target", "next_day_return"]
    )

    return merged
