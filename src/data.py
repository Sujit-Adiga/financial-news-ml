from pathlib import Path
import pandas as pd
import yfinance as yf


def download_ohlcv(ticker, start, end, output_path):
    """Download daily OHLCV data."""
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise RuntimeError("No market data was downloaded.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing market columns: {missing}")

    df[cols].to_csv(output_path, index=False)
    return df[cols]


def find_column(columns, candidates):
    normalized = {str(c).strip().lower(): c for c in columns}

    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    for c in columns:
        low = str(c).strip().lower()
        if any(candidate.lower() in low for candidate in candidates):
            return c

    return None


def prepare_news_csv(input_path, output_path):
    """
    Normalize common financial-news CSV schemas into:
      published_at, date, headline, ticker

    ticker is optional.
    """
    df = pd.read_csv(input_path)

    date_col = find_column(
        df.columns,
        ["published_at", "publish_time", "timestamp", "datetime",
         "date", "published", "publication_date"]
    )

    headline_col = find_column(
        df.columns,
        ["headline", "title", "news headline", "text", "summary"]
    )

    ticker_col = find_column(
        df.columns,
        ["ticker", "symbol", "stock", "tickers"]
    )

    if date_col is None or headline_col is None:
        raise ValueError(
            "Could not identify date and headline columns. "
            f"Columns found: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["published_at"] = pd.to_datetime(
        df[date_col], errors="coerce", utc=True
    )
    out["date"] = out["published_at"].dt.date
    out["headline"] = df[headline_col].astype(str)

    if ticker_col:
        out["ticker"] = df[ticker_col].astype(str)
    else:
        out["ticker"] = ""

    out = out.dropna(subset=["published_at"])
    out = out[out["headline"].str.len() > 0]
    out = out.drop_duplicates(subset=["published_at", "headline"])

    out.to_csv(output_path, index=False)
    return out
