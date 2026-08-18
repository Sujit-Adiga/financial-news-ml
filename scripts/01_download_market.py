import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import TICKER, START_DATE, END_DATE, RAW_DIR
from src.data import download_ohlcv


if __name__ == "__main__":
    output = RAW_DIR / "ohlcv.csv"

    df = download_ohlcv(
        TICKER,
        START_DATE,
        END_DATE,
        output,
    )

    print(f"Saved {len(df):,} market rows to {output}")
    print(df.head())
