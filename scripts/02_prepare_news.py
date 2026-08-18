import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RAW_DIR, PROCESSED_DIR
from src.data import prepare_news_csv


if __name__ == "__main__":
    input_path = RAW_DIR / "news.csv"
    output_path = PROCESSED_DIR / "news_prepared.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Download a news dataset first."
        )

    df = prepare_news_csv(
        input_path,
        output_path,
    )

    print(f"Prepared {len(df):,} news articles.")
    print(f"Saved to {output_path}")
    print(df.head())
