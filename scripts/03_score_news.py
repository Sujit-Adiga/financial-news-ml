import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR
from src.sentiment import FinBERTScorer
import pandas as pd


if __name__ == "__main__":
    input_path = PROCESSED_DIR / "news_prepared.csv"
    output_path = PROCESSED_DIR / "news_scored.csv"

    news = pd.read_csv(input_path)

    if len(news) < 1500:
        print(
            f"WARNING: only {len(news):,} articles found. "
            "The resume claim of 1,500+ articles would not be "
            "supported by this run."
        )

    scorer = FinBERTScorer(batch_size=16)

    sentiment = scorer.score(
        news["headline"].fillna("").tolist()
    )

    scored = pd.concat(
        [
            news.reset_index(drop=True),
            sentiment.reset_index(drop=True),
        ],
        axis=1,
    )

    scored.to_csv(output_path, index=False)

    print(f"Scored {len(scored):,} articles.")
    print(f"Saved to {output_path}")
