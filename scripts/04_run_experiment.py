import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import (
    RAW_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    NEWS_CUTOFF_HOUR,
    NEWS_CUTOFF_MINUTE,
    MIN_TRAIN_SIZE,
    VALID_SIZE,
    RANDOM_STATE,
    TRANSACTION_COST,
    PROBABILITY_THRESHOLD,
)
from src.features import (
    TECHNICAL_FEATURES,
    SENTIMENT_FEATURES,
    build_model_dataset,
)
from src.model import walk_forward_predict
from src.backtest import run_backtest
from src.evaluate import (
    compare_baseline_and_enhanced,
    save_json,
)


if __name__ == "__main__":
    market = pd.read_csv(RAW_DIR / "ohlcv.csv")
    market["Date"] = pd.to_datetime(market["Date"])

    news = pd.read_csv(PROCESSED_DIR / "news_scored.csv")

    df = build_model_dataset(
        market,
        news,
        cutoff_hour=NEWS_CUTOFF_HOUR,
        cutoff_minute=NEWS_CUTOFF_MINUTE,
    )

    print(f"Model rows: {len(df):,}")
    print(f"News articles: {len(news):,}")

    # Technical-only baseline.
    baseline_pred, baseline_metrics = walk_forward_predict(
        df,
        TECHNICAL_FEATURES,
        min_train_size=MIN_TRAIN_SIZE,
        test_size=VALID_SIZE,
        random_state=RANDOM_STATE,
    )

    # Technical + FinBERT.
    enhanced_pred, enhanced_metrics = walk_forward_predict(
        df,
        TECHNICAL_FEATURES + SENTIMENT_FEATURES,
        min_train_size=MIN_TRAIN_SIZE,
        test_size=VALID_SIZE,
        random_state=RANDOM_STATE,
    )

    comparison = compare_baseline_and_enhanced(
        baseline_pred,
        enhanced_pred,
    )

    backtest_df, backtest_metrics = run_backtest(
        enhanced_pred,
        threshold=PROBABILITY_THRESHOLD,
        transaction_cost=TRANSACTION_COST,
    )

    metrics = {
        "n_market_rows": len(df),
        "n_news_articles": len(news),
        "baseline": baseline_metrics,
        "enhanced": enhanced_metrics,
        "comparison": comparison,
        "backtest": backtest_metrics,
        "config": {
            "news_cutoff_hour": NEWS_CUTOFF_HOUR,
            "news_cutoff_minute": NEWS_CUTOFF_MINUTE,
            "min_train_size": MIN_TRAIN_SIZE,
            "walk_forward_test_size": VALID_SIZE,
            "transaction_cost": TRANSACTION_COST,
            "probability_threshold": PROBABILITY_THRESHOLD,
        },
    }

    baseline_pred.to_csv(
        REPORTS_DIR / "baseline_predictions.csv",
        index=False,
    )
    enhanced_pred.to_csv(
        REPORTS_DIR / "enhanced_predictions.csv",
        index=False,
    )
    backtest_df.to_csv(
        REPORTS_DIR / "backtest.csv",
        index=False,
    )
    save_json(
        metrics,
        REPORTS_DIR / "metrics.json",
    )

    print("\n=== RESULTS ===")
    print(f"Baseline F1:      {comparison['baseline_f1']:.4f}")
    print(f"Enhanced F1:      {comparison['enhanced_f1']:.4f}")
    print(
        "F1 improvement:   "
        f"{comparison['f1_relative_improvement_pct']:.2f}%"
    )
    print(
        f"Baseline ROC-AUC: {comparison['baseline_roc_auc']:.4f}"
    )
    print(
        f"Enhanced ROC-AUC: {comparison['enhanced_roc_auc']:.4f}"
    )
    print(
        f"Backtest return:  "
        f"{backtest_metrics['strategy_return']:.2%}"
    )
    print(
        f"Benchmark return: "
        f"{backtest_metrics['benchmark_return']:.2%}"
    )
    print(
        f"Sharpe:           "
        f"{backtest_metrics['sharpe']:.2f}"
    )
    print(
        f"Max drawdown:     "
        f"{backtest_metrics['max_drawdown']:.2%}"
    )
