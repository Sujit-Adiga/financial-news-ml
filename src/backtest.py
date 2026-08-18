import numpy as np
import pandas as pd


def run_backtest(
    predictions,
    threshold=0.50,
    transaction_cost=0.0005,
):
    df = predictions.copy()

    df["position"] = (
        df["probability"] >= threshold
    ).astype(int)

    # Position held for the next day's return.
    df["gross_return"] = (
        df["position"] * df["next_day_return"]
    )

    df["turnover"] = (
        df["position"].diff().abs().fillna(
            df["position"].abs()
        )
    )

    df["cost"] = df["turnover"] * transaction_cost
    df["strategy_return"] = (
        df["gross_return"] - df["cost"]
    )

    df["equity"] = (
        1.0 + df["strategy_return"]
    ).cumprod()

    total_return = df["equity"].iloc[-1] - 1.0

    daily = df["strategy_return"]
    sharpe = (
        daily.mean() / daily.std() * np.sqrt(252)
        if daily.std() > 0 else 0.0
    )

    running_max = df["equity"].cummax()
    drawdown = df["equity"] / running_max - 1
    max_drawdown = drawdown.min()

    # Buy-and-hold over the same test observations.
    benchmark = (
        (1 + df["next_day_return"].fillna(0)).cumprod()
    )
    benchmark_return = benchmark.iloc[-1] - 1.0

    metrics = {
        "strategy_return": float(total_return),
        "benchmark_return": float(benchmark_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "trades": int(df["turnover"].sum()),
    }

    return df, metrics
