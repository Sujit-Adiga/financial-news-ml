# News-Enhanced Market Prediction

End-to-end next-day market-direction prediction using:
- 3 years of OHLCV data
- technical indicators
- FinBERT sentiment from financial news
- XGBoost classification
- walk-forward/time-aware validation
- transaction-cost-aware backtesting
- ROC-AUC, F1, return, Sharpe ratio and max drawdown

## Important

The project is designed to reproduce the *methodology* behind resume claims such as:

- 3 years of OHLCV data
- 1,500+ financial-news articles
- FinBERT sentiment
- XGBoost
- time-aware validation
- ROC-AUC / F1 improvement
- backtest return

The exact numbers (for example 71.4% ROC-AUC, 6.8% F1 improvement and 14.6% return) are NOT hard-coded. They must be produced by the actual experiment after fixing the ticker, dates, news data, timestamp policy and transaction costs.

## Recommended data setup

For a clean, reproducible first version, use:
- Market: NIFTY 50 (`^NSEI`) or S&P 500 (`^GSPC`)
- News: a dated financial-news dataset with at least 1,500 articles
- Sentiment: `ProsusAI/finbert`

A particularly convenient news source is the Kaggle "Financial News with Ticker-Level Sentiment" dataset because it contains 5,000+ articles, tickers and publication metadata. The pipeline below still runs FinBERT itself rather than using the dataset's existing sentiment labels.

## Repository layout

```text
news-enhanced-market-prediction/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── technical.py
│   ├── sentiment.py
│   ├── features.py
│   ├── model.py
│   ├── backtest.py
│   └── evaluate.py
├── scripts/
│   ├── 01_download_market.py
│   ├── 02_prepare_news.py
│   ├── 03_score_news.py
│   └── 04_run_experiment.py
├── .env.example
├── requirements.txt
└── README.md
```

## 1. Install

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

## 2. Download market data

```bash
python scripts/01_download_market.py
```

This downloads daily OHLCV data using yfinance.

## 3. Get news

Download a dated financial-news CSV from one of the sources listed at the end of this README and place it in:

```text
data/raw/news.csv
```

The preparation script tries to detect common column names automatically.

Run:

```bash
python scripts/02_prepare_news.py
```

It creates:

```text
data/processed/news_prepared.csv
```

Required logical fields are:

```text
date/time + headline/title/text
```

A timestamp is strongly preferred because it allows strict prevention of look-ahead leakage.

## 4. Score news with FinBERT

```bash
python scripts/03_score_news.py
```

The first run downloads `ProsusAI/finbert` automatically from Hugging Face.

Output:

```text
data/processed/news_scored.csv
```

## 5. Run the complete experiment

```bash
python scripts/04_run_experiment.py
```

The script:
1. creates technical indicators
2. creates the next-day direction target
3. aggregates only information available by the prediction cutoff
4. trains a technical-only XGBoost baseline
5. trains technical + FinBERT XGBoost
6. performs walk-forward evaluation
7. calculates F1 and ROC-AUC
8. runs a transaction-cost-aware long/cash backtest
9. reports return, Sharpe and max drawdown
10. saves predictions and metrics

## Leakage policy

The default prediction convention is:

> At the end of trading day `t`, predict whether the close on day `t+1` will be higher than the close on day `t`.

Therefore:
- OHLCV features are from day `t`
- technical indicators use data through `t`
- news is included only if its publication timestamp is before the configured cutoff
- target is based on day `t+1`

Do not merge news using future dates.

## Data sources

### Market data
yfinance:
https://github.com/ranaroussi/yfinance

Documentation:
https://ranaroussi.github.io/yfinance/

### FinBERT
Hugging Face:
https://huggingface.co/ProsusAI/finbert

Original implementation:
https://github.com/ProsusAI/finBERT

### Recommended news dataset
Kaggle: Financial News with Ticker-Level Sentiment
https://www.kaggle.com/datasets/rdolphin/financial-news-with-ticker-level-sentiment

It contains 5,000+ articles and publication/ticker metadata. The project ignores its supplied sentiment labels and independently runs FinBERT so that the resume claim is genuinely based on FinBERT.

### Alternative news datasets
S&P 500 with Financial News Headlines (2008–2024):
https://www.kaggle.com/datasets/dyutidasmahaptra/s-and-p-500-with-financial-news-headlines-20082024

Financial News Dataset (large historical corpus):
https://www.kaggle.com/datasets/yogeshchary/financial-news-dataset

Alpha Vantage News & Sentiment API:
https://www.alphavantage.co/documentation/

## Resume-number validation

Do NOT write:

"achieving 71.4% ROC-AUC and 14.6% backtest return"

until the experiment actually produces those numbers.

Keep a record of:
- ticker
- exact date range
- number of news articles
- train/validation/test windows
- prediction cutoff
- XGBoost hyperparameters
- transaction cost
- threshold
- benchmark return
- ROC-AUC
- F1 baseline
- F1 enhanced
- backtest return
- Sharpe
- maximum drawdown

This makes the project interview-defensible.
