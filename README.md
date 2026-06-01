# financial-news-ml
Financial News Sentiment + Stock Prediction

This project aims to predict short-term stock movement using historical stock prices and financial news sentiment.

## Project Structure

```text
data/
  raw/          # Raw stock data downloaded from yfinance
  processed/    # Cleaned and feature-engineered datasets

notebooks/
  01_stock_data_eda.ipynb
  02_stock_preprocessing.ipynb
  03_baseline_model.ipynb

src/
  download_stock_data.py
```


## Current Status

Collected 2 years of OHLCV stock data for selected NSE stocks using yfinance

Performed exploratory data analysis (EDA)

Engineered technical indicators:
- Daily_Return
- MA_7
- MA_14
- MA_30
- Volatility_30
- Volume_Change

Created next-day movement target variable

Trained baseline machine learning models:
- Logistic Regression
- Random Forest

## Features

Technical Indicators:
- Daily_Return
- MA_7
- MA_14
- MA_30
- Volatility_30
- Volume_Change

Target:
- 1 → Next day's close is higher
- 0 → Otherwise

## Baseline Results

| Model | Accuracy | F1 Score |
|---------|---------|---------|
| Logistic Regression | 0.47 | 0.40 |
| Random Forest | 0.49 | 0.48 |

### Key Observations

- Both models performed close to random guessing.
- Logistic Regression showed a strong bias toward predicting upward movement.
- Random Forest produced more balanced predictions.
- Technical indicators alone appear insufficient for reliable next-day stock movement prediction.

## Next Steps

- Collect financial news data
- Perform sentiment analysis on news headlines
- Merge sentiment features with stock data
- Train sentiment-enhanced prediction models
- Compare performance against baseline models

## Key Insight

The baseline experiments indicate that technical indicators alone provide limited predictive power for next-day stock movement. The next phase of the project focuses on incorporating external information sources such as financial news sentiment to improve forecasting performance.