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

src/
  download_stock_data.py

## Current Status

- Collected 2 years of OHLCV stock data for selected NSE stocks using yfinance
- Stored raw data in `data/raw/`
- Completed initial EDA on stock price trends and daily returns
- Started preprocessing pipeline for feature engineering and target creation

## Next Steps

- Combine all stock CSVs into one dataset
- Add technical indicators such as moving averages and volatility
- Create next-day movement target variable
- Save processed dataset to `data/processed/stock_features.csv`