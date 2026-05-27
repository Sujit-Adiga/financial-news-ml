import pandas as pd
import yfinance as yf
from pathlib import Path

tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

saved_files = []

for ticker in tickers:
    print(f"Downloading data for {ticker}...")

    # Download historical stock data for the past 2 years with daily intervals
    data = yf.download(
        ticker,
        period = "2y", 
        interval = "1d"
    )

    if data is None or data.empty:
        print(f"No data downloaded for {ticker}. Skipping.")
        continue

    # Moves dates from index to a dedicated column
    data.reset_index(inplace=True)

    # Create the filename by removing the '.NS' suffix and saving it in the 'data/raw' directory
    filename = RAW_DATA_DIR / f"{ticker.replace('.NS', '')}_stock_data.csv"
    # Save the data to a CSV file without the index
    data.to_csv(filename, index=False)
    saved_files.append(filename)

    print(f"Saved: {filename}. \n")

if saved_files:
    print(f"Downloaded and saved {len(saved_files)} file(s) to {RAW_DATA_DIR}.")
else:
    print("No stock data was downloaded, so no files were saved.")
