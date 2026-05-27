# Daily Progress Log

## 2026-05-27

### Done
- Created project repo
- Downloaded stock data using yfinance
- Stored CSVs in data/raw/

### Issues Faced
- Data was getting saved inside src/ instead of data/raw/

### Learned
- src/ should contain code only
- raw datasets should go inside data/raw/

### Next
- Create EDA notebook
- Plot closing prices
- Calculate daily returns

## 2026-05-27

### Done
- Created initial stock EDA notebook
- Verified raw stock data files
- Plotted closing prices
- Calculated daily returns
- Started planning preprocessing pipeline

### Learned
- Raw OHLCV data needs feature engineering before ML
- Target variable must be created carefully to avoid data leakage

### Next
- Create `02_stock_preprocessing.ipynb`
- Combine all stock CSVs
- Add moving averages, volatility, and next-day target label