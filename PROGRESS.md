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

## 2026-06-02

### Done
- Combined stock data from multiple NSE companies into a unified dataset
- Engineered technical indicators:
  - Daily_Return
  - MA_7
  - MA_14
  - MA_30
  - Volatility_30
  - Volume_Change
- Created next-day stock movement target variable
- Generated ML-ready dataset and saved it to `data/processed/stock_features.csv`
- Trained Logistic Regression baseline model
- Trained Random Forest baseline model
- Evaluated models using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix
- Compared baseline model performance
- Updated project README with baseline results and observations
- Analyzed CNBC, Reuters, and Guardian news datasets
- Investigated dataset schemas, missing values, and timestamp formats
- Standardized news dataset timestamps into a common format
- Combined all news sources into a unified news dataset
- Saved combined news data for sentiment analysis

### Issues Faced
- Rolling window features generated NaN values at the beginning of each stock time series
- Different news sources used different timestamp formats
- Logistic Regression was heavily biased toward predicting positive stock movement
- Baseline models performed close to random guessing

### Learned
- Missing values caused by rolling statistics should generally be removed rather than replaced with zero
- Time-series problems require chronological train-test splits to avoid data leakage
- Technical indicators alone provide limited predictive power for next-day stock movement
- A common schema is necessary before merging data from multiple news sources
- Financial news sentiment may provide additional predictive signals beyond technical indicators

### Key Findings
- Logistic Regression Accuracy: ~47%
- Random Forest Accuracy: ~49%
- Random Forest produced more balanced predictions than Logistic Regression
- Technical indicators alone were insufficient for reliable stock movement prediction
- News sentiment is the most promising next source of additional predictive signal

### Next
- Clean combined news dataset
- Set up FinBERT sentiment analysis pipeline
- Generate sentiment scores for a sample of headlines
- Validate sentiment outputs manually
- Create daily sentiment features
- Merge sentiment features with stock features
- Train and evaluate sentiment-enhanced models