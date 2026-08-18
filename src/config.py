from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

for directory in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Change these to your exact experiment.
TICKER = "^NSEI"
START_DATE = "2023-01-01"
END_DATE = "2026-01-01"

# Prediction:
# At the end of t, predict close(t+1) > close(t).
NEWS_CUTOFF_HOUR = 16
NEWS_CUTOFF_MINUTE = 0

RANDOM_STATE = 42
TRANSACTION_COST = 0.0005
INITIAL_CAPITAL = 100000.0

# Walk-forward windows.
MIN_TRAIN_SIZE = 500
VALID_SIZE = 60

# Trading threshold.
PROBABILITY_THRESHOLD = 0.50
