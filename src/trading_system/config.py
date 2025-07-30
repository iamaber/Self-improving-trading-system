from pathlib import Path
import os

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Data files
GOOGLE_STOCK_DATA = RAW_DATA_DIR / "google_stock_prices_2015_2024.csv"

# Model files
MODEL_PATH = MODELS_DIR / "best_stock_model.pth"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# Model parameters
LOOK_BACK = 60
HIDDEN_SIZE = 64
NUM_LAYERS = 4
DROPOUT = 0.5
LEARNING_RATE = 0.001
EPOCHS = 10

# Stock settings
DEFAULT_TICKER = "GOOGL"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
