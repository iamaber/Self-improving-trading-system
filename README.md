# Self Improving Trading System (LSTM-based)

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

This project is a web-based application for predicting stock prices using a Long Short-Term Memory (LSTM) neural network. It enables users to visualize historical stock data, evaluate model performance, forecast future prices, and analyze market trends through interactive charts. The model can be easily fine-tuned with new data to keep predictions up-to-date.

---

## 🚀 Features

- **Historical Data Visualization:** Interactive charts of past stock prices.
- **LSTM Model Training/Fine-tuning:** Train or fine-tune an LSTM model using historical and newly fetched data (trained on the 'Close' price).
- **Historical Prediction Performance:** Compare actual vs. predicted prices to assess model accuracy.
- **Error Metrics:** View RMSE, MAE, and a distribution plot of prediction errors.
- **Future Price Forecasting:** Forecast future stock prices for various horizons (e.g., 7, 15, 30, 60, 90, 120, 180 days).
- **Real-time Data Fetching:** Automatically fetches the latest stock data from Yahoo Finance.
- **Market Analysis Charts:**
    - Daily Trading Volume
    - Close Price with Moving Averages (50-day, 200-day)
    - Daily Returns Distribution
- **Interactive Streamlit Dashboard:** User-friendly interface built with Streamlit.

---

## 🛠️ Technologies Used

- Python 3.x
- PyTorch (LSTM model)
- Pandas (data manipulation)
- Scikit-learn (MinMaxScaler)
- yfinance (data fetching)
- Streamlit (web app)
- Plotly (visualizations)
- NumPy (numerical operations)

---

## 📂 Project Structure

```
self-improving-trading-system/
├── src/
│   └── trading_system/           # Main package
│       ├── __init__.py
│       ├── config.py             # Configuration settings
│       ├── data_fetcher.py       # Data fetching and preprocessing
│       ├── model.py              # LSTM model definition
│       ├── predict.py            # Prediction functionality
│       └── train.py              # Model training and fine-tuning
├── scripts/
│   └── app.py                    # Streamlit application
├── data/
│   └── raw/                      # Raw data files
│       └── google_stock_prices_2015_2024.csv
├── models/                       # Trained models and scalers
│   ├── best_stock_model.pth      # Pre-trained model weights
│   └── scaler.pkl                # Pre-fitted data scaler
├── notebooks/                    # Jupyter notebooks for analysis
│   └── LSTM_model_of_GOOGL_stock.ipynb
├── .gitignore                    # Git ignore rules
├── .python-version               # Python version specification
├── pyproject.toml                # Project configuration and dependencies
├── requirements.txt              # Dependencies (for pip)
├── uv.lock                       # Lock file for uv package manager
├── run.py                        # Quick start script
└── README.md                     # This file
```

### Key Components

- **`src/trading_system/`**: Core package containing all trading system logic
  - **`config.py`**: Configuration settings and parameters
  - **`data_fetcher.py`**: Data fetching from Yahoo Finance and preprocessing
  - **`model.py`**: LSTM neural network architecture
  - **`predict.py`**: Price prediction functionality
  - **`train.py`**: Model training and fine-tuning logic
- **`scripts/app.py`**: Streamlit web application for user interface
- **`data/raw/`**: Historical stock data storage
- **`models/`**: Trained models and preprocessing artifacts
- **`notebooks/`**: Jupyter notebooks for exploration and analysis
- **`run.py`**: Convenient script to launch the application

---

## ⚙️ Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/iamaber/Self-improving-trading-system.git
   cd Self-improving-trading-system
   ```

2. **Create a Virtual Environment (Recommended)**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**
   
   Using pip:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or using uv (recommended):
   ```bash
   uv sync
   ```
   
   Or install the package in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Data and Pre-trained Model Setup**
   - The repository includes pre-trained models in [`models/`](models/) and historical data in [`data/raw/`](data/raw/)
   - These are ready for use, so you can skip initial training if you want to use the provided model

5. **(Optional) Train/Fine-tune the LSTM Model**
   ```bash
   python -m src.trading_system.train
   ```
   This will:
   - Load historical data from [`data/raw/`](data/raw/)
   - Fetch latest data from Yahoo Finance
   - Train/fine-tune the model
   - Save updated model and scaler to [`models/`](models/)

6. **Run the Streamlit Application**
   
   Quick start:
   ```bash
   python run.py
   ```
   
   Or directly:
   ```bash
   streamlit run scripts/app.py
   ```

---

## 🚀 Usage

- **Quick Start:** Run `python run.py` to launch the application immediately
- **Select Forecast Horizon:** Use the sidebar slider to choose how many days into the future to predict (e.g., 7, 30, 60, 90 days).
- **Fine-tune Model:** Click "Fine-tune Model with Latest Data" in the sidebar to update the model with recent stock data.
- **Navigate Tabs:**
    - **Forecast & Performance:** View future price forecasts, historical performance (Actual vs. Predicted), and error metrics (RMSE, MAE).
    - **Market Analysis:** Explore charts like Trading Volume, Moving Averages, and Daily Returns Distribution.
    - **Raw Data:** Inspect the raw historical data and the full dataset used for analysis.

---

## 📈 Model Architecture

The system uses an LSTM (Long Short-Term Memory) neural network implemented in PyTorch:
- **Input Features:** Historical stock prices (Close price)
- **Architecture:** Multi-layer LSTM with configurable hidden dimensions
- **Output:** Predicted stock prices for specified forecast horizons
- **Training:** Uses historical data with sliding window approach
- **Preprocessing:** MinMaxScaler for data normalization

---

## 🔧 Configuration

Model parameters and settings can be adjusted in [`src/trading_system/config.py`](src/trading_system/config.py):
- Sequence length for LSTM input
- Hidden layer dimensions
- Number of LSTM layers
- Learning rate and training epochs
- Forecast horizons

---

## 📊 Data Sources

- **Historical Data:** Pre-loaded Google (GOOGL) stock data from 2015-2024
- **Real-time Data:** Yahoo Finance API via yfinance library
- **Automatic Updates:** System can fetch and incorporate latest market data

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

This application is for educational and research purposes only. Stock price predictions should not be used as the sole basis for investment decisions. Always consult with financial professionals and conduct your own research before making investment choices.
