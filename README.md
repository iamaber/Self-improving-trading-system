# Stock Price Predictor (LSTM-based)

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
stock_predictor/
├── data_fetcher.py
├── model.py
├── predict.py
├── train.py
├── app.py
├── google_stock_prices_2015_2024.csv
├── best_stock_model.pth           # Pre-trained model weights
└── scaler.pkl                     # Pre-fitted data scaler
```

- **data_fetcher.py:** Fetches and standardizes historical stock data from Yahoo Finance.
- **model.py:** Defines the LSTM neural network (StockLSTM).
- **predict.py:** Loads the trained model and scaler, and makes predictions.
- **train.py:** Trains or fine-tunes the LSTM model and saves the model and scaler.
- **streamlit_app.py:** Main Streamlit dashboard integrating all components.
- **google_stock_prices_2015_2024.csv:** Initial dataset (GOOGL) for model training.
- **best_stock_model.pth:** Pre-trained LSTM model weights.
- **scaler.pkl:** Pre-fitted MinMaxScaler for preprocessing.

---

## ⚙️ Setup Instructions

1. **Clone the Repository**
     ```bash
     git clone <repository_url>
     cd stock_predictor
     ```

2. **Create a Virtual Environment (Recommended)**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate  # On Windows: .venv\Scripts\activate
     ```

3. **Install Dependencies**
     ```bash
     pip install -r requirements.txt
     ```
     If `requirements.txt` is missing, create one with:
     ```
     torch
     pandas
     scikit-learn
     yfinance
     streamlit
     plotly
     numpy
     ```

4. **Data and Pre-trained Model Setup**
     - The repository includes `google_stock_prices_2015_2024.csv`, `best_stock_model.pth`, and `scaler.pkl`. These are ready for use, so you can skip initial training if you want to use the provided model.

5. **(Optional) Train/Fine-tune the LSTM Model**
     - To train or fine-tune with the latest data:
         ```bash
         python train.py
         ```
     - This will:
         - Load `google_stock_prices_2015_2024.csv`
         - Fetch new GOOGL data from 2025-01-01 to present
         - Combine datasets and train/fine-tune the model
         - Save updated `best_stock_model.pth` and `scaler.pkl`

6. **Run the Streamlit Application**
     ```bash
     streamlit run streamlit_app.py
     ```
     - This opens the app in your web browser.

---

## 🚀 Usage

- **Select Forecast Horizon:** Use the sidebar slider to choose how many days into the future to predict (e.g., 7, 30, 60, 90 days).
- **Fine-tune Model:** Click "Fine-tune Model with Latest Data" in the sidebar to update the model with recent stock data.
- **Navigate Tabs:**
    - **Forecast & Performance:** View future price forecasts, historical performance (Actual vs. Predicted), and error metrics (RMSE, MAE).
    - **Market Analysis:** Explore charts like Trading Volume, Moving Averages, and Daily Returns Distribution.
    - **Raw Data:** Inspect the raw historical data and the full dataset used for analysis.

---
