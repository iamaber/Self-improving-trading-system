import torch
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime, timedelta

from model import StockLSTM, device

LOOK_BACK = 60
MODEL_PATH = "best_stock_model.pth"
SCALER_PATH = "scaler.pkl"

def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train the model first.")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}. Please ensure it was saved during training.")

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    # input_size will be 1 as the scaler was fitted on a single feature
    input_size = scaler.scale_.shape[0]

    model = StockLSTM(input_size=input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, scaler

def predict_future(model, scaler, historical_data_df, num_prediction_days):
    if historical_data_df.empty:
        return pd.DataFrame(), []

    # Select only the 'Close' column from historical data
    historical_data_df = historical_data_df[['Close']] # <--- CRUCIAL CHANGE: Select only 'Close'

    last_look_back_data = historical_data_df.tail(LOOK_BACK)
    scaled_last_look_back = scaler.transform(last_look_back_data)

    predictions = []
    current_input = torch.tensor(scaled_last_look_back, dtype=torch.float32).unsqueeze(0).to(device)

    for _ in range(num_prediction_days):
        with torch.no_grad():
            predicted_scaled_close = model(current_input).item()

        # To inverse transform, create a dummy array for a single feature
        dummy_row = np.array([predicted_scaled_close]) # <--- CRUCIAL CHANGE: dummy_row for single feature
        predicted_original_close = scaler.inverse_transform(dummy_row.reshape(1, -1))[0, 0] # <--- CRUCIAL CHANGE: index 0

        predictions.append(predicted_original_close)

        # Update the current_input for the next prediction
        # For a single feature, the new input is just the predicted value
        new_input_row = np.array([predicted_scaled_close]) # <--- CRUCIAL CHANGE: new_input_row for single feature

        current_input = torch.cat((current_input[:, 1:, :], torch.tensor(new_input_row, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)), dim=1)

    # Generate future dates
    last_date = historical_data_df.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_prediction_days)
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted Close': predictions})
    forecast_df.set_index('Date', inplace=True)
    return forecast_df


if __name__ == "__main__":
    from data_fetcher import fetch_stock_data
    
    # Load original data to act as "historical data" for prediction input
    historical_df = pd.read_csv("google_stock_prices_2015_2024.csv", parse_dates=['Date'], index_col='Date')
    # Get the last date from the original dataset
    last_original_date = historical_df.index.max()

    new_data_start = (last_original_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end_date_new_data = datetime.now().strftime("%Y-%m-%d") # Current date is 2025-07-22
    
    latest_fetched_data = fetch_stock_data("GOOGL", new_data_start, end_date_new_data)
    if latest_fetched_data is not None:
        # Concatenate (only 'Close' will be used in predict_future function)
        historical_df = pd.concat([historical_df, latest_fetched_data]).drop_duplicates().sort_index()

    try:
        model, scaler = load_model_and_scaler()
        print("Model and scaler loaded successfully.")

        forecast_7_days = predict_future(model, scaler, historical_df, 7)
        print("\n7-day Forecast:")
        print(forecast_7_days)

        forecast_15_days = predict_future(model, scaler, historical_df, 15)
        print("\n15-day Forecast:")
        print(forecast_15_days)

        forecast_30_days = predict_future(model, scaler, historical_df, 30)
        print("\n30-day Forecast:")
        print(forecast_30_days)

    except FileNotFoundError as e:
        print(e)
        print("Please run `python train.py` first to train/fine-tune the model and save the scaler.") 

