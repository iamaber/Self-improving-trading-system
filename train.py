import os
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from data_fetcher import fetch_stock_data
from model import StockLSTM, device

# Constants
LOOK_BACK = 60  # Number of past days to consider for prediction
MODEL_PATH = "best_stock_model.pth"
SCALER_PATH = "scaler.pkl"
ORIGINAL_DATA_PATH = "google_stock_prices_2015_2024.csv"  # Your provided dataset


def create_sequences(data, look_back):
    """
    Creates sequences for LSTM input.
    'data' here will only contain the 'Close' price.
    """
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(
            data[i : (i + look_back), :]
        )  # Still slice all columns (which is just 'Close')
        y.append(
            data[i + look_back, 0]
        )  # Predict 'Close' price (now at index 0 because only one column)
    return np.array(X), np.array(y)


def train_model(model, train_loader, criterion, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")


def prepare_data_and_train(new_data_df=None):
    # Load original data - ONLY SELECT 'Close'
    original_df = pd.read_csv(
        ORIGINAL_DATA_PATH, parse_dates=["Date"], index_col="Date"
    )
    original_df = original_df[["Close"]]  # <--- CRUCIAL CHANGE: Select only 'Close'

    # If new data is provided, combine it with the original data
    if new_data_df is not None:
        # Fetch only 'Close' from new data as well
        new_data_df = new_data_df[
            ["Close"]
        ]  # <--- CRUCIAL CHANGE: Select only 'Close' from new data
        combined_df = (
            pd.concat([original_df, new_data_df]).drop_duplicates().sort_index()
        )
    else:
        combined_df = original_df

    combined_df.columns = combined_df.columns.astype(
        str
    )  # Ensure column name is string

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(
        combined_df
    )  # This scaler will now be fitted on a single feature

    # Save the scaler
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # Create sequences
    X, y = create_sequences(scaled_data, LOOK_BACK)

    # Convert to PyTorch tensors and move to device
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Model, Loss, Optimizer
    input_size = scaled_data.shape[1]  # This will correctly be 1
    # Instantiate StockLSTM with the derived input_size (which is 1)
    model = StockLSTM(
        input_size=input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2
    ).to(device)

    # Load pre-trained model if it exists
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model loaded successfully.")
    else:
        print("No existing model found. Training a new model.")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Starting model training/fine-tuning...")
    train_model(model, train_loader, criterion, optimizer, epochs=10)

    # Save the updated model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    ticker_symbol = "GOOGL"
    # To correctly fetch new data from the *last date of your original dataset*,
    # you need to find the max date from your original CSV first.
    original_df_temp = pd.read_csv(ORIGINAL_DATA_PATH, parse_dates=["Date"])
    last_original_date = original_df_temp["Date"].max()

    start_date_new_data = (last_original_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end_date_new_data = datetime.now().strftime(
        "%Y-%m-%d"
    )  # Current date is 2025-07-22

    # Make sure fetch_stock_data still fetches all columns, then we select 'Close' inside prepare_data_and_train
    new_data = fetch_stock_data(ticker_symbol, start_date_new_data, end_date_new_data)

    if new_data is not None and not new_data.empty:
        prepare_data_and_train(new_data_df=new_data)
    else:
        print("No new data to fine-tune. Model not updated.")
