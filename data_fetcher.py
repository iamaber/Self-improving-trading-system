import yfinance as yf
import pandas as pd
from datetime import datetime


def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            print(f"No data fetched for {ticker} from {start_date} to {end_date}")
            return None

        # Flatten MultiIndex columns if they exist
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [
                col[-1] if isinstance(col, tuple) else col for col in data.columns
            ]

        # Standardize column names (e.g., 'Adj Close' to 'Close')
        data.rename(columns={"Adj Close": "Close"}, inplace=True)

        # Filter to common stock columns, if they exist
        available_cols = [
            col
            for col in ["Open", "High", "Low", "Close", "Volume"]
            if col in data.columns
        ]
        if not available_cols:
            print(
                f"Warning: No standard stock columns found for {ticker} in range {start_date} to {end_date}."
            )
            return None

        data = data[available_cols]  # Select only the columns we care about
        data = data.dropna()

        if data.empty:
            print(
                f"No valid data after dropping NaNs for {ticker} from {start_date} to {end_date}"
            )
            return None

        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


if __name__ == "__main__":
    ticker_symbol = "GOOGL"
    start_date = "2025-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    latest_data = fetch_stock_data(ticker_symbol, start_date, end_date)
    if latest_data is not None:
        print(f"Fetched {len(latest_data)} new data points.")
        print(latest_data.head())
        print(f"Columns after fetcher: {latest_data.columns.tolist()}")
