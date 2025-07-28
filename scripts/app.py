import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error  # For error metrics

from data_fetcher import fetch_stock_data
from predict import load_model_and_scaler, predict_future
from train import prepare_data_and_train
from model import device

# --- Configuration ---
TICKER_SYMBOL = "GOOGL"
ORIGINAL_DATA_PATH = "google_stock_prices_2015_2024.csv"
MODEL_PATH = "best_stock_model.pth"
SCALER_PATH = "scaler.pkl"
LOOK_BACK = 60  # Needs to be consistent with model training

st.set_page_config(layout="wide", page_title="Stock Price Predictor")


@st.cache_resource
def get_model_and_scaler():
    """Caches the model and scaler to avoid reloading on every rerun."""
    try:
        model, scaler = load_model_and_scaler()
        model.to(device)  # Ensure model is on the correct device after loading
        return model, scaler
    except FileNotFoundError as e:
        st.error(
            f"Error loading model or scaler: {e}. Please ensure the model has been trained by running `python train.py`."
        )
        return None, None


@st.cache_data(ttl=timedelta(hours=1))  # Cache data for 1 hour
def get_historical_data():
    """Loads original data and fetches new data, then combines.
    Returns only the 'Close' column as the model is trained on a single feature."""
    original_df = pd.read_csv(
        ORIGINAL_DATA_PATH, parse_dates=["Date"], index_col="Date"
    )
    original_df = original_df[["Close"]]

    last_original_date = original_df.index.max()
    start_date_new = (last_original_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end_date_new = datetime.now().strftime("%Y-%m-%d")

    new_data = fetch_stock_data(TICKER_SYMBOL, start_date_new, end_date_new)

    if new_data is not None and not new_data.empty:
        new_data = new_data[["Close"]]  # Ensure new_data also only has 'Close'
        combined_df = pd.concat([original_df, new_data]).drop_duplicates().sort_index()
    else:
        combined_df = original_df

    return combined_df


@st.cache_data(ttl=timedelta(hours=1))
def get_historical_predictions(
    _model, _scaler, historical_df, look_back
):  # Added underscore to 'scaler'
    """
    Generates predictions for the historical data itself to show model fit.
    """
    if historical_df.empty or len(historical_df) < look_back:
        return pd.DataFrame()

    scaled_historical_data = _scaler.transform(
        historical_df[["Close"]]
    )  # Use _scaler here

    historical_predictions = []
    actual_values = []
    dates_for_plot = []

    for i in range(len(scaled_historical_data) - look_back):
        input_seq = scaled_historical_data[i : (i + look_back)]
        input_tensor = (
            torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
        )

        with torch.no_grad():
            predicted_scaled_close = _model(input_tensor).item()

        dummy_row = np.array([predicted_scaled_close])
        predicted_original_close = _scaler.inverse_transform(dummy_row.reshape(1, -1))[
            0, 0
        ]  # Use _scaler here

        historical_predictions.append(predicted_original_close)
        actual_values.append(historical_df["Close"].iloc[i + look_back])
        dates_for_plot.append(historical_df.index[i + look_back])

    results_df = pd.DataFrame(
        {
            "Date": dates_for_plot,
            "Actual Close": actual_values,
            "Predicted Close": historical_predictions,
        }
    ).set_index("Date")

    return results_df


# Streamlit App
st.title(f"Stock Price Prediction & Analysis for {TICKER_SYMBOL}")

# Sidebar for controls
st.sidebar.header("Settings")
forecast_horizon = st.sidebar.select_slider(
    "Select Forecast Horizon (days)", options=[7, 15, 30, 60, 90, 180, 360], value=7
)

# On-demand fine-tuning button (for demonstration)
if st.sidebar.button("Fine-tune Model with Latest Data"):
    with st.spinner("Fine-tuning model... This may take a moment."):
        current_date_data = fetch_stock_data(
            TICKER_SYMBOL, "2025-01-01", datetime.now().strftime("%Y-%m-%d")
        )
        if current_date_data is not None and not current_date_data.empty:
            prepare_data_and_train(new_data_df=current_date_data)
            st.success("Model fine-tuned and saved successfully!")
            st.cache_resource.clear()  # Clear cache to reload new model and scaler
            st.cache_data.clear()  # Clear data cache to re-fetch historical data and predictions
        else:
            st.warning("No new data to fine-tune. Model not updated.")

model, scaler = get_model_and_scaler()
historical_df = get_historical_data()

if model is not None and scaler is not None and not historical_df.empty:
    tab1, tab2, tab3 = st.tabs(
        ["Forecast & Performance", "Market Analysis", "Raw Data"]
    )

    with tab1:
        st.header("Forecast & Model Performance")

        # Historical Performance (Predicted vs. Actual)
        st.subheader("Historical Performance (Predicted vs. Actual)")
        if len(historical_df) >= LOOK_BACK:
            historical_preds_df = get_historical_predictions(
                _model=model,
                _scaler=scaler,
                historical_df=historical_df,
                look_back=LOOK_BACK,
            )  # Changed scaler to _scaler in call
            if not historical_preds_df.empty:
                fig_hist_perf = go.Figure()
                fig_hist_perf.add_trace(
                    go.Scatter(
                        x=historical_preds_df.index,
                        y=historical_preds_df["Actual Close"],
                        mode="lines",
                        name="Actual Close",
                        line=dict(color="blue"),
                    )
                )
                fig_hist_perf.add_trace(
                    go.Scatter(
                        x=historical_preds_df.index,
                        y=historical_preds_df["Predicted Close"],
                        mode="lines",
                        name="Predicted Close (Historical)",
                        line=dict(color="red", dash="dot"),
                    )
                )
                fig_hist_perf.update_layout(
                    title=f"{TICKER_SYMBOL} Historical Predicted vs. Actual Close Price",
                    xaxis_title="Date",
                    yaxis_title="Close Price (USD)",
                    hovermode="x unified",
                    template="plotly_white",
                )
                st.plotly_chart(fig_hist_perf, use_container_width=True)

                # --- Error Metrics ---
                st.subheader("Model Error Metrics (Historical Predictions)")
                col1, col2 = st.columns(2)
                rmse = np.sqrt(
                    mean_squared_error(
                        historical_preds_df["Actual Close"],
                        historical_preds_df["Predicted Close"],
                    )
                )
                mae = mean_absolute_error(
                    historical_preds_df["Actual Close"],
                    historical_preds_df["Predicted Close"],
                )

                col1.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")
                col2.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")

                # --- Prediction Error Distribution ---
                st.subheader("Prediction Error Distribution")
                historical_preds_df["Error"] = (
                    historical_preds_df["Actual Close"]
                    - historical_preds_df["Predicted Close"]
                )
                fig_error_dist = go.Figure(
                    data=[
                        go.Histogram(x=historical_preds_df["Error"].dropna(), nbinsx=50)
                    ]
                )
                fig_error_dist.update_layout(
                    title="Distribution of Prediction Errors (Residuals)",
                    xaxis_title="Error (Actual - Predicted)",
                    yaxis_title="Frequency",
                    template="plotly_white",
                )
                st.plotly_chart(fig_error_dist, use_container_width=True)

            else:
                st.info(
                    f"Not enough historical data ({len(historical_df)} days) to generate historical predictions (need at least {LOOK_BACK} days)."
                )
        else:
            st.info(
                f"Not enough historical data ({len(historical_df)} days) to generate historical predictions (need at least {LOOK_BACK} days)."
            )

        # Latest Forecast
        st.subheader(f"Latest {forecast_horizon}-Day Forecast")
        if len(historical_df) >= LOOK_BACK:
            try:
                forecast_df = predict_future(
                    model, scaler, historical_df, forecast_horizon
                )

                fig_forecast = go.Figure()

                # Plot actual historical prices
                fig_forecast.add_trace(
                    go.Scatter(
                        x=historical_df.index,
                        y=historical_df["Close"],
                        mode="lines",
                        name="Actual Close",
                        line=dict(color="blue"),
                    )
                )

                # Plot predicted prices
                if not forecast_df.empty:
                    fig_forecast.add_trace(
                        go.Scatter(
                            x=forecast_df.index,
                            y=forecast_df["Predicted Close"],
                            mode="lines",
                            name=f"{forecast_horizon}-Day Predicted Close",
                            line=dict(color="red", dash="dot"),
                        )
                    )

                    # Add a marker for the last actual close price
                    last_actual_date = historical_df.index[-1]
                    last_actual_close = historical_df["Close"].iloc[-1]
                    fig_forecast.add_trace(
                        go.Scatter(
                            x=[last_actual_date],
                            y=[last_actual_close],
                            mode="markers",
                            name="Last Actual Close",
                            marker=dict(size=10, color="purple", symbol="circle"),
                        )
                    )

                fig_forecast.update_layout(
                    title=f"{TICKER_SYMBOL} Stock Price: Historical vs. Forecast",
                    xaxis_title="Date",
                    yaxis_title="Close Price (USD)",
                    hovermode="x unified",
                    template="plotly_white",
                )
                st.plotly_chart(fig_forecast, use_container_width=True)

                st.subheader("Forecast Details")
                st.dataframe(forecast_df)

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.warning(
                    "Please ensure you have enough historical data (at least 60 days) and the model is trained."
                )
        else:
            st.warning(
                f"Not enough historical data available (need at least {LOOK_BACK} days) to make future predictions."
            )

    with tab2:
        st.header("Market Analysis Charts")

        @st.cache_data(ttl=timedelta(hours=1))
        def get_full_historical_data_for_analysis():
            full_original_df = pd.read_csv(
                ORIGINAL_DATA_PATH, parse_dates=["Date"], index_col="Date"
            )
            # Ensure these columns exist, data_fetcher should handle it
            full_original_df = full_original_df[
                ["Open", "High", "Low", "Close", "Volume"]
            ]

            last_original_date = full_original_df.index.max()
            start_date_new = (last_original_date + timedelta(days=1)).strftime(
                "%Y-%m-%d"
            )
            end_date_new = datetime.now().strftime("%Y-%m-%d")
            full_new_data = fetch_stock_data(
                TICKER_SYMBOL, start_date_new, end_date_new
            )

            if full_new_data is not None and not full_new_data.empty:
                full_combined_df = (
                    pd.concat([full_original_df, full_new_data])
                    .drop_duplicates()
                    .sort_index()
                )
            else:
                full_combined_df = full_original_df
            return full_combined_df

        full_historical_df_for_analysis = get_full_historical_data_for_analysis()

        if not full_historical_df_for_analysis.empty:
            # --- Trading Volume Chart ---
            st.subheader("Trading Volume Over Time")
            fig_volume = go.Figure(
                data=[
                    go.Bar(
                        x=full_historical_df_for_analysis.index,
                        y=full_historical_df_for_analysis["Volume"],
                    )
                ]
            )
            fig_volume.update_layout(
                title="Daily Trading Volume",
                xaxis_title="Date",
                yaxis_title="Volume",
                template="plotly_white",
            )
            st.plotly_chart(fig_volume, use_container_width=True)

            # --- Moving Averages Chart ---
            st.subheader("Close Price with Moving Averages")
            ma_periods = [50, 200]  # Define moving average periods
            for period in ma_periods:
                full_historical_df_for_analysis[f"MA_{period}"] = (
                    full_historical_df_for_analysis["Close"]
                    .rolling(window=period)
                    .mean()
                )

            fig_ma = go.Figure()
            fig_ma.add_trace(
                go.Scatter(
                    x=full_historical_df_for_analysis.index,
                    y=full_historical_df_for_analysis["Close"],
                    mode="lines",
                    name="Close Price",
                    line=dict(color="blue"),
                )
            )
            for period in ma_periods:
                fig_ma.add_trace(
                    go.Scatter(
                        x=full_historical_df_for_analysis.index,
                        y=full_historical_df_for_analysis[f"MA_{period}"],
                        mode="lines",
                        name=f"{period}-Day MA",
                        line=dict(dash="dash"),
                    )
                )
            fig_ma.update_layout(
                title="Close Price with Moving Averages",
                xaxis_title="Date",
                yaxis_title="Close Price (USD)",
                hovermode="x unified",
                template="plotly_white",
            )
            st.plotly_chart(fig_ma, use_container_width=True)

            # --- Daily Returns Distribution ---
            st.subheader("Daily Returns Distribution")
            full_historical_df_for_analysis["Daily_Return"] = (
                full_historical_df_for_analysis["Close"].pct_change() * 100
            )  # Percentage change

            fig_returns = go.Figure(
                data=[
                    go.Histogram(
                        x=full_historical_df_for_analysis["Daily_Return"].dropna(),
                        nbinsx=50,
                    )
                ]
            )
            fig_returns.update_layout(
                title="Distribution of Daily Returns",
                xaxis_title="Daily Return (%)",
                yaxis_title="Frequency",
                template="plotly_white",
            )
            st.plotly_chart(fig_returns, use_container_width=True)
        else:
            st.warning("No full historical data available for market analysis charts.")

    with tab3:
        st.header("Raw Historical Data")
        st.subheader("Data used for Model Training/Prediction (Close Price Only)")
        st.dataframe(historical_df)  # Show the 'Close' only data used for model
        st.subheader("Full Raw Historical Data (for Analysis Charts)")
        st.dataframe(full_historical_df_for_analysis)  # Show the full data for context

else:
    st.info(
        "Model, scaler, or historical data not loaded. Please ensure the model is trained and data is available."
    )