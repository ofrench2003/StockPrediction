import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load trained model
MODEL_PATH = "stock_model.h5"
model = load_model(MODEL_PATH)

# Function to fetch stock data
def get_stock_data(ticker, start="2015-01-01", end="2024-01-01"):
    stock_data = yf.download(ticker, start=start, end=end)
    return stock_data[['Close']]

# Create sequences for LSTM
def create_sequences(data, sequence_length=50):
    X = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
    return np.array(X)

# Streamlit UI
st.title("📈 Stock Price Prediction Dashboard")

# Select stock ticker
ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA, GOOGL):", "AAPL")

# Load and process data
df = get_stock_data(ticker)
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)

# Prepare data for model
sequence_length = 50
X_test = create_sequences(df_scaled, sequence_length)

# Predict stock prices
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot results
st.subheader(f"{ticker} Stock Price Prediction")
fig, ax = plt.subplots(figsize=(12, 6))

# Plot actual prices
ax.plot(df.index[-len(predicted_prices):], df['Close'].iloc[-len(predicted_prices):], label="Actual Price", color="blue")

# Plot predicted prices
ax.plot(df.index[-len(predicted_prices):], predicted_prices, linestyle="dashed", label="Predicted Price", color="red")

ax.set_xlabel("Date")
ax.set_ylabel("Stock Price (USD)")
ax.legend()
st.pyplot(fig)
