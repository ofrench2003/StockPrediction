# train_model.py
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib

# Fetch crypto data from Binance API
def get_binance_data(symbol='BTCUSDT', interval='1h', limit=500):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
    df['close'] = df['close'].astype(float)
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    return df[['time', 'close']]

# Preprocess data
def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['close']])
    
    X, y = [], []
    lookback = 24  # Using last 24 hours to predict the next hour
    for i in range(len(scaled_data) - lookback):
        X.append(scaled_data[i:i+lookback])
        y.append(scaled_data[i+lookback])
    
    X, y = np.array(X), np.array(y)
    return X, y, scaler

# Build LSTM model
def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train model
df = get_binance_data()
X, y, scaler = preprocess_data(df)

# Train-test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = build_model((X_train.shape[1], X_train.shape[2]))
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

# Save model and scaler
model.save("crypto_model.h5")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved!")