# predict.py
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import plotly.graph_objects as go
import joblib
from tensorflow.keras.models import load_model

# Fetch crypto data from Binance API
def get_binance_data(symbol='BTCUSDT', interval='1h', limit=500):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
    df['close'] = df['close'].astype(float)
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    return df[['time', 'close']]

# Load trained model and scaler
model = load_model("crypto_model.h5")
scaler = joblib.load("scaler.pkl")

# Get latest data
df = get_binance_data()
scaled_data = scaler.transform(df[['close']])

# Prepare input for prediction
lookback = 24
X_input = np.array([scaled_data[-lookback:]])
predicted_prices_scaled = model.predict(X_input)
predicted_price = scaler.inverse_transform(predicted_prices_scaled)

# Get past 24 hours
past_24h = df[-lookback:]
predicted_times = past_24h['time'].tolist() + [past_24h['time'].iloc[-1] + pd.Timedelta(hours=1)]
predicted_prices_full = np.append(past_24h['close'].values, predicted_price[0][0])

# Plot past 24 hours and predictions
fig = go.Figure()

# Add actual past prices
fig.add_trace(go.Scatter(
    x=past_24h['time'], 
    y=past_24h['close'], 
    mode='lines+markers', 
    name='Actual Prices'
))



# Add predicted prices
fig.add_trace(go.Scatter(
    x=predicted_times, 
    y=predicted_prices_full, 
    mode='lines+markers', 
    name='Predicted Prices',
    line=dict(dash='dot', color='red')
))


fig.update_layout(title='Crypto Price Prediction', xaxis_title='Time', yaxis_title='Price', hovermode='x unified')
fig.show()

print(f'Predicted next hour price: {predicted_price[0][0]:.2f}')
