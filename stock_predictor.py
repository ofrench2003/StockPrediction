import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Function to get stock data
def get_stock_data(ticker="AAPL", start="2020-01-01", end="2024-01-01"):
    stockData = yf.download(ticker, start=start, end=end)
    return stockData[['Close']]

# Prepare Data
def create_sequences(data, sequence_length=50):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

# Load data
df = get_stock_data()
scaler = MinMaxScaler(feature_range=(0,1))
dfScaled = scaler.fit_transform(df)

# Create sequences
sequenceLength = 50
X, y = create_sequences(dfScaled, sequenceLength)

# Split data
trainSize = int(len(X) * 0.8)
XTrain, XTest = X[:trainSize], X[trainSize:]
yTrain, yTest = y[:trainSize], y[trainSize:]

# Build the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(sequenceLength, 1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(50, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(25),
    tf.keras.layers.Dense(1)
])

# Compile & Train
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(XTrain, yTrain, epochs=100, batch_size=32, validation_data=(XTest, yTest))

# Save the trained model
model.save("stock_model.h5")  # Save at the end, after training

# Make predictions
predictedPrices = model.predict(XTest)
predictedPrices = scaler.inverse_transform(predictedPrices)

# Plot results
plt.figure(figsize=(12,6))
plt.plot(df.index[trainSize + sequenceLength:], df['Close'][trainSize + sequenceLength:], label="Actual Price")
plt.plot(df.index[trainSize + sequenceLength:], predictedPrices, label="Predicted Price")
plt.legend()
plt.show()
