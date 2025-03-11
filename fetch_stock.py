import yfinance as yf

def get_stock_data(ticker="AAPL", start="2020-01-01", end="2024-01-01"):
    stockData = yf.download(ticker, start=start, end=end)
    return stockData

if __name__ == "__main__":
    data = get_stock_data()
    print(data.head())
