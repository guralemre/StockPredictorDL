from fetch_data import get_stock_data
from preprocess import preprocess_data

df = get_stock_data("AAPL", "2018-01-01", "2023-12-31")
X, y, scaler = preprocess_data(df)

print("X shape:", X.shape)
print("y shape:", y.shape)

