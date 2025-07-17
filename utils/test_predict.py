from fetch_data import get_stock_data
from predict import predict_next_price

df = get_stock_data("AAPL", "2023-01-01", "2024-12-31")
price = predict_next_price(df)
print(f"Yarınki tahmini fiyat: {price:.2f} USD")
