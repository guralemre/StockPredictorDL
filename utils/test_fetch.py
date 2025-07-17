from fetch_data import get_stock_data

df = get_stock_data("AAPL", "2018-01-01", "2023-12-31")
print(df.head())        