import yfinance as yf
import pandas as pd

def get_stock_data(symbol: str,start_date: str,end_date: str) -> pd.DataFrame:
    try:
        data=yf.download(symbol,start=start_date,end=end_date)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

