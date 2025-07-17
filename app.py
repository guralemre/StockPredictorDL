import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils.fetch_data import get_stock_data
from utils.predict import predict_next_price
import numpy as np

st.title("StockPredictor App")

st.sidebar.header("Parameters")
symbol = st.sidebar.text_input("Stock Symbol (example: AAPL, AKBNK.IS)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

if st.sidebar.button("Predict"):
    data_load_state = st.text("Fetching data...")
    df = get_stock_data(symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    data_load_state.text("Data fetched!")

    if df.empty:
        st.error("Data not found. Please check the symbol and date range.")
    else:
        st.subheader(f"{symbol} Closing Prices")
        st.write(df.tail(10))

       
        try:
            predicted_price = predict_next_price(df)
            st.success(f"The predicted closing price for the next day: {predicted_price:.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

        
        fig, ax = plt.subplots()
        ax.plot(df['Date'], df['Close'], label='Real Closing Price')

        
        try:
            predicted_price = float(np.array(predicted_price).flatten()[0])
        except:
            predicted_price = float(predicted_price)
        
        
        last_date = df['Date'].iloc[-1]
        last_price = float(df['Close'].iloc[-1])
        future_date = df['Date'].max() + pd.Timedelta(days=1)
        
        
        ax.plot([last_date, future_date], [last_price, predicted_price], 
                label='Predicted Price', linestyle='--', color='red', marker='o')

        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        st.pyplot(fig)