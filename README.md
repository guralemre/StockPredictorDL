# StockPredictorDL

StockPredictorDL is a deep learning-based web application for predicting the next day's closing price of a given stock. It uses historical stock data, preprocesses it, and leverages an LSTM neural network to make predictions. The app is built with Streamlit for an interactive user experience.

## Features
- Fetches historical stock data using Yahoo Finance (via yfinance)
- Preprocesses data with normalization and windowing
- Trains an LSTM-based neural network for time series forecasting
- Predicts the next day's closing price for a given stock symbol
- Visualizes real and predicted prices
- User-friendly web interface with Streamlit

## Project Structure
```
StockPredictorDL/
├── app.py                  # Streamlit web app
├── model/
│   ├── model.h5            # Trained LSTM model
│   ├── scaler.pkl          # Scaler for data normalization
│   └── loss_plot.png       # Training loss plot
├── requirements.txt        # Python dependencies
├── utils/
│   ├── fetch_data.py       # Data fetching utility
│   ├── preprocess.py       # Data preprocessing utility
│   ├── predict.py          # Prediction utility
│   ├── test_fetch.py       # Test for data fetching
│   ├── test_preprocess.py  # Test for preprocessing
│   └── test_predict.py     # Test for prediction
```

## Installation
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd StockPredictorDL
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
2. **Interact with the app:**
   - Enter a stock symbol (e.g., `AAPL`, `AKBNK.IS`)
   - Select the date range
   - Click the "Predict" button to fetch data and predict the next closing price
   - View the results and the price chart

## Model Details
- **Architecture:**
  - LSTM (64 units, return_sequences=True)
  - Dropout (0.2)
  - LSTM (32 units)
  - Dropout (0.2)
  - Dense (1 unit)
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Training:**
  - 80% train / 20% validation split
  - Early stopping on validation loss
  - Training loss plot saved as `model/loss_plot.png`

## Data Source
- Stock data is fetched from Yahoo Finance using the `yfinance` library.

## Testing
You can run the provided test scripts in the `utils/` directory to verify each component:

- **Fetch data:**
  ```bash
  python utils/test_fetch.py
  ```
- **Preprocess data:**
  ```bash
  python utils/test_preprocess.py
  ```
- **Predict price:**
  ```bash
  python utils/test_predict.py
  ```

## License
This project is for educational purposes. 