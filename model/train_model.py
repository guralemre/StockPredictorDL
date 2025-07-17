import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.fetch_data import get_stock_data
from utils.preprocess import preprocess_data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def train_and_save_model(symbol="AAPL", start="2015-01-01", end="2023-12-31", window_size=60, epochs=20):
   
    print("Fetching data...")
    df = get_stock_data(symbol, start, end)
    print("Data shape:", df.shape)
    print(df.head())


  
    X, y, scaler = preprocess_data(df, window_size=window_size)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

 
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")


    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop]
    )



    os.makedirs("model", exist_ok=True)
    model.save("model/model.h5")

   
    import joblib
    joblib.dump(scaler, "model/scaler.pkl")

   
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("Model Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.savefig("model/loss_plot.png")
    plt.close()

if __name__ == "__main__":
    train_and_save_model()