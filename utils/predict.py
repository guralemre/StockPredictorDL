import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

def predict_next_price(data: pd.DataFrame, window_size: int = 60):

    last_60 = data['Close'].values[-window_size:].reshape(-1, 1)

    scaler = joblib.load("model/scaler.pkl")
    scaled_input = scaler.transform(last_60)
   
    X_pred = np.reshape(scaled_input, (1, window_size, 1))
    model = load_model("model/model.h5")
   
    scaled_prediction = model.predict(X_pred)

    predicted_price = scaler.inverse_transform(scaled_prediction)[0][0]
    return predicted_price

