import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("data_stock.csv")
data["Volume"] = list((float(vol.replace(",", ""))) / 100 for vol in data["Volume"])
data["Close"] = list((float(c.replace(",", ""))) for c in data["Close"])

scaler_close = MinMaxScaler()
scaler_open = MinMaxScaler()
scaler_high = MinMaxScaler()
scaler_low = MinMaxScaler()
scaler_vol = MinMaxScaler()
data["Close"] = np.around(
    scaler_close.fit_transform(data["Close"].values.reshape(-1, 1)), decimals=7
)
data["Open"] = np.around(
    scaler_open.fit_transform(data["Open"].values.reshape(-1, 1)), decimals=7
)
data["Low"] = np.around(
    scaler_low.fit_transform(data["Low"].values.reshape(-1, 1)), decimals=7
)
data["High"] = np.around(
    scaler_high.fit_transform(data["High"].values.reshape(-1, 1)), decimals=7
)
data["Volume"] = np.around(
    scaler_vol.fit_transform(data["Volume"].values.reshape(-1, 1)), decimals=7
)

data.to_csv("data_stock_scaled.csv")
