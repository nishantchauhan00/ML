from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import pickle

data = pd.read_csv("data_stock_scaled.csv")

X1, X_test1, Y, Y_test = train_test_split(
    data.iloc[:, 1:5].values, data.iloc[:, 5:].values, test_size=0.2, shuffle=True
)

X = X1[:, 1:]
X_test = X_test1[:, 1:]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

regressor = Sequential()

regressor.add(
    LSTM(10, activation="relu", return_sequences=True, input_shape=(X.shape[1], 1),)
)

regressor.add(LSTM(20, activation="relu", return_sequences=True))

regressor.add(Flatten())

regressor.add(Dense(2, activation="sigmoid"))

regressor.compile(optimizer="adam", loss="binary_crossentropy")

regressor.fit(X, Y, batch_size=10, epochs=10)

Y_pred = regressor.predict(X_test)

print(mean_absolute_error(Y_test, Y_pred))
print(mean_absolute_error(Y_test, Y_pred) * len(Y_test))

# no overfitting so removed dropout
