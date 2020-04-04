from time import clock
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Flatten
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

t1 = clock()

regressor = Sequential()

regressor.add(
    SimpleRNN(
        10,
        activation="relu",
        dropout=0.2,
        return_sequences=True,
        input_shape=(X.shape[1], 1),
    )
)
regressor.add(SimpleRNN(20, activation="relu", dropout=0.2, return_sequences=True))

regressor.add(Flatten())

regressor.add(Dense(2))

regressor.compile(optimizer="adam", loss="mse")

regressor.fit(X, Y, batch_size=10, epochs=15)

Y_pred = regressor.predict(X_test)

t2 = clock()

print(mean_absolute_error(Y_pred, Y_test))
print(mean_absolute_error(Y_pred, Y_test) * len(X_test))
print((t2 - t1), " Sec")
# 0.11804877402635625
# 29.748291054641776
# 4.3586423  Sec


plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=20))
plt.plot(X_test1[:, 0], Y_test[:, 0], "b", label="Original")
plt.plot(X_test1[:, 0], Y_pred[:, 0], "r", label="Predicted")
plt.xlabel("Time")
plt.ylabel("Close")
plt.legend(title="Legend")
plt.show()

with open("regressor_rnn.pkl", "wb") as f:
    pickle.dump(regressor, f)
