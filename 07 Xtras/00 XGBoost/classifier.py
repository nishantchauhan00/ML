from time import time
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import pickle

data = pd.read_csv("data_stock_scaled.csv")

X1, X_test1, Y, Y_test = train_test_split(
    data.iloc[:, 1:5].values, data.iloc[:, 5:-1].values, test_size=0.2, shuffle=True
)

X = X1[:, 1:]
X_test = X_test1[:, 1:]

t1 = time()

regressor = XGBRegressor()
regressor.fit(X, Y)

Y_pred = regressor.predict(X_test)

t2 = time()

print(mean_absolute_error(Y_pred, Y_test))
print(mean_absolute_error(Y_pred, Y_test) * len(X_test))
print((t2 - t1), "Sec")
# 0.05335297298403846
# 13.444949191977692
# 0.10934853553771973 Sec

with open("regressor_gb.pkl", "wb") as f:
    pickle.dump(regressor, f)
