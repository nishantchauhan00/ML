import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report

data = pd.DataFrame(pd.read_csv('data.csv'))

X = data.iloc[:, :-1] 
Y = data.iloc[:, -1] 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)
print(mean_squared_error(Y_test, Y_pred))

plt.plot(X_test['alcohol'], Y_test, '*',markersize=5,c='b', label="fitted data")
plt.plot(X_test['alcohol'], Y_pred, '*',markersize=5,c='r', label="original data")
plt.legend(title='Legend')
plt.title('Alcohol And Rating')
plt.xlabel('alcohol')
plt.ylabel('rating')
plt.show()
