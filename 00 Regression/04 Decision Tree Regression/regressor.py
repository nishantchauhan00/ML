import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.DataFrame(pd.read_csv('cars.csv'))
data.replace(' ', 0,inplace=True)
X = data.iloc[:, :7].copy()
Y = data.iloc[:, 7].copy()

encoder = LabelEncoder()
Y = encoder.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, shuffle=True)

regressor = DecisionTreeRegressor()
regressor.fit(np.array(X_train), Y_train)

Y_pred = regressor.predict(np.array(X_test))

# print(mean_squared_error(Y_test, Y_pred))
err= "Mean Squared Error : " + str(mean_squared_error(Y_test, Y_pred))

plt.plot(X_test['hp'], Y_pred, '*',markersize=5,c='b', label="fitted data")
plt.plot(X_test['hp'], Y_test, '+',markersize=5,c='r', label="original data")
plt.legend(title="Legend")
plt.xlabel('hp')
plt.ylabel('brands')
plt.title("Horsepower and Brands")
plt.text(80, 1.5,err)
plt.show()

