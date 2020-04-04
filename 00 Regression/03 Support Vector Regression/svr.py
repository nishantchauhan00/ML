import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = pd.DataFrame(pd.read_csv('insurance.csv'))
X = data.iloc[:, :6]
Y = data.iloc[:, 6]
print(X, Y)
encoder = LabelEncoder()
X['sex'] = encoder.fit_transform(X['sex'])
X['smoker'] = encoder.fit_transform(X['smoker'])
X['region'] = encoder.fit_transform(X['region'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

# In this case, Linear function or kernel is performing better than rbf and poly
regressor = SVR(kernel='linear') 
regressor.fit(X_train, np.array(Y_train))
Y_pred = regressor.predict(X_test)

plt.plot(X_test['age'], Y_pred, 'b', label='fitted line')
plt.plot(X_test['age'], Y_test, '*', c='r', label='original data')
plt.title('Insurance Charges with Age')
plt.xlabel('age')
plt.ylabel('charges')
plt.legend(title='legend')
plt.show()

