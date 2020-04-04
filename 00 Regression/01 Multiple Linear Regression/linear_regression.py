from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import statsmodels.api as sm

data = pd.read_csv('data.csv')

X = (data[['year', 'population']]).copy()
Y = data[['gdp_per_capita ($)']].copy()

# Standardization
# print(np.mean(Y)) # gdp mean- 1859
# print(np.std(Y, ddof= 1)) #gdp standard deviation- 1387
# print(np.mean(X.values[:263, :2])) # population mean - 119031
# print(np.std(X.values[:263, :2])) # population standard deviation- 140469
X['population'] = preprocessing.scale(X['population'])

regressor = LinearRegression(fit_intercept=True,copy_X=True)
regressor.fit(X, Y)
y= regressor.predict(X)
plt.plot(np.array(X['year']), y, 'b', label = 'fitted line')
plt.plot(np.array(X['year']), Y['gdp_per_capita ($)'], '*',c='r', label = 'original data')
plt.title("Albania GDP growth")
plt.legend(title="Legend")
plt.xlabel("year")
plt.ylabel("gdp per capita")
plt.show()

x_const = sm.add_constant(X)
est = sm.OLS(Y,x_const)
est2 = est.fit()
print(est2.summary())