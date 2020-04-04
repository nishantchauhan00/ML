from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

m = 1000  # no. of samples
n = 29  # no. of features
data = pd.DataFrame(pd.read_csv("data.csv")).iloc[:m, :]

X = data[
    [
        "LotFrontage",
        "LotArea",
        "LotShape",
        "LotConfig",
        "Neighborhood",
        "HouseStyle",
        "CentralAir",
        "OverallQual",
        "OverallCond",
        "YearBuilt",
        "YearRemodAdd",
        "RoofStyle",
        "BsmtFullBath",
        "BsmtHalfBath",
        "FullBath",
        "HalfBath",
        "BedroomAbvGr",
        "KitchenAbvGr",
        "SaleType",
        "MiscVal",
        "MoSold",
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
        "PoolArea",
        "GarageCars",
        "GarageArea",
    ]
].copy()
Y = data["SalePrice"].copy()
encoder = LabelEncoder()
X["LotShape"] = encoder.fit_transform(X["LotShape"])
X["LotConfig"] = encoder.fit_transform(X["LotConfig"])
X["Neighborhood"] = encoder.fit_transform(X["Neighborhood"])
X["HouseStyle"] = encoder.fit_transform(X["HouseStyle"])
X["RoofStyle"] = encoder.fit_transform(X["RoofStyle"])
X["CentralAir"] = encoder.fit_transform(X["CentralAir"])
X["SaleType"] = encoder.fit_transform(X["SaleType"])

for i in range(0, m):
    for j in range(0, n):
        if np.isnan(X.iloc[i, j]):
            X.iloc[i, j] = 0
Y = Y/1000

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, shuffle=True, test_size=0.2, random_state=0
)

year_built = np.array(X_test["YearBuilt"])

regressor = PolynomialFeatures(degree=2)  # min and default degree : 2
regressor.fit(X)
X_train = regressor.transform(X_train)
X_test = regressor.transform(X_test)

linear_regressor = LinearRegression()
linear_regressor.fit(X_train, Y_train)

Y_pred =linear_regressor.predict(X_test)
print("Mean Squared Error: ", mean_squared_error(Y_test, Y_pred))
print("Root Mean Squared Error: ", sqrt(mean_squared_error(Y_test, Y_pred)))

plt.plot(
    year_built,
    Y_pred,
    "*",
    c="b",
    label="fitted line"
)
plt.plot(year_built, Y_test, "*", c="r", label="original data")
plt.title("Decrease in Price with Year")
plt.xlabel("Year")
plt.ylabel("Price")
plt.legend(title="Legend")
plt.show()
