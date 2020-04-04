import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

data = pd.DataFrame(pd.read_csv("data.csv"))
scaler = Normalizer()
data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1].values)
X_train, X_test, Y_train, Y_test = train_test_split(
    data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2, shuffle=True
)
print("Dimension before PCA:",len(X_train.columns))

regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)
print("Error Without PCA: {:.2f}%".format(mean_absolute_error(Y_test, Y_pred)*10))

# # PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=7).fit(data.iloc[:, :-1])
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
print("Dimension after PCA:", X_train.shape[1])

 # Regressor
regressor1 = RandomForestRegressor(n_estimators=100)
regressor1.fit(X_train, Y_train)
Y_pred = regressor1.predict(X_test)
print("Error With PCA: {:.2f}%".format(mean_absolute_error(Y_test, Y_pred)*10))

# Dimension before PCA: 11
# Error Without PCA: 4.25%
# Dimension after PCA: 7
# Error With PCA: 3.98%