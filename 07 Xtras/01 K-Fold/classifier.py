from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import cross_val_score

data = pd.read_csv("data_stock_scaled.csv")

X1, X_test1, Y, Y_test = train_test_split(
    data.iloc[:, 1:5].values, data.iloc[:, 5:-1].values, test_size=0.2, shuffle=True
)
X = X1[:, 1:]
regressor = XGBRegressor()


# 5 folds
score = cross_val_score(estimator=regressor, X=X, y=Y, cv=5)
# can be done using cross_val_predict too
print(score)
# a negative score just means that the
# particular model is performing quite poorly
print(score.mean())
print(score.std())

# cross val score
# Returns - Array of scores of the estimator for each run
# of the cross validation.
