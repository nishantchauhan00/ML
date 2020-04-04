import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV

data = pd.DataFrame(pd.read_csv("insurance.csv"))
X = data.iloc[:, :6]
Y = data.iloc[:, 6]

encoder = LabelEncoder()
X["sex"] = encoder.fit_transform(X["sex"])
X["smoker"] = encoder.fit_transform(X["smoker"])
X["region"] = encoder.fit_transform(X["region"])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

regressor = SVR(kernel="linear")

# grid = [
#     {"kernel": ["poly"], "degree": [2, 3, 4], "C":[1, 10, 100]},
#     {"kernel": ["linear"], "C":[1, 10, 100]},
#     {"kernel": ["rbf"], "C":[1, 10, 100]},
# ]
# First Result - {'C': 100, 'kernel': 'linear'}

# grid = [{"kernel": ["linear"], "C":[50, 100, 150]}]
# Second - {'C': 150, 'kernel': 'linear'}

# grid = [{"kernel": ["linear"], "C":[200, 300, 400, 1000, 2000]}]
# Third - {'C': 1000, 'kernel': 'linear'}

# grid = [{"kernel": ["linear"], "C":[700, 900, 1000, 1200, 1400]}]
# Fourth - {'C': 1200, 'kernel': 'linear'}

# grid = [{"kernel": ["linear"], "C":[1150, 1200, 1350, 1300]}]
# Fifth - {'C': 1200, 'kernel': 'linear'}
# 0.7084103250948471

grid = [{"kernel": ["linear"], "C": [1190, 1200, 1250, 1225]}]
# 0.7089195135386943(not improving much now, we got 54 to 70)
# {'C': 1225, 'kernel': 'linear'}

grid_search = GridSearchCV(estimator=regressor, param_grid=grid, n_jobs=-1, cv=10)
grid_search.fit(X, Y)
print(grid_search.best_score_)
print(grid_search.best_params_)
