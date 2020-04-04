from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score

data = pd.DataFrame(pd.read_csv("data.csv"))
test_data = pd.DataFrame(pd.read_csv("test_data.csv"))

# print(data.shape, test_data.shape) # 792*16
# print(data.isnull().sum()) # No null values present

data.drop("PassengerId", axis=1, inplace=True)
test_data.drop("PassengerId", axis=1, inplace=True)
classifier = LogisticRegression()

X = data.iloc[:, 1:]
X_test = test_data.iloc[:, 1:]
Y = data.iloc[:, 0]
Y_test = test_data.iloc[:, 0]

rfecv = RFECV(estimator=classifier, scoring="accuracy")
rfecv.fit(X, Y)

selected_features = X.columns[rfecv.support_]
print(rfecv.n_features_, selected_features)
X = X[selected_features]
X_test = X_test[selected_features]

# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
# plt.show()
# print(X.corr())

classifier.fit(X, Y)
Y_pred = classifier.predict(X_test)

print("Accuracy Score: ", accuracy_score(Y_test, Y_pred))
print("\nConfusion Matrix: \n", confusion_matrix(Y_test, Y_pred))
print("Not survived:", Y_test.isin([0]).sum(), "   survived:", Y_test.isin([1]).sum())

# print(cross_val_score(classifier,X, Y, cv=10, scoring='accuracy').mean())
# print(cross_val_score(classifier,X, Y, cv=10, scoring='roc_auc').mean())
# print(cross_val_score(classifier,X, Y, cv=10, scoring='neg_log_loss').mean())

