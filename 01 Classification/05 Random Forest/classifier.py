import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

data = pd.read_csv("data_winequality.csv")

sc = Normalizer()
data.iloc[:, 1:-1] = sc.fit_transform(data.iloc[:, 1:-1])

X, X_test, Y, Y_test = train_test_split(
    data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2, shuffle=True
)

classifier = RandomForestClassifier(criterion="entropy")
classifier.fit(X, Y)

Y_pred = classifier.predict(X_test)

print("%.2f" % accuracy_score(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))

plt.scatter(X_test["fixed acidity"], Y_test, marker='*', c=Y_test)
plt.scatter(X_test["fixed acidity"], Y_pred, marker='+', c=Y_pred)
plt.show()

