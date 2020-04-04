import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('Titanic/data.csv')
test_data = pd.read_csv('Titanic/test_data.csv')
X = data.iloc[:, 3:]
Y = data.iloc[:, 2]
X_test = test_data.iloc[:, 3:]
Y_test = test_data.iloc[:, 2]

classifier = KNeighborsClassifier()
classifier.fit(X, Y)

Y_pred = classifier.predict(X_test)
print(accuracy_score(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))
