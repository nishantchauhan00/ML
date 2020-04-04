from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

data = datasets.load_digits()

X, X_test, Y, Y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

classifier = KNeighborsClassifier(n_neighbors=6)
classifier.fit(X, Y)

Y_pred = classifier.predict(X_test)
print(accuracy_score(Y_test, Y_pred))
