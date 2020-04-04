import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv("Titanic/data.csv")
scaler = Normalizer()
data.iloc[:, 3:] = scaler.fit_transform(data.iloc[:, 3:])
X, X_test, Y, Y_test = train_test_split(
    data.iloc[:, 3:], data.iloc[:, 2], test_size=0.2, shuffle=True
)

print("Without LDA")
print("No. of components- ", len(X.columns))

classifier = KNeighborsClassifier()
classifier.fit(X, Y)

Y_pred = classifier.predict(X_test)
print(accuracy_score(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))

# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=1)
lda = lda.fit(np.array(data.iloc[:, 3:]), np.array(data.iloc[:, 2]))
X = lda.transform(X)
X_test = lda.transform(X_test)


print("\nWith LDA-")
print("No. of components- ", 1)
classifier1 = KNeighborsClassifier()
classifier1.fit(X, Y)

Y_pred = classifier1.predict(X_test)
print(accuracy_score(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))
