from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import cv2.cv2 as cv2
import os
import numpy as np

img = cv2.imread(os.path.join(os.getcwd(), 'digit.png'))

X = [np.zeros((1200)) for n in range(0, 4000)]
X_test = [np.zeros((1200)) for n in range(0, 1000)]
Y = [np.zeros((1200)) for n in range(0, 4000)]
Y_test = [np.zeros((1200)) for n in range(0, 1000)]
cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]

n=-1
n_test=-1
for i in range(0, 50):
    for j in range(0, 100):
        if j >= 80:
            n_test = n_test+1
            X_test[n_test] = cells[i][j].ravel()
            Y_test[n_test] = int(int((i + 5) / 5) - 1)
        else:
            n = n+1
            X[n] = cells[i][j].ravel()
            Y[n] = int(int((i + 5) / 5) - 1)

classifier = KNeighborsClassifier(n_neighbors=6)
classifier.fit(X, Y)

Y_pred = classifier.predict(X_test)
print(accuracy_score(Y_test, Y_pred))
# print(classification_report(Y_test, Y_pred))