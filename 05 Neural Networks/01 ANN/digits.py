import cv2.cv2 as cv2
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder

img = cv2.imread(os.path.join(os.getcwd(), "digit.png"))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

X = [np.zeros((400)) for n in range(0, 4500)]
X_test = [np.zeros((400)) for n in range(0, 500)]
Y = [np.zeros((400)) for n in range(0, 4500)]
Y_test = [np.zeros((400)) for n in range(0, 500)]
cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]

n = -1
n_test = -1
for i in range(0, 50):
    for j in range(0, 100):
        if j >= 90:
            n_test = n_test + 1
            X_test[n_test] = cells[i][j].ravel() / 255
            Y_test[n_test] = int(int((i + 5) / 5) - 1)
        else:
            n = n + 1
            X[n] = cells[i][j].ravel() / 255
            Y[n] = int(int((i + 5) / 5) - 1)

encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()
Y_test = encoder.transform(np.array(Y_test).reshape(-1, 1)).toarray()


def digits():
    return X, Y, X_test, Y_test, encoder

