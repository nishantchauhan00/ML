import cv2.cv2 as cv2
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer

class data_loader:
    def __init__(self):
        img = cv2.imread(os.path.join(os.getcwd(), "digit.png"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.X = [np.zeros((400)) for n in range(0, 4000)]
        self.X_test = [np.zeros((400)) for n in range(0, 1000)]
        self.Y = [np.zeros((400)) for n in range(0, 4000)]
        self.Y_test = [np.zeros((400)) for n in range(0, 1000)]
        self.cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]

    def load(self):
        n = -1
        n_test = -1
        for i in range(0, 50):
            for j in range(0, 100):
                if j >= 80:
                    n_test = n_test + 1
                    self.X_test[n_test] = (self.cells[i][j].ravel())/255
                    self.Y_test[n_test] = int(int((i + 5) / 5) - 1)
                else:
                    n = n + 1
                    self.X[n] = (self.cells[i][j].ravel())/255
                    self.Y[n] = int(int((i + 5) / 5) - 1)
        train_data = tuple((x, y[0]) for x, y in zip(np.array(self.X), np.array(self.Y).reshape(-1, 1)))
        test_data = tuple((x, y[0]) for x, y in zip(np.array(self.X_test), np.array(self.Y_test).reshape(-1, 1)))
        return (train_data, test_data)

