import numpy as np
from operator import sub
from math import exp

class func(object):
    def sigmoid(self, z):
        return 1.0 / (1.0 + exp(-z))

    def Dsigmoid(self, z):
        return self.sigmoid(z) / (1.0 - self.sigmoid(z))

class cost(object):
    def MSE(self, y, a, n):
        """
        Mean Squared Error
            or
        Quadratic Cost Function
        """
        if not n:
            n = len(y)
        return np.mean(np.square(np.array(list(y)) - np.array(list(a)))) / (2 * n)

    def D_MSE(self, y, a):
        """
        Derivative of 
            Mean Squared Error
                or
            Quadratic Cost Function

        Returns a vector
        """
        return list(map(sub, y, a))

    def RMSE(self, y, a, n):
        """
        Root Mean Squared Error
        """
        if not n:
            n = len(y)
        return np.sqrt(np.mean(np.square(np.array(list(y)) - np.array(list(a)))))
