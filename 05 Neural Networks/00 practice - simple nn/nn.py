import numpy as np
from DataProcessing import data_loader
import random
from sklearn import datasets
from sklearn.model_selection import train_test_split

# data[0] - image data,   data[1] - digit
train_data, test_data = data_loader().load()

# parameters
n_epoch = 20
n_batch = 10
n_eta = 3


class network(object):
    def __init__(self, layers):
        self.n_layers = len(layers)
        self.n_neurons = layers
        self.weights = [
            np.random.randn(x1, x)
            for x, x1 in zip(self.n_neurons[:-1], self.n_neurons[1:])
        ]
        # excluding for input layer
        self.bias = [np.random.rand(x) for x in self.n_neurons[1:]]
        # print(self.weights, self.bias)

    def SGD(
        self,
        data_train,
        eta=n_eta,
        epoch=n_epoch,
        mini_batch_size=n_batch,
        data_test=None,
    ):
        # mini batch SGD
        if not data_test is None:
            n_test = len(data_test)
        n = len(data_train)
        for i in range(epoch):
            # random.shuffle(data_train)
            mini_batches = [
                data_train[j : j + mini_batch_size]
                for j in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, mini_batch_size)
            if not data_test is None:
                print(
                    "Epoch {0}: {1}/{2}\n".format(i, self.evaluate(data_test), n_test)
                )
            else:
                print("Epoch {0} complete.\n".format(i))

    def update_mini_batch(self, mini_batch, eta, n):
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            # cost derivatives
            del_nabla_b, del_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, del_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, del_nabla_w)]
        self.bias = [b - (eta / n) * nb for b, nb in zip(self.bias, nabla_b)]
        self.weights = [w - (eta / n) * nw for w, nw in zip(self.weights, nabla_w)]

    def backprop(self, x, y):
        # cost derivatives
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]
        zs = []  # weighted inputs for each layer
        # forward pass - computing activations
        for b, w in zip(self.bias, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # backward pass - computing errors
        delta = self.derivative_cost(activations[-1], y) * self.derivative_sigmoid(
            zs[-1]
        )
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(
            delta.reshape(-1, 1), np.transpose((activations[-2]).reshape(-1, 1))
        )
        for l in range(2, self.n_layers):
            z = zs[-1]
            derivative_sigmoid = self.derivative_sigmoid(z)
            delta = (
                np.dot(np.transpose(self.weights[-l + 1]), delta.reshape(-1, 1))
                * derivative_sigmoid
            )
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, activation[-l + 1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum(int(x == y) for x, y in test_results)

    def feedforward(self, x):
        for b, w in zip(self.bias, self.weights):
            z = np.dot(w, x) + b  # weighted input
            # for sigmoid neurons
            a = self.sigmoid(z)
        return a

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(np.array(z) * (-1)))

    def derivative_sigmoid(self, z):
        return self.sigmoid(z) * (1.0 - self.sigmoid(z))

    def derivative_cost(self, y, a):
        # only for mean squared error
        return y - a


nn = network([400, 20, 10])
nn.SGD(data_train=train_data, epoch=30, mini_batch_size=10, data_test=test_data)

