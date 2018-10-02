# Chase Brown
# SID 106015389
# DeepLearning PA 3: CNN

import numpy as np
from sklearn.utils import shuffle
from scipy.special import expit

class MiscFunctions:
    @staticmethod
    def reshape_x(x_array):
        new_x = []
        for row in x_array:
            img = row.reshape(28, 28)
            new_x.append(img)
        x = np.array(new_x)
        return x

    @staticmethod
    def soft_max(x):
        x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return x/np.sum(x, axis=1, keepdims=True)

    @staticmethod
    def SoftmaxLoss(x, y):
        m = y.shape[0]
        p = MiscFunctions.soft_max(x)
        log_likelihood = -np.log(p[range(m), y])
        loss = np.sum(log_likelihood)/m

        dx = p.copy()
        dx[range(m), y] -= 1
        dx /= m
        return loss, dx

    @staticmethod
    def accuracy(y_true, y_pred):
        return np.mean(y_pred == y_true)

    @staticmethod
    def l2_regularization(layers, lam=0.001):
        reg_loss = 0.0
        for layer in layers:
            if hasattr(layer, 'w'):
                reg_loss += 0.5 * lam * np.sum(layer.w * layer.w)
        return reg_loss

    @staticmethod
    def delta_l2_regularization(layers, grads, lam=0.001):
        for layer, grad in zip(layers, reversed(grads)):
            if hasattr(layer, 'w'):
                grad[0] += lam * layer.w
        return grads

    @staticmethod
    def get_minibatches(x, y, minibatch_size, shuf=True):
        m = x.shape[0]
        minibatches = []
        if shuf:
            x, y = shuffle(x, y)
        for i in range(0, m, minibatch_size):
            x_batch = x[i:i + minibatch_size, :, :, :]
            y_batch = y[i:i + minibatch_size, ]
            minibatches.append((x_batch, y_batch))
        return minibatches

    @staticmethod
    def vanilla_update(params, grads, learning_rate=0.01):
        for param, grad in zip(params, reversed(grads)):
            for i in range(len(grad)):
                param[i] += - learning_rate * grad[i]

    @staticmethod
    def sgd(nnet, x_train, y_train, minibatch_size, epoch, learning_rate, verbose=True, x_test=None, y_test=None):
        minibatches = MiscFunctions.get_minibatches(x_train, y_train, minibatch_size)
        for i in range(epoch):
            loss = 0
            if verbose:
                print('Epoch {0}'.format(i + 1))
            for x_mini, y_mini in minibatches:
                loss, grads = nnet.train_step(x_mini, y_mini)
                MiscFunctions.vanilla_update(nnet.params, grads, learning_rate=learning_rate)
            if verbose:
                train_acc = MiscFunctions.accuracy(y_train, nnet.predict(x_train))
                test_acc = MiscFunctions.accuracy(y_test, nnet.predict(x_test))
                print('Loss = {0} | Training Accuracy = {1} | Test Accuracy = {2}'.format(loss, train_acc, test_acc))
        return nnet


class TanH:
    def __init__(self):
        self.params = []

    def forward(self, x):
        out = np.tanh(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1 - self.out **2)
        return dx, []


class ReLU:
    def __init__(self):
        self.params = []

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout):
        dx = dout.copy()
        dx[self.x <= 0] = 0
        return dx, []

class Sigmoid:
    def __init__(self):
        self.params = []

    def forward(self, x):
        out = expit(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1 - self.out**2)
        return dx, []
