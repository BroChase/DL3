# Chase Brown
# SID 106015389
# DeepLearning PA 3: CNN

import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
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
        """
        Stable softmax function as to not over run the float max of python 10^308
        :param x:
        :return:
        """
        x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return x/np.sum(x, axis=1, keepdims=True)

    @staticmethod # loss_function
    def SoftmaxLoss(x, y):
        """
        Performs softmax on X and then calculates cross_entropy 'Loss'
        :param x: Output of the fully connected layer for output Samples x classes
        :param y: 1 x number of examples 'true values of the samples of x'
        :return: Loss and delta of X
        """
        prediction = MiscFunctions.soft_max(x)
        samples = y.shape[0]
        # from the predicted values take the value in prediction under the column that it is predicted to be.
        # If column 7 for predicted is .123432 then take that value and put it into log_likelibood for the first sample.
        z = prediction[range(samples), y]
        log_likelihood = -np.log(z)
        # The Loss is the Sum of the likelihood / sample size
        loss = np.sum(log_likelihood)/samples

        delta_x = prediction.copy()
        # from the 'true y position' in the predicted x subtract 1
        delta_x[range(samples), y] -= 1
        # devide all the values in delta_x by the sample size
        delta_x /= samples

        return loss, delta_x

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
                z = nnet.predict(x_train)
                train_acc = accuracy_score(y_train, z)
                test_acc = accuracy_score(y_test, nnet.predict(x_test))
                print('Loss = {0} | Training Accuracy = {1} | Test Accuracy = {2}'.format(loss, train_acc, test_acc))
        return nnet

    @staticmethod
    def momentum_update(velocity, params, grads, learning_rate=0.01, mu=0.9):
        for v, param, grad in zip(velocity, params, reversed(grads)):
            for i in range(len(grad)):
                v[i] = mu*v[i] + learning_rate * grad[i]
                param[i] -= v[i]

    @staticmethod
    def sgd_momentum(nnet, x_train, y_train, minibatch_size, epoch, lr, mu, x_test, y_test, verbose=True):
        minibatches = MiscFunctions.get_minibatches(x_train, y_train, minibatch_size)
        for i in range(epoch):
            loss = 0
            velocity = []
            for param_layer in nnet.params:
                p = [np.zeros_like(param) for param in list(param_layer)]
                velocity.append(p)

            if verbose:
                print('Epoch {0}'.format(i + 1))
            for x_mini, y_mini in minibatches:
                loss, grads = nnet.train_step(x_mini, y_mini)
                MiscFunctions.momentum_update(velocity, nnet.params, grads, learning_rate=lr, mu=mu)
            if verbose:
                train_acc = accuracy_score(y_train, nnet.predict(x_train))
                test_acc = accuracy_score(y_test, nnet.predict(x_test))
                print('Loss = {0} | Training Accuracy = {1} | Test Accuracy = {2}'.format(loss, train_acc, test_acc))

        return nnet


class TanH:
    def __init__(self):
        self.params = []

    def forward(self, x):
        """
        Passes the Matrix X into the tanh function for activation.
        :param x: nxm Matrix
        :return:  nxm matrix
        """
        out = np.tanh(x)
        self.out = out
        return out

    def backward(self, dout):
        delta_x = dout * (1 - self.out ** 2)
        return delta_x, []


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
