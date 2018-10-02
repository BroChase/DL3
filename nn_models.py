# Chase Brown
# SID 106015389
# DeepLearning PA 3: CNN

import numpy as np
import layers as lay
import miscfunctions as mf


class NnModels:
    @staticmethod
    def make_cloths_cnn(x_dim, num_class):
        # convolutional layer with x dimentions 2 filters with kernel 3x3 stride of 1 and padding of 0
        conv = lay.Convolutional(x_dim, n_filter=2, h_filter=3, w_filter=3, stride=1, padding=0)
        # activation for layer1 is sigmoid
        sig = mf.Sigmoid()

        # MaxPool layer 2x2 todo should my stride be 1 or 2 or ?
        maxpool = lay.MaxPool(conv.out_dim, size=2, stride=1)

        # convolutional layer with x dimentions 2 filters kernel size of 3x3 stride of 1 and padding of 0
        conv2 = lay.Convolutional(maxpool.out_dim, n_filter=2, h_filter=3, w_filter=3, stride=1, padding=0)
        # activation for layer 2 is rectified linear
        relu = mf.ReLU()

        # MaxPool layer 2x2
        maxpool2 = lay.MaxPool(conv2.out_dim, size=2, stride=1)

        # Flatten the image for input into fully connected layer
        flat = lay.Flatten()

        # Fully connected layer with 50 neurons
        fc1 = lay.FullyConnected(np.prod(maxpool2.out_dim), 50)
        # Activation for fully connected layer of 50 neurons is tanH
        tanh = mf.TanH()

        # Fully connected layer with 10 neurons 'output layer'
        out = lay.FullyConnected(50, num_class)

        return [conv, sig, maxpool, conv2, relu, maxpool2, flat, fc1, tanh, out]

class CNN:
    def __init__(self, layers, loss_func):
        self.layers = layers
        self.params = []
        for layer in self.layers:
            self.params.append(layer.params)
        self.loss_func = loss_func

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        grads = []
        for layer in reversed(self.layers):
            dout, grad = layer.backward(dout)
            grads.append(grad)
        return grads

    def train_step(self, x, y):
        out = self.forward(x)
        loss, dout = self.loss_func(out, y)
        loss += mf.MiscFunctions.l2_regularization(self.layers)
        grads = self.backward(dout)
        grads = mf.MiscFunctions.delta_l2_regularization(self.layers, grads)
        return loss, grads

    def predict(self, x):
        x = self.forward(x)
        return np.argmax(mf.MiscFunctions.soft_max(x), axis=1)