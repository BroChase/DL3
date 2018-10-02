# Chase Brown
# SID 106015389
# DeepLearning PA 3: CNN

import numpy as np

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