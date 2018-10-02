import numpy as np
import col_indices as ci


class Convolutional:
    def __init__(self, x_dim, n_filter, h_filter, w_filter, stride, padding):
        self.seed = 106015389
        np.random.seed(self.seed)

        self.d_x, self.h_x, self.w_x = x_dim

        self.n_filter = n_filter
        self.h_filter = h_filter
        self.w_filter = w_filter

        self.stride = stride
        self.padding = padding
        # w = (number of filters, dimensions, filter height, filter width)
        self.w = np.random.randn(n_filter, self.d_x, h_filter, w_filter) / np.sqrt(n_filter / 2)
        # bias = (n_filters, 1)
        self.b = np.zeros((self.n_filter, 1))

        self.params = [self.w, self.b]

        self.h_out = (self.h_x - h_filter + 2 * padding) / stride + 1
        self.w_out = (self.w_x - w_filter + 2 * padding) / stride + 1

        if not self.h_out.is_integer() or not self.w_out.is_integer():
            raise Exception('Invalid dimensions!')
        # convert the floats of h_out and w_out to ints
        self.h_out, self.w_out = int(self.h_out), int(self.w_out)
        self.out_dim = (self.n_filter, self.h_out, self.w_out)

    def forward(self, x):
        self.n_x = x.shape[0]

        self.x_col = ci.im2col(x, self.h_filter, self.w_filter, stride=self.stride, padding=self.padding)
        w_row = self.w.reshape(self.n_filter, -1)

        out = w_row @ self.x_col + self.b
        out = out.reshape(self.n_filter, self.h_out, self.w_out, self.n_x)
        out = out.transpose(3, 0, 1, 2)
        return out

    def backward(self, dout):
        dout_flat = dout.transpose(1, 2, 3, 0).reshape(self.n_filter, -1)

        dw = dout_flat @ self.x_col.T
        dw = dw.reshape(self.w.shape)

        db = np.sum(dout, axis=(0, 2, 3)).reshape(self.n_filter, -1)

        w_flat = self.w.reshape(self.n_filter, -1)

        dx_col = w_flat.T @ dout_flat
        shape = (self.n_x, self.d_x, self.h_x, self.w_x)
        dx = ci.col2im(dx_col, shape, self.h_filter, self.w_filter, self.padding, self.stride)

        return dx, [dw, db]


class MaxPool:
    def __init__(self, x_dim, size, stride):
        # d_x = dimensions h_x =  height of x w_x = width of x
        self.d_x, self.h_x, self.w_x = x_dim
        # keep the params for the MaxPool
        self.params = []

        self.size = size
        self.stride = stride
        # The output size after MaxPool filter window
        self.h_out = (self.h_x - size)/stride + 1
        self.w_out = (self.w_x - size)/stride + 1
        # error check
        if not self.h_out.is_integer() or not self.w_out.is_integer():
            raise Exception('Invalid dimensions')

        self.h_out, self.w_out = int(self.h_out), int(self.w_out)
        self.out_dim = (self.d_x, self.h_out, self.w_out)

    def forward(self, x):
        self.n_x = x.shape[0]
        x_reshaped = x.reshape(x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3])

        self.x_col = ci.im2col(x_reshaped, self.size, self.size, padding=0, stride=self.stride)

        self.max_indexes = np.argmax(self.x_col, axis=0)

        out = self.x_col[self.max_indexes, range(self.max_indexes.size)]

        out = out.reshape(self.h_out, self.w_out, self.n_x, self.d_x).transpose(2, 3, 0, 1)

        return out

    def backward(self, dout):
        dx_col = np.zeros_like(self.x_col)
        # flatten the gradient
        dout_flat = dout.transpose(2, 3, 0, 1).ravel()

        dx_col[self.max_indexes, range(self.max_indexes.size)] = dout_flat

        # get the original x_reshaped structure from col2im
        shape = (self.n_x * self.d_x, 1, self.h_x, self.w_x)
        dx = ci.col2im(dx_col, shape, self.size, self.size, padding=0, stride=self.stride)
        dx = dx.reshape(self.n_x, self.d_x, self.h_x, self.w_x)
        return dx, []


class Flatten:
    def __init__(self):
        self.params = []

    def forward(self, x):
        self.x_shape = x.shape
        self.out_shape = (self.x_shape[0], -1)
        out = x.ravel().reshape(self.out_shape)
        self.out_shape = self.out_shape[1]
        return out

    def backward(self, dout):
        out = dout.reshape(self.x_shape)
        return out, ()


class FullyConnected:
    def __init__(self, in_size, out_size):
        self.w = np.random.randn(in_size, out_size) / np.sqrt(in_size / 2.)
        self.b = np.zeros((1, out_size))
        self.params = [self.w, self.b]

    def forward(self, x):
        self.x = x
        out = self.x @ self.w + self.b
        return out

    def backward(self, dout):
        dw = self.x.T @ dout
        db = np.sum(dout, axis=0)
        dx = dout @ self.w.T
        return dx, [dw, db]