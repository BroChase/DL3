# Chase Brown
# SID 106015389
# DeepLearning PA 3: CNN

import numpy as np
import data_process
from miscfunctions import MiscFunctions as mf
import nn_models as nn


if __name__ == '__main__':
    # Load Data
    ld = data_process.LoadDataModule()
    # Load Data into training/testing sets
    x_train, y_train = ld.load('train')
    x_test, y_test = ld.load('test')
    # Reshape the data back into images from 784 to 28x28 images
    x_train = mf.reshape_x(x_train)
    x_test = mf.reshape_x(x_test)
    # train shape of (60000, 1, 28, 28) test shape (10000, 1, 28, 28)
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')
    # min max scale from 0-255 to 0-1 scale
    x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
    x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))
    # Dimensions of one image
    cloths_dim = (1, 28, 28)
    # Create the Network.
    cnn = nn.CNN(nn.NnModels.make_cloths_cnn(cloths_dim, num_class=10), loss_func=mf.SoftmaxLoss)
    # todo try to change the learning rate for sgd.  .1 seems to be oversetpping getting 5 for all outputs.
    # cnn = mf.sgd(cnn, x_train, y_train, minibatch_size=200, epoch=20, learning_rate=0.1, x_test=x_test, y_test=y_test)
    cnn = mf.sgd_momentum(cnn, x_train, y_train, minibatch_size=200, epoch=20, lr=0.1, mu=0.9, x_test=x_test, y_test=y_test)