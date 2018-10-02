# Chase Brown
# SID 106015389
# DeepLearning PA 3: CNN

import layers as lay
import data_process
from miscfunctions import MiscFunctions as mf
import nn_models as nn

if __name__ == '__main__':
    # Load Data
    ld = data_process.LoadDataModule()
    # Load Data into training/testing sets
    x_train, y_train = ld.load('train')
    x_test, y_test = ld.load('test')

    x_train = mf.reshape_x(x_train)
    x_test = mf.reshape_x(x_test)

    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')

    cloths_dim = (1, 28, 28)

    cnn = nn.CNN(nn.NnModels.make_cloths_cnn(cloths_dim, num_class=10), loss_func=mf.SoftmaxLoss)

    cnn = mf.sgd(cnn, x_train, y_train, minibatch_size=200, epoch=20, learning_rate=0.1)