# Chase Brown
# SID 106015389
# DeepLearning PA 3: CNN

import layers as lay
import data_process
from miscfunctions import MiscFunctions as mf


if __name__ == '__main__':
    # Load Data
    ld = data_process.LoadDataModule()
    # Load Data into training/testing sets
    x_train, y_train = ld.load('train')
    x_test, y_test = ld.load('test')

    x_train = mf.reshape_x(x_train)
    x_test = mf.reshape_x(x_test)

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

    cloths_dim = (1, 28, 28)

