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
        layer0 = lay.Convolutional(x_dim, n_filter=2, h_filter=3, w_filter=3, stride=1, padding=0)
        # activation for layer1 is sigmoid
        layer0_activation = mf.Sigmoid()

        # MaxPool layer 2x2 todo should my stride be 1 or 2 or ?
        layer1 = lay.MaxPool(layer0.out_dim, size=2, stride=2)

        # convolutional layer with x dimentions 2 filters kernel size of 3x3 stride of 1 and padding of 0
        layer2 = lay.Convolutional(layer1.out_dim, n_filter=2, h_filter=3, w_filter=3, stride=1, padding=0)
        # activation for layer 2 is rectified linear
        layer2_activation = mf.ReLU()

        # MaxPool layer 2x2
        layer3 = lay.MaxPool(layer2.out_dim, size=2, stride=2)

        # Flatten the image for input into fully connected layer
        layer4 = lay.Flatten()

        # Fully connected layer with 50 neurons
        layer5 = lay.FullyConnected(np.prod(layer3.out_dim), 50)
        # Activation for fully connected layer of 50 neurons is tanH
        layer5_activation = mf.TanH()

        # Fully connected layer with 10 neurons 'output layer'
        layer6 = lay.FullyConnected(50, num_class)

        return [layer0, layer0_activation, layer1, layer2, layer2_activation, layer3, layer4]