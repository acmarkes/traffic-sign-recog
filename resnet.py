#Code adapted from the book Hands-on Machine Learning by Aurélien Géron
import tensorflow as tf
from tensorflow import keras

class ResidualBlock(keras.layers.Layer):

    id_block_amount = 0
    conv_block_amount = 0

    def __init__(self, filters, strides=1, activation="relu",**kwargs):
        super().__init__(**kwargs)
    
        if strides == 1:
            ResidualBlock.id_block_amount = ResidualBlock.id_block_amount + 1
        else:
            ResidualBlock.conv_block_amount = ResidualBlock.conv_block_amount + 1

        self._name = f'convolutional_block_{ResidualBlock.conv_block_amount}' if strides == 2 else f'identity_block_{ResidualBlock.id_block_amount}'

        if isinstance(filters,list):
            f1,f2,f3 = filters[0], filters[0], filters[1]
        else:
            f1 = f2 = f3 = filters

        self.activation = keras.activations.get(activation)
        self.id_block = [
            keras.layers.Conv2D(f1, 1, strides=strides,
            padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,

            keras.layers.Conv2D(f2, 3, strides=1,
                                padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,

            keras.layers.Conv2D(f3, 1, strides=1,
                                padding="same", use_bias=False),
            keras.layers.BatchNormalization()
            ]

        self.conv_block = []
        if strides > 1:
            self.conv_block = [
                keras.layers.Conv2D(f3, 1, strides=strides,
                                    padding="same", use_bias=False),
                keras.layers.BatchNormalization()]

    def call(self, inputs):
        Z = inputs
        for layer in self.id_block:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.conv_block:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'activation': self.activation
        })
        return config

class ResNet50(keras.Model):
    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()
        self.num_classes = num_classes
        self.model = keras.models.Sequential(name='ResNet')
        self.model.add(keras.layers.Conv2D(64, 7, strides=2, input_shape=[32, 32, 1],
                                    padding="same", use_bias=False))
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.Activation("relu"))
        self.model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"))

        filter_list = [[64,256]] * 3 + [[128,512]] * 4 + [[256,1024]] * 6 + [[512,2048]] * 3
        prev_filters = []

        for filters in filter_list:        #list of filters to be used
            strides = 1 if filters == prev_filters else 2                   #making images smaller at every change of number of filters
            self.model.add(ResidualBlock(filters, strides=strides))
            prev_filters = filters

        self.model.add(keras.layers.GlobalAvgPool2D())
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(self.num_classes, activation="softmax"))             #fully connected layer outputting 43 classes

    def call(self, inputs):
        return self.model(inputs)

class ResNetOwn(keras.Model):
    def __init__(self, num_classes=1000):
        super(ResNetOwn, self).__init__()
        self.num_classes = num_classes
        self.model = keras.models.Sequential(name='ResNet')
        self.model.add(keras.layers.Conv2D(64, 7, strides=2, input_shape=[32, 32, 1],
                                    padding="same", use_bias=False))
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.Activation("relu"))
        self.model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"))

        filter_list = [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3

        prev_filters = []

        for filters in filter_list:        #list of filters to be used
            strides = 1 if filters == prev_filters else 2                   #making images smaller at every change of number of filters
            self.model.add(ResidualBlock(filters, strides=strides))
            prev_filters = filters

        self.model.add(keras.layers.GlobalAvgPool2D())
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(self.num_classes, activation="softmax"))             #fully connected layer outputting 43 classes

    def call(self, inputs):
        return self.model(inputs)
