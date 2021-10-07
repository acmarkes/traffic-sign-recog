#Code adapted from the book Hands-on Machine Learning by Aurélien Géron
import tensorflow as tf
from tensorflow import keras

class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.id_block = [
            keras.layers.Conv2D(filters, 1, strides=strides,
            padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,

            keras.layers.Conv2D(filters, 3, strides=1,
                                padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,

            keras.layers.Conv2D(filters, 1, strides=1,
                                padding="same", use_bias=False),
            keras.layers.BatchNormalization()
            ]

        self.conv_block = []
        if strides > 1:
            self.conv_block = [
                keras.layers.Conv2D(filters, 1, strides=strides,
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