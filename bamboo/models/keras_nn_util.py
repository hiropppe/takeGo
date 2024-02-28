import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import Layer


class Bias(Layer):
    """Custom keras layer that simply adds a scalar bias to each location in the input

    Largely copied from the keras docs:
    http://keras.io/layers/writing-your-own-keras-layers/#writing-your-own-keras-layers
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='bias',
                                 shape=input_shape[1:],
                                 dtype=np.float32,
                                 initializer=tf.zeros_initializer,
                                 trainable=True)
        super().build(input_shape)  # Be sure to call this somewhere!

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, x):
        return x + self.W