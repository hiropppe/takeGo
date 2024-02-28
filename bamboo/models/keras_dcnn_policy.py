import tensorflow as tf

from tensorflow.keras import layers, models

from .keras_nn_util import Bias


class CNNPolicy():

    def __init__(self, **kwargs):
        """construct a convolutional neural network.

        Keword Arguments:
        - input_dim:             depth of features to be processed by first layer (default 48)
        - board:                 width of the go board to be processed (default 19)
        - filters_per_layer:     number of filters used on every layer (default 128)
        - filters_per_layer_K:   (where K is between 1 and <layers>) number of filters
                                 used on layer K (default #filters_per_layer)
        - layers:                number of convolutional steps (default 12)
        - filter_width_K:        (where K is between 1 and <layers>) width of filter on
                                 layer K (default 3 except 1st layer which defaults to 5).
                                 Must be odd.
        """
        defaults = {
            "input_dim": 48,
            "board": 19,
            "filters_per_layer": 192,
            "layers": 12,
            "filter_width_1": 5
        }
        # copy defaults, but override with anything in kwargs
        params = defaults
        params.update(kwargs)

        # create the network:
        # a series of zero-paddings followed by convolutions
        # such that the output dimensions are also board x board
        self.model = models.Sequential()

        input_shape = (params["board"], params["board"], params["input_dim"])

        # create first layer
        self.model.add(layers.Conv2D(
            params["filters_per_layer"],
            (params["filter_width_1"], params["filter_width_1"]),
            kernel_initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05, dtype=tf.float32),
            activation='relu',
            padding='same',
            input_shape=input_shape,
            name='Conv2D_1'))

        # create all other layers
        for i in range(2, params["layers"] + 1):
            # use filter_width_K if it is there, otherwise use 3
            filter_key = "filter_width_%d" % i
            filter_width = params.get(filter_key, 3)

            # use filters_per_layer_K if it is there, otherwise use default value
            filter_count_key = "filters_per_layer_%d" % i
            filter_nb = params.get(filter_count_key, params["filters_per_layer"])

            self.model.add(layers.Conv2D(
                filter_nb,
                (filter_width, filter_width),
                kernel_initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05, dtype=tf.float32),
                activation='relu',
                padding='same',
                name='Conv2D_' + str(i)))

        # the last layer maps each <filters_per_layer> feature to a number
        self.model.add(layers.Conv2D(
            1,
            (1, 1),
            kernel_initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05, dtype=tf.float32),
            padding='same',
            name='Conv2D_13'))
        # reshape output to be board x board
        self.model.add(layers.Flatten())
        # add a bias to each board location
        self.model.add(Bias())
        # softmax makes it into a probability distribution
        self.model.add(layers.Activation('softmax'))
