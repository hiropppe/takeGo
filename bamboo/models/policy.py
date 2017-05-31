from keras.models import Sequential
from keras.layers import convolutional
from keras.layers.core import Activation, Flatten

from bamboo.models.nn_util import Bias, NeuralNetBase, neuralnet


@neuralnet
class CNNPolicy(NeuralNetBase):
    """uses a convolutional neural network to evaluate the state of the game
    and compute a probability distribution over the next action
    """

    def _select_moves_and_normalize(self, nn_output, moves):
        """helper function to normalize a distribution over the given list of moves
        and return a list of (move, prob) tuples
        """
        if len(moves) == 0:
            return []
        distribution = nn_output[moves]
        # get network activations at legal move locations
        distribution = distribution / distribution.sum()
        return zip(moves, distribution)

    def eval_state(self, tensor, moves=None):
        """Given a GameState object, returns a list of (action, probability) pairs
        according to the network outputs

        If a list of moves is specified, only those moves are kept in the distribution
        """
        # run the tensor through the network
        network_output = self.forward(tensor)
        return network_output[0]

    @staticmethod
    def create_network(**kwargs):
        """construct a convolutional neural network.

        Keword Arguments:
        - input_dim:             depth of features to be processed by first layer (no default)
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
            "board": 19,
            "filters_per_layer": 128,
            "layers": 12,
            "filter_width_1": 5
        }
        # copy defaults, but override with anything in kwargs
        params = defaults
        params.update(kwargs)

        # create the network:
        # a series of zero-paddings followed by convolutions
        # such that the output dimensions are also board x board
        network = Sequential()

        # create first layer
        network.add(convolutional.Convolution2D(
            input_shape=(params["input_dim"], params["board"], params["board"]),
            nb_filter=params.get("filters_per_layer_1", params["filters_per_layer"]),
            nb_row=params["filter_width_1"],
            nb_col=params["filter_width_1"],
            init='uniform',
            activation='relu',
            border_mode='same'))

        # create all other layers
        for i in range(2, params["layers"] + 1):
            # use filter_width_K if it is there, otherwise use 3
            filter_key = "filter_width_%d" % i
            filter_width = params.get(filter_key, 3)

            # use filters_per_layer_K if it is there, otherwise use default value
            filter_count_key = "filters_per_layer_%d" % i
            filter_nb = params.get(filter_count_key, params["filters_per_layer"])

            network.add(convolutional.Convolution2D(
                nb_filter=filter_nb,
                nb_row=filter_width,
                nb_col=filter_width,
                init='uniform',
                activation='relu',
                border_mode='same'))

        # the last layer maps each <filters_per_layer> feature to a number
        network.add(convolutional.Convolution2D(
            nb_filter=1,
            nb_row=1,
            nb_col=1,
            init='uniform',
            border_mode='same'))
        # reshape output to be board x board
        network.add(Flatten())
        # add a bias to each board location
        network.add(Bias())
        # softmax makes it into a probability distribution
        network.add(Activation('softmax'))

        return network
