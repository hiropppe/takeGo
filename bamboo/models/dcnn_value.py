from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

from . import nn_util


BOARD_SIZE = 19
INPUT_DEPTH = 49
FILTER_SIZE = 192
FILTER_WIDTH_1 = 5
FILTER_WIDTH_2_12 = 3


_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(inputs, is_training):
    """Performs a batch normalization followed by a ReLU."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=is_training, fused=True)
    inputs = tf.nn.relu(inputs)
    return inputs


def get_initial_weight(layer, wb, scope_name):
    if wb.lower() == 'w':
        if layer == 1:
            return nn_util.random_uniform(
                scope_name + '_W',
                [FILTER_WIDTH_1, FILTER_WIDTH_1, INPUT_DEPTH, FILTER_SIZE],
                minval=-0.05, maxval=0.05)
        elif layer <= 12:
            return nn_util.random_uniform(
                scope_name + '_W',
                [FILTER_WIDTH_2_12, FILTER_WIDTH_2_12, FILTER_SIZE, FILTER_SIZE],
                minval=-0.05, maxval=0.05)
        elif layer == 13:
            return nn_util.random_uniform(
                scope_name + '_W',
                [1, 1, FILTER_SIZE, 1],
                minval=-0.05, maxval=0.05)
        elif layer == 14:
            return nn_util.random_uniform(
                scope_name + '_W',
                [361, 256],
                minval=-0.05, maxval=0.05)
        elif layer == 15:
            return nn_util.random_uniform(
                scope_name + '_W',
                [256, 1],
                minval=-0.05, maxval=0.05)
    elif wb.lower() == 'b':
        if 1 <= layer and layer <= 12:
            return nn_util.zero_variable(scope_name + '_b', [FILTER_SIZE])
        elif layer == 13:
            return nn_util.zero_variable(scope_name + '_b', [1])
        elif layer == 14:
            return nn_util.zero_variable(scope_name + '_b', [256])
        elif layer == 15:
            return nn_util.zero_variable(scope_name + '_b', [1])


def inference(states, is_training=False):
    # convolution2d_1
    with tf.variable_scope('convolution2d_1') as scope:
        weights = get_initial_weight(1, 'w', scope.name)
        conv = tf.nn.conv2d(states, weights, [1, 1, 1, 1], padding='SAME')
        conv1 = batch_norm_relu(conv, is_training)
        # biases = get_initial_weight(1, 'b', scope.name)
        # bias_add = tf.nn.bias_add(conv, biases)
        # conv1 = tf.nn.relu(bias_add, name=scope.name)

    # convolution2d_2-12
    convi = conv1
    for i in range(2, 13):
        with tf.variable_scope('convolution2d_' + str(i)) as scope:
            weights = get_initial_weight(i, 'w', scope.name)
            conv = tf.nn.conv2d(convi, weights, [1, 1, 1, 1], padding='SAME')
            conv = batch_norm_relu(conv, is_training)
            # biases = get_initial_weight(i, 'b', scope.name)
            # bias_add = tf.nn.bias_add(conv, biases)
            # conv = tf.nn.relu(bias_add, name=scope.name)
        convi = conv

    # convolution2d_13
    with tf.variable_scope('convolution2d_13') as scope:
        weights = get_initial_weight(13, 'w', scope.name)
        conv = tf.nn.conv2d(convi, weights, [1, 1, 1, 1], padding='SAME')
        layer_13 = batch_norm_relu(conv, is_training)
        # biases = get_initial_weight(13, 'b', scope.name)
        # bias_add = tf.nn.bias_add(conv, biases)
        # layer_13 = tf.nn.relu(bias_add, name=scope.name)

    # Dense 14
    with tf.variable_scope('dense_14') as scope:
        flatten = tf.reshape(layer_13, [-1, BOARD_SIZE**2])
        weights = get_initial_weight(14, 'w', scope.name)
        biases = get_initial_weight(14, 'b', scope.name)
        layer_14 = tf.nn.relu(tf.matmul(flatten, weights) + biases)

    # output
    with tf.variable_scope('tanh') as scope:
        weights = get_initial_weight(15, 'w', scope.name)
        biases = get_initial_weight(15, 'b', scope.name)
        output = tf.nn.tanh(tf.matmul(layer_14, weights) + biases)

    return output


def loss(output, z):
    loss_op = tf.losses.mean_squared_error(z, output)
    return loss_op
