from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

from . import nn_util


BOARD_SIZE = 19
INPUT_DEPTH = 49
FILTER_SIZE = 192
FILTER_WIDTH_1 = 5
FILTER_WIDTH_2_12 = 3


def get_states():
    return tf.placeholder(
                tf.float32,
                shape=(None, BOARD_SIZE, BOARD_SIZE, INPUT_DEPTH))


def get_actions():
    return tf.placeholder(
                tf.float32,
                shape=(None, BOARD_SIZE**2))


def get_initial_weight(layer, wb, scope_name):
    if wb.lower() == 'w':
        if layer == 1:
            return nn_util.random_uniform(
                scope_name + '_W',
                [FILTER_WIDTH_1, FILTER_WIDTH_1, INPUT_DEPTH, FILTER_SIZE],
                minval=-0.05, maxval=0.05)
        elif layer <= 13:
            return nn_util.random_uniform(
                scope_name + '_W',
                [FILTER_WIDTH_2_12, FILTER_WIDTH_2_12, FILTER_SIZE, FILTER_SIZE],
                minval=-0.05, maxval=0.05)
        elif layer == 14:
            return nn_util.random_uniform(
                scope_name + '_W',
                [1, 1, FILTER_SIZE, 1],
                minval=-0.05, maxval=0.05)
        elif layer == 15:
            return nn_util.random_uniform(
                scope_name + '_W',
                [361, 256],
                minval=-0.05, maxval=0.05)
        elif layer == 16:
            return nn_util.random_uniform(
                scope_name + '_W',
                [256, 1],
                minval=-0.05, maxval=0.05)
    elif wb.lower() == 'b':
        if 1 <= layer and layer <= 13:
            return nn_util.zero_variable(scope_name + '_b', [FILTER_SIZE])
        elif layer == 14:
            return nn_util.zero_variable(scope_name + '_b', [1])
        elif layer == 15:
            return nn_util.zero_variable(scope_name + '_b', [256])
        elif layer == 16:
            return nn_util.zero_variable(scope_name + '_b', [1])


def inference(states):
    # convolution2d_1
    with tf.variable_scope('convolution2d_1') as scope:
        weights = get_initial_weight(1, 'w', scope.name)
        biases = get_initial_weight(1, 'b', scope.name)
        conv = tf.nn.conv2d(states, weights, [1, 1, 1, 1], padding='SAME')
        bias_add = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias_add, name=scope.name)

    # convolution2d_2-13
    convi = conv1
    for i in range(2, 14):
        with tf.variable_scope('convolution2d_' + str(i)) as scope:
            weights = get_initial_weight(i, 'w', scope.name)
            biases = get_initial_weight(i, 'b', scope.name)
            conv = tf.nn.conv2d(convi, weights, [1, 1, 1, 1], padding='SAME')
            bias_add = tf.nn.bias_add(conv, biases)
            conv = tf.nn.relu(bias_add, name=scope.name)
        convi = conv

    # convolution2d_14
    with tf.variable_scope('convolution2d_14') as scope:
        weights = get_initial_weight(14, 'w', scope.name)
        biases = get_initial_weight(14, 'b', scope.name)
        conv = tf.nn.conv2d(convi, weights, [1, 1, 1, 1], padding='SAME')
        bias_add = tf.nn.bias_add(conv, biases)
        layer_14 = tf.nn.relu(bias_add, name=scope.name)

    # Dense 15
    with tf.variable_scope('dense_15') as scope:
        flatten = tf.reshape(layer_14, [-1, BOARD_SIZE**2])
        weights = get_initial_weight(15, 'w', scope.name)
        biases = get_initial_weight(15, 'b', scope.name)
        layer_15 = tf.nn.relu(tf.matmul(flatten, weights) + biases)

    # output
    with tf.variable_scope('tanh') as scope:
        weights = get_initial_weight(16, 'w', scope.name)
        biases = get_initial_weight(16, 'b', scope.name)
        output = tf.nn.tanh(tf.matmul(layer_15, weights) + biases)

    return output


def loss(output, z):
    loss_op = tf.losses.mean_squared_error(output, z)
    return loss_op
