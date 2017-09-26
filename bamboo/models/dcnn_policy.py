from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

from . import nn_util


BOARD_SIZE = 19
INPUT_DEPTH = 48
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
    elif wb.lower() == 'b':
        if 1 <= layer and layer <= 12:
            return nn_util.zero_variable(scope_name + '_b', [FILTER_SIZE])
        elif layer == 13:
            return nn_util.zero_variable(scope_name + '_b', [1])
        elif layer == 14:
            return nn_util.zero_variable('Variable', [BOARD_SIZE**2])


def inference(states):
    # convolution2d_1
    with tf.variable_scope('convolution2d_1') as scope:
        weights = get_initial_weight(1, 'w', scope.name)
        biases = get_initial_weight(1, 'b', scope.name)
        conv = tf.nn.conv2d(states, weights, [1, 1, 1, 1], padding='SAME')
        bias_add = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias_add, name=scope.name)

    # convolution2d_2-12
    convi = conv1
    for i in range(2, 13):
        with tf.variable_scope('convolution2d_' + str(i)) as scope:
            weights = get_initial_weight(i, 'w', scope.name)
            biases = get_initial_weight(i, 'b', scope.name)
            conv = tf.nn.conv2d(convi, weights, [1, 1, 1, 1], padding='SAME')
            bias_add = tf.nn.bias_add(conv, biases)
            conv = tf.nn.relu(bias_add, name=scope.name)
        convi = conv

    # convolution2d_13
    with tf.variable_scope('convolution2d_13') as scope:
        weights = get_initial_weight(13, 'w', scope.name)
        biases = get_initial_weight(13, 'b', scope.name)
        conv = tf.nn.conv2d(convi, weights, [1, 1, 1, 1], padding='SAME')
        conv13 = tf.nn.bias_add(conv, biases, name=scope.name)

    # linear
    with tf.variable_scope('bias_1') as scope:
        bias = get_initial_weight(14, 'b', scope.name)
        flatten = tf.reshape(conv13, [-1, BOARD_SIZE**2])
        logits = tf.add(flatten, bias, name=scope.name)

    return logits


def loss(probs, actions):
    # Note: compute crossentropy from probs like Keras. a little better performace.
    output = probs
    output /= tf.reduce_sum(output,
                            reduction_indices=len(output.get_shape()) - 1,
                            keep_dims=True)
    output = tf.clip_by_value(output,
                              tf.cast(1e-07, dtype=tf.float32),
                              tf.cast(1. - 1e-07, dtype=tf.float32))
    output = - tf.reduce_sum(actions * tf.log(output),
                             reduction_indices=len(output.get_shape()) - 1)
    loss_op = tf.reduce_mean(output)
    """
    loss_op = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=actions_placeholder))
    """
    return loss_op


def accuracy(probs, actions):
    correct = tf.nn.in_top_k(probs, tf.argmax(actions, 1), 1)
    acc_op = tf.reduce_mean(tf.cast(correct, tf.float32))
    return acc_op
