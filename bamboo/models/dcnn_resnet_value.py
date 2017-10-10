from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

from .resnet import conv2d_fixed_padding, block_layer, building_block, batch_norm_relu


def inference(states, data_format='channels_last', is_training=False):
    with tf.variable_scope('initial_conv') as scope:
        inputs = conv2d_fixed_padding(
            inputs=states, filters=64, kernel_size=5, strides=1,
            data_format=data_format)
        inputs = tf.identity(inputs, scope.name)

    with tf.variable_scope('block_layer1') as scope:
        inputs = block_layer(
            inputs=inputs, filters=64, block_fn=building_block, blocks=6,
            strides=1, is_training=is_training, name=scope.name,
            data_format=data_format)

    with tf.variable_scope('final_conv'):
        inputs = batch_norm_relu(inputs, is_training, data_format)
        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=1, kernel_size=1, strides=1,
            data_format=data_format)
        inputs = tf.identity(inputs, scope.name)

    with tf.variable_scope('dense_256'):
        inputs = tf.reshape(inputs, [-1, inputs.shape[2].value**2])
        inputs = tf.layers.dense(inputs=inputs, units=256, activation=tf.nn.relu)
        inputs = tf.identity(inputs, scope.name)

    with tf.variable_scope('final_activation'):
        inputs = tf.layers.dense(inputs=inputs, units=1)
        inputs = tf.nn.tanh(inputs)
        inputs = tf.identity(inputs, scope.name)

    return inputs


def loss(output, z):
    loss_op = tf.losses.mean_squared_error(z, output)
    return loss_op
