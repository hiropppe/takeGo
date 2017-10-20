from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

from .resnet import conv2d_fixed_padding, block_layer, building_block, batch_norm_relu


def inference(states, data_format='channels_first', is_training=False):
    with tf.variable_scope('initial_conv') as scope:
        inputs = conv2d_fixed_padding(
            inputs=states, filters=64, kernel_size=5, strides=1,
            data_format=data_format)
        inputs = tf.identity(inputs, scope.name)

    with tf.variable_scope('block_layer1') as scope:
        inputs = block_layer(
            inputs=inputs, filters=64, block_fn=building_block, blocks=2,
            strides=1, is_training=is_training, name=scope.name,
            data_format=data_format)

    with tf.variable_scope('block_layer2') as scope:
        inputs = block_layer(
            inputs=inputs, filters=128, block_fn=building_block, blocks=2,
            strides=2, is_training=is_training, name=scope.name,
            data_format=data_format)

    with tf.variable_scope('block_layer3') as scope:
        inputs = block_layer(
            inputs=inputs, filters=256, block_fn=building_block, blocks=2,
            strides=2, is_training=is_training, name=scope.name,
            data_format=data_format)

    with tf.variable_scope('block_layer4') as scope:
        inputs = block_layer(
            inputs=inputs, filters=512, block_fn=building_block, blocks=2,
            strides=2, is_training=is_training, name=scope.name,
            data_format=data_format)

    with tf.variable_scope('final_conv'):
        inputs = batch_norm_relu(inputs, is_training, data_format)
        inputs = tf.layers.average_pooling2d(
            inputs=inputs, pool_size=3, strides=1, padding='VALID',
            data_format=data_format)
        inputs = tf.identity(inputs, scope.name)

    with tf.variable_scope('final_activation'):
        inputs = tf.reshape(inputs, [-1, 512])
        inputs = tf.layers.dense(inputs=inputs, units=1)
        inputs = tf.nn.tanh(inputs)
        inputs = tf.identity(inputs, scope.name)

    return inputs


def loss(output, z):
    loss_op = tf.losses.mean_squared_error(z, output)
    return loss_op
