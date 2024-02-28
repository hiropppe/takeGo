import numpy as np
import tensorflow as tf


def zero_variable(name, shape):
    initializer = tf.constant_initializer(0.0)
    return tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)


def random_uniform(name, shape, minval=0, maxval=None, seed=None, dtype=tf.float32):
    initializer = tf.random_uniform_initializer(minval=minval, maxval=maxval, seed=seed, dtype=dtype)
    return tf.get_variable(name, shape, initializer=initializer, dtype=dtype)


def xavier_conv2d(name, shape, uniform=True, seed=None, dtype=tf.float32):
    initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=uniform, seed=seed, dtype=dtype)
    var = tf.get_variable(name,
                          shape,
                          initializer=initializer,
                          dtype=dtype)
    return var


def variable_with_weight_decay(name, shape, stddev=5e-2, wd=0.0):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
    Returns:
        Variable Tensor
    """
    dtype = tf.float32
    initializer = tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
    var = tf.get_variable(name,
                          shape,
                          initializer=initializer,
                          dtype=dtype)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var
