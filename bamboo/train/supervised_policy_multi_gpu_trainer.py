#!/usr/bin/env python

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import os
import re
import sys
import tensorflow as tf
import traceback
import warnings

from . import dcnn_policy as policy

from keras_utils import callbacks as cbks


flags = tf.app.flags
flags.DEFINE_string("train_data", "", "")
flags.DEFINE_string("validation_data", "", "")

flags.DEFINE_string('logdir', '/tmp/logs',
                    'Directory where to save latest parameters for playout.')
flags.DEFINE_integer('checkpoint', 100, 'Interval steps to execute checkpoint.')

flags.DEFINE_integer("dataset_length", None, "")
flags.DEFINE_integer("epoch", 10, "")
flags.DEFINE_integer("epoch_length", 0, "")
flags.DEFINE_integer("batch_size", 16, "")

flags.DEFINE_float("learning_rate", 3e-3, "Learning rate.")
flags.DEFINE_float("decay", .5, "")
flags.DEFINE_integer("decay_step", 80000000, "")

flags.DEFINE_float('gpu_memory_fraction', 0.15,
                   'config.per_process_gpu_memory_fraction for training session')
flags.DEFINE_integer('num_gpus', 1,
                     """ How many GPUs to use""")
flags.DEFINE_boolean('log_device_placement', False, '')

flags.DEFINE_string("symmetries", "all",
                    """none, all or comma-separated list of transforms,
                    subset of: noop,rot90,rot180,rot270,fliplr,flipud,diag1,diag2. Default: all""")

flags.DEFINE_boolean('verbose', True, '')

FLAGS = flags.FLAGS

TRANSFORMATION_INDICES = {
    "noop": 0,
    "rot90": 1,
    "rot180": 2,
    "rot270": 3,
    "fliplr": 4,
    "flipud": 5,
    "diag1": 6,
    "diag2": 7
}

BOARD_TRANSFORMATIONS = {
    0: lambda feature: feature,
    1: lambda feature: tf.image.rot90(feature, 1),
    2: lambda feature: tf.image.rot90(feature, 2),
    3: lambda feature: tf.image.rot90(feature, 3),
    4: lambda feature: tf.image.flip_left_right(feature),
    5: lambda feature: tf.image.flip_up_down(feature),
    6: lambda feature: tf.transpose(feature),
    7: lambda feature: tf.image.flip_left_right(tf.image.rot90(feature, 1))
}

MOVING_AVERAGE_DECAY = 0.9999

TOWER_NAME = 'tower'


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def tower_loss(scope, probs, actions):
    loss_op = policy.loss(probs, actions)

    tf.add_to_collection('losses', loss_op)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    tf.add_n(tf.get_collection('losses'), name='total_loss')

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name, l)

    return total_loss


def tower_acc(scope, probs, actions):
    acc_op = policy.accuracy(probs, actions)

    tf.add_to_collection('accs', acc_op)

    tf.add_n(tf.get_collection('accs'), name='total_acc')

    # Assemble all of the accuracy for the current tower only.
    accs = tf.get_collection('accs', scope)

    # Calculate the total accuray for the current tower.
    total_acc = tf.add_n(accs, name='total_acc')

    # Attach a scalar summary to all individual accs and the total acc; do the
    # same for the averaged version of the accs.
    for a in accs + [total_acc]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        acc_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', a.op.name)
        tf.summary.scalar(acc_name, a)

    return total_acc


def run_training():

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.contrib.framework.get_or_create_global_step()

        lr = tf.train.exponential_decay(
                FLAGS.learning_rate,
                global_step,
                FLAGS.decay_step, FLAGS.decay)
        opt = tf.train.GradientDescentOptimizer(lr)

        # features of training data
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        if FLAGS.dataset_length:
            dataset_length = FLAGS.dataset_length
        else:
            # TODO too slow
            dataset_length = sum(1 for _ in tf.python_io.tf_record_iterator(FLAGS.train_data, options=options))
        if FLAGS.epoch_length == 0:
            epoch_length = dataset_length
        else:
            epoch_length = FLAGS.epoch_length

        do_validation = bool(FLAGS.validation_data)

        # used symmetries
        if FLAGS.symmetries == "all":
            # add all symmetries
            symmetries = TRANSFORMATION_INDICES.values()
        elif FLAGS.symmetries == "none":
            # only add standart orientation
            symmetries = [TRANSFORMATION_INDICES["noop"]]
        else:
            # add specified symmetries
            symmetries = [TRANSFORMATION_INDICES[name] for name in FLAGS.symmetries.strip().split(",")]

        if FLAGS.verbose:
            print("Used symmetries: " + FLAGS.symmetries)

        np.random.seed(np.random.randint(1, 4294967295+1))

        features = {
            "state": tf.FixedLenFeature([48*19*19], tf.float32),
            "action": tf.FixedLenFeature([19*19], tf.float32)
        }

        def parse_function(example_proto):
            parsed_features = tf.parse_single_example(example_proto, features)

            transform = BOARD_TRANSFORMATIONS[symmetries[np.random.randint(len(symmetries))]]

            state = parsed_features['state']
            state = tf.reshape(state, (19, 19, 48))
            state = transform(state)

            action = parsed_features['action']
            action = tf.reshape(action, (19, 19, 1))
            action = transform(action)
            action = tf.reshape(action, (361,))

            return state, action

        filenames = tf.placeholder(tf.string, shape=[None])

        # Calculate the gradients for each model tower.
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
                        dataset = tf.contrib.data.TFRecordDataset(filenames, compression_type='GZIP')
                        dataset = dataset.map(parse_function).shuffle(16).repeat().batch(16)
                        iterator = dataset.make_initializable_iterator()
                        (state_batch, action_batch) = iterator.get_next()

                        logits = policy.inference(state_batch)

                        probs = tf.nn.softmax(logits)

                        loss_op = tower_loss(scope, probs, action_batch)
                        acc_op = tower_acc(scope, probs, action_batch)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = opt.compute_gradients(loss_op)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Add a summary to track the learning rate.
        summaries.append(tf.summary.scalar('learning_rate', lr))

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement)
        config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction

        sess = tf.Session(config=config)
        sess.run(init)

        sess.run(iterator.initializer, feed_dict={filenames: [FLAGS.train_data]})

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)

        # prepare callbacks
        callbacks = [cbks.BaseLogger(), cbks.History(), cbks.ProgbarLogger()]
        callbacks = cbks.CallbackList(callbacks)
        callbacks._set_params({
            'nb_epoch': FLAGS.epoch,
            'nb_sample': epoch_length,
            'verbose': FLAGS.verbose,
            'do_validation': do_validation,
            'metrics': ['loss', 'acc', 'val_loss', 'val_acc'],
        })

        # perform training cycles
        epoch = 0
        step = 0
        reports = 0
        callbacks.on_train_begin()
        while epoch < FLAGS.epoch:
            callbacks.on_epoch_begin(epoch)
            samples_seen = 0
            batch_index = 0
            while samples_seen < epoch_length:
                # build batch logs
                batch_logs = {"batch": batch_index, "size": FLAGS.batch_size}
                callbacks.on_batch_begin(batch_index, batch_logs)

                _, loss, acc, summary, step = sess.run(
                    [train_op, loss_op, acc_op, summary_op, global_step])

                batch_logs["loss"] = loss
                batch_logs["acc"] = acc
                callbacks.on_batch_end(batch_index, batch_logs)

                # construct epoch logs
                epoch_logs = {}
                batch_index += 1
                samples_seen += FLAGS.batch_size

                try:
                    if step >= FLAGS.checkpoint * (reports+1):
                        reports += 1
                        summary_writer.add_summary(summary, global_step=step)
                        summary_writer.flush()
                except:
                    err, msg, _ = sys.exc_info()
                    sys.stderr.write("{} {}\n".format(err, msg))
                    sys.stderr.write(traceback.format_exc())

            # evaluate the model
            # TODO

            checkpoint_file = os.path.join(FLAGS.logdir, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=step)

            callbacks.on_epoch_end(epoch, epoch_logs)
            epoch += 1


def main(argv=None):
    run_training()


if __name__ == '__main__':
    tf.app.run(main=main)
