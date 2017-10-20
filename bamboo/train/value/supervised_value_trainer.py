#!/usr/bin/env python

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import os
import re
import sys
import tensorflow as tf
import traceback

from bamboo.models import dcnn_resnet_value as value

from ..keras_utils import callbacks as cbks


flags = tf.app.flags
flags.DEFINE_string("train_data", "", "")
flags.DEFINE_string("validation_data", "", "")

flags.DEFINE_string('logdir', '/tmp/logs',
                    'Directory where to save latest parameters for playout.')
flags.DEFINE_integer('checkpoint', 100, 'Interval steps to execute checkpoint.')

flags.DEFINE_integer("epoch", 10, "")
flags.DEFINE_integer("epoch_length", None, "")
flags.DEFINE_integer("validation_length", None, "")
flags.DEFINE_integer("batch_size", 32, "")

flags.DEFINE_integer("num_threads", 1, "")
flags.DEFINE_integer("shuffle_buffer_size", 100, "")

flags.DEFINE_float("learning_rate", 3e-3, "Learning rate.")
flags.DEFINE_float("decay", .5, "")
flags.DEFINE_integer("decay_step", 80000000, "")

flags.DEFINE_float('gpu_memory_fraction', 0.2,
                   'config.per_process_gpu_memory_fraction for training session')
flags.DEFINE_boolean('log_device_placement', False, '')

flags.DEFINE_boolean('verbose', True, '')

FLAGS = flags.FLAGS

_MOMENTUM = 0.9


def run_training():

    with tf.Graph().as_default() as graph:
        global_step = tf.contrib.framework.get_or_create_global_step()

        lr = tf.train.exponential_decay(
                FLAGS.learning_rate,
                global_step,
                FLAGS.decay_step, FLAGS.decay)
        grad = tf.train.GradientDescentOptimizer(lr)

        """
        grad = tf.train.MomentumOptimizer(
            learning_rate=FLAGS.learning_rate,
            momentum=_MOMENTUM)
        """

        # features of training data
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        if FLAGS.epoch_length:
            epoch_length = FLAGS.epoch_length
        else:
            # TODO too slow
            epoch_length = 0
            for each_data in [each_data.strip() for each_data in re.split(r'[\s,]+', FLAGS.train_data)]:
                epoch_length += sum(1 for _ in tf.python_io.tf_record_iterator(each_data, options=options))

        print('Epoch length: {:d}'.format(epoch_length))
        train_filenames = [filename.strip() for filename in re.split(r'[\s,]+', FLAGS.train_data)]

        do_validation = bool(FLAGS.validation_data)
        if do_validation:
            if FLAGS.validation_length:
                validation_length = FLAGS.validation_length
            else:
                # TODO too slow
                validation_length = 0
                for each_data in [each_data.strip() for each_data in re.split(r'[\s,]+', FLAGS.validation_data)]:
                    validation_length += sum(1 for _ in tf.python_io.tf_record_iterator(each_data, options=options))
            print('Validation length: {:d}'.format(validation_length))
            validation_filenames = [filename.strip() for filename in re.split(r'[\s,]+', FLAGS.validation_data)]

        def parse_function(example_proto):
            features = {
                "state": tf.FixedLenFeature([49*19*19], tf.float32),
                "z": tf.FixedLenFeature([1], tf.float32)
            }
            parsed_features = tf.parse_single_example(example_proto, features)
            state = parsed_features['state']
            state = tf.reshape(state, (49, 19, 19))
            z = parsed_features['z']
            return state, z

        filenames = tf.placeholder(tf.string, shape=[None])
        dataset = tf.contrib.data.TFRecordDataset(filenames, compression_type='GZIP')
        dataset = dataset.map(parse_function, num_threads=FLAGS.num_threads) \
                         .shuffle(FLAGS.shuffle_buffer_size) \
                         .repeat() \
                         .batch(FLAGS.batch_size)

        iterator = dataset.make_initializable_iterator()
        (state_batch, action_batch) = iterator.get_next()

        # define computation graph
        outputs = value.inference(state_batch, is_training=True)

        loss_op = value.loss(outputs, action_batch)

        train_op = grad.minimize(loss_op, global_step=global_step)

        # create a summary for our cost
        tf.summary.scalar("loss", loss_op)
        tf.summary.scalar("learning_rate", lr)

        # merge all summaries into a single "operation" which we can execute in a session
        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()

        config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
        if FLAGS.gpu_memory_fraction:
            config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction

        sess = tf.Session(config=config, graph=graph)
        sess.run(init_op)

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(FLAGS.logdir, graph=sess.graph)
        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # prepare callbacks
        callbacks = [cbks.BaseLogger(FLAGS.logdir), cbks.History(), cbks.ProgbarLogger()]
        callbacks = cbks.CallbackList(callbacks)
        callbacks._set_params({
            'nb_epoch': FLAGS.epoch,
            'nb_sample': epoch_length,
            'verbose': FLAGS.verbose,
            'do_validation': do_validation,
            'checkpoint': FLAGS.checkpoint,
            'metrics': ['loss', 'val_loss'],
        })

        # perform training cycles
        epoch = 0
        step = 0
        reports = 0
        callbacks.on_train_begin()
        while epoch < FLAGS.epoch:
            sess.run(iterator.initializer, feed_dict={filenames: train_filenames})

            callbacks.on_epoch_begin(epoch)
            while callbacks.callbacks[0].seen < epoch_length:
                batch_logs = {"size": FLAGS.batch_size}
                callbacks.on_batch_begin(None, batch_logs)

                _, loss, step = sess.run([train_op, loss_op, global_step])

                batch_logs["loss"] = loss
                batch_logs["step"] = step

                callbacks.on_batch_end(None, batch_logs)

                try:
                    if step >= FLAGS.checkpoint * (reports+1):
                        reports += 1
                        summary = sess.run(summary_op)
                        summary_writer.add_summary(summary, global_step=step)
                        summary_writer.flush()
                except:
                    err, msg, _ = sys.exc_info()
                    sys.stderr.write("{} {}\n".format(err, msg))
                    sys.stderr.write(traceback.format_exc())

            checkpoint_file = os.path.join(FLAGS.logdir, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=epoch)

            epoch_logs = {}
            if do_validation:
                sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})
                val_losses = []
                val_seen = 0
                while val_seen < validation_length:
                    val_loss = sess.run(loss_op)
                    val_losses.append(val_loss)
                    val_seen += FLAGS.batch_size
                epoch_logs['val_loss'] = np.mean(val_losses)

            callbacks.on_epoch_end(epoch, epoch_logs)
            epoch += 1


def main(argv=None):
    run_training()


if __name__ == '__main__':
    tf.app.run(main=main)
