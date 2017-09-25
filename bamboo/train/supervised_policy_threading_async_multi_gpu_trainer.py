#!/usr/bin/env python

from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import tensorflow as tf
import threading
import traceback

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
flags.DEFINE_integer("num_train_threads", 1, "")
flags.DEFINE_integer("num_data_threads", 1, "")
flags.DEFINE_integer("shuffle_buffer_size", 100, "")

flags.DEFINE_float("learning_rate", 3e-3, "Learning rate.")
flags.DEFINE_float("decay", .5, "")
flags.DEFINE_integer("decay_step", 80000000, "")

flags.DEFINE_float('gpu_memory_fraction', None,
                   'config.per_process_gpu_memory_fraction for training session')
flags.DEFINE_integer('num_gpus', 1,
                     """ How many GPUs to use""")
flags.DEFINE_boolean('log_device_placement', False, '')

flags.DEFINE_boolean('verbose', True, '')

FLAGS = flags.FLAGS

TOWER_NAME = 'tower'


def run_training():

    # with tf.Graph().as_default() as graph:
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.contrib.framework.get_or_create_global_step()

        lr = tf.train.exponential_decay(
                FLAGS.learning_rate,
                global_step,
                FLAGS.decay_step, FLAGS.decay)
        grad = tf.train.GradientDescentOptimizer(lr)

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

        def parse_function(example_proto):
            features = {
                "state": tf.FixedLenFeature([48*19*19], tf.float32),
                "action": tf.FixedLenFeature([19*19], tf.float32)
            }
            parsed_features = tf.parse_single_example(example_proto, features)
            state = parsed_features['state']
            state = tf.reshape(state, (19, 19, 48))
            action = parsed_features['action']
            return state, action

        filenames = tf.placeholder(tf.string, shape=[None])
        dataset = tf.contrib.data.TFRecordDataset(filenames, compression_type='GZIP')
        dataset = dataset.map(parse_function, num_threads=FLAGS.num_data_threads) \
                         .shuffle(FLAGS.shuffle_buffer_size) \
                         .repeat() \
                         .batch(FLAGS.batch_size)

        iterator = dataset.make_initializable_iterator()
        (state_batch, action_batch) = iterator.get_next()

        gpu_ops = []
        # define computation graph for each gpu
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (TOWER_NAME, i)):
                        logits = policy.inference(state_batch)

                        probs = tf.nn.softmax(logits)

                        loss_op = policy.loss(probs, action_batch)
                        acc_op = policy.accuracy(probs, action_batch)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        train_op = grad.minimize(loss_op, global_step=global_step)

                        gpu_ops.append((train_op, loss_op, acc_op))

        # create a summary for our cost and accuracy
        tf.summary.scalar("loss", loss_op)
        tf.summary.scalar("accuracy", acc_op)
        tf.summary.scalar("learning_rate", lr)

        # merge all summaries into a single "operation" which we can execute in a session
        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()

        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=FLAGS.log_device_placement)
        if FLAGS.gpu_memory_fraction:
            config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction

        sess = tf.Session(config=config)
        sess.run(iterator.initializer, feed_dict={filenames: [FLAGS.train_data]})
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
            'metrics': ['loss', 'acc', 'val_loss', 'val_acc'],
        })

        # perform training cycles
        epoch = 0
        callbacks.on_train_begin()
        while epoch < FLAGS.epoch:
            callbacks.on_epoch_begin(epoch)

            if FLAGS.num_train_threads == 1:
                train(sess,
                      train_op,
                      loss_op,
                      acc_op,
                      global_step,
                      epoch_length,
                      callbacks,
                      summary_writer,
                      summary_op,
                      is_chief=True)
            else:
                train_threads = []
                for i in range(FLAGS.num_train_threads):
                    if i == 0:
                        is_chief = True
                    else:
                        is_chief = False

                    ops = gpu_ops[i % FLAGS.num_gpus]

                    train_args = (sess, ops[0], ops[1], ops[2], global_step, epoch_length, callbacks, summary_writer, summary_op, is_chief)
                    train_thread = threading.Thread(name='train_thread_%d'.format(i),
                                                    target=train,
                                                    args=train_args)
                    train_threads.append(train_thread)

                for tt in train_threads:
                    tt.start()

                for tt in train_threads:
                    tt.join()

            checkpoint_file = os.path.join(FLAGS.logdir, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=epoch)

            callbacks.on_epoch_end(epoch)
            epoch += 1


def train(sess,
          train_op,
          loss_op,
          acc_op,
          global_step,
          epoch_length,
          callbacks,
          summary_writer,
          summary_op,
          is_chief=False):
    reports = 0
    while callbacks.callbacks[0].seen < epoch_length:
        batch_logs = {"size": FLAGS.batch_size}
        callbacks.on_batch_begin(None, batch_logs)

        _, loss, acc, step = sess.run([train_op, loss_op, acc_op, global_step])

        batch_logs["loss"] = loss
        batch_logs["acc"] = acc
        batch_logs["step"] = step

        callbacks.on_batch_end(None, batch_logs)

        if is_chief:
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


def main(argv=None):
    run_training()


if __name__ == '__main__':
    tf.app.run(main=main)
