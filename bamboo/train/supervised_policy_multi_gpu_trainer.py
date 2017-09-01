#!/usr/bin/env python

from __future__ import print_function
from __future__ import absolute_import

import h5py as h5
import threading
import numpy as np
import os
try:
    import queue
except ImportError:
    import Queue as queue
import re
import sys
import tensorflow as tf
import time
import traceback
import warnings

from . import dcnn_policy as policy

from keras import callbacks as cbks


flags = tf.app.flags
flags.DEFINE_string("train_data", "", "")

flags.DEFINE_string('logdir', '/tmp/logs',
                    'Directory where to save latest parameters for playout.')
flags.DEFINE_integer('checkpoint', 100, 'Interval steps to execute checkpoint.')

flags.DEFINE_integer("batch_size", 16, "")
flags.DEFINE_integer("epoch", 10, "")
flags.DEFINE_integer("epoch_length", 0, "")
flags.DEFINE_integer("max_validation", 1000000000, "")
flags.DEFINE_float("validation_size", .05, "")
flags.DEFINE_float("test_size", .0, "")

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

# shuffle files
FILE_VALIDATE = 'shuffle_policy_validate.npz'
FILE_TRAIN = 'shuffle_policy_train.npz'
FILE_TEST = 'shuffle_policy_test.npz'

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
    1: lambda feature: np.rot90(feature, 1),
    2: lambda feature: np.rot90(feature, 2),
    3: lambda feature: np.rot90(feature, 3),
    4: lambda feature: np.fliplr(feature),
    5: lambda feature: np.flipud(feature),
    6: lambda feature: np.transpose(feature),
    7: lambda feature: np.fliplr(np.rot90(feature, 1))
}

MOVING_AVERAGE_DECAY = 0.9999

TOWER_NAME = 'tower'


def generator_queue(generator, max_q_size=10,
                    wait_time=0.05, nb_worker=1):
    '''Builds a threading queue out of a data generator.
    Used in `fit_generator`, `evaluate_generator`.
    '''
    q = queue.Queue()
    _stop = threading.Event()

    def data_generator_task():
        while not _stop.is_set():
            try:
                if q.qsize() < max_q_size:
                    try:
                        generator_output = next(generator)
                    except ValueError:
                        continue
                    q.put(generator_output)
                else:
                    time.sleep(wait_time)
            except Exception:
                _stop.set()
                raise

    generator_threads = [threading.Thread(target=data_generator_task)
                         for _ in range(nb_worker)]

    for thread in generator_threads:
        thread.daemon = True
        thread.start()

    return q, _stop


def one_hot_action(action, size=19):
    """Convert an (x,y) action into a size x size array of zeros with a 1 at x,y
    """

    categorical = np.zeros((size, size))
    categorical[action] = 1
    return categorical


class threading_shuffled_hdf5_batch_generator:
    """A generator of batches of training data for use with the fit_generator function
       of Keras. Data is accessed in the order of the given indices for shuffling.

       it is threading safe but not multiprocessing therefore only use it with
       pickle_safe=False when using multiple workers
    """

    def shuffle_indices(self, seed=None, idx=0):
        # set generator_sample to idx
        self.metadata['generator_sample'] = idx

        # check if seed is provided or generate random
        if seed is None:
            # create random seed
            self.metadata['generator_seed'] = np.random.random_integers(4294967295)

        # feed numpy.random with seed in order to continue with certain batch
        np.random.seed(self.metadata['generator_seed'])
        # shuffle indices according to seed
        if not self.validation:
            np.random.shuffle(self.indices)

    def __init__(self, state_dataset, action_dataset, indices, batch_size, metadata=None,
                 validation=False):
        self.action_dataset = action_dataset
        self.state_dataset = state_dataset
        # lock used for multithreaded workers
        self.data_lock = threading.Lock()
        self.indices_max = len(indices)
        self.validation = validation
        self.batch_size = batch_size
        self.indices = indices

        if metadata is not None:
            self.metadata = metadata
        else:
            # create metadata object
            self.metadata = {
                "generator_seed": None,
                "generator_sample": 0
            }

        # shuffle indices
        # when restarting generator_seed and generator_batch will
        # reset generator to the same point as before
        self.shuffle_indices(self.metadata['generator_seed'], self.metadata['generator_sample'])

    def __iter__(self):
        return self

    def next_indice(self):
        # use lock to prevent double hdf5 acces and incorrect generator_sample increment
        with self.data_lock:

            # get next training sample
            training_sample = self.indices[self.metadata['generator_sample'], :]
            # get state
            state = self.state_dataset[training_sample[0]]
            # get action
            # must be cast to a tuple so that it is interpreted as (x,y) not [(x,:), (y,:)]
            action = tuple(self.action_dataset[training_sample[0]])

            # increment generator_sample
            self.metadata['generator_sample'] += 1
            # shuffle indices when all have been used
            if self.metadata['generator_sample'] >= self.indices_max:
                self.shuffle_indices()

            # return state, action and transformation
            return state, action, training_sample[1]

    def next(self):
        state_batch_shape = (self.batch_size,) + self.state_dataset.shape[2:] + self.state_dataset.shape[1:2]
        game_size = state_batch_shape[1]
        Xbatch = np.zeros(state_batch_shape)
        Ybatch = np.zeros((self.batch_size, game_size * game_size))

        for batch_idx in xrange(self.batch_size):
            state, action, transformation = self.next_indice()

            # get rotation symmetry belonging to state
            transform = BOARD_TRANSFORMATIONS[transformation]

            # get state from dataset and transform it.
            # loop comprehension is used so that the transformation acts on the
            # 3rd and 4th dimensions
            state_transform = np.array([transform(plane) for plane in state])
            action_transform = transform(one_hot_action(action, game_size))

            # Transpose input(state) dimention ordering.
            # TF uses the last dimension as channel dimension,
            # K input shape: (samples, input_depth, row, cols)
            # TF input shape: (samples, rows, cols, input_depth)
            Xbatch[batch_idx] = state_transform.transpose((1, 2, 0))
            Ybatch[batch_idx] = action_transform.flatten()

        return (Xbatch, Ybatch)


def load_indices_from_file(shuffle_file):
    # load indices from shuffle_file
    with open(shuffle_file, "r") as f:
        indices = np.load(f)

    return indices


def save_indices_to_file(shuffle_file, indices):
    # save indices to shuffle_file
    with open(shuffle_file, "w") as f:
        np.save(f, indices)


def remove_unused_symmetries(indices, symmetries):
    # remove all rows with a symmetry not in symmetries
    remove = []

    # find all rows with incorrect symmetries
    for row in range(len(indices)):
        if not indices[row][1] in symmetries:
            remove.append(row)

    # remove rows and return new array
    return np.delete(indices, remove, 0)


def load_train_val_test_indices(verbose, arg_symmetries, dataset_length, batch_size, directory):
    """Load indices from .npz files
       Remove unwanted symmerties
       Make Train set dividable by batch_size
       Return train/val/test set
    """
    # shuffle file locations for train/validation/test set
    shuffle_file_train = os.path.join(directory, FILE_TRAIN)
    shuffle_file_val = os.path.join(directory, FILE_VALIDATE)
    shuffle_file_test = os.path.join(directory, FILE_TEST)

    # load from .npz files
    train_indices = load_indices_from_file(shuffle_file_train)
    val_indices = load_indices_from_file(shuffle_file_val)
    test_indices = load_indices_from_file(shuffle_file_test)

    # used symmetries
    if arg_symmetries == "all":
        # add all symmetries
        symmetries = TRANSFORMATION_INDICES.values()
    elif arg_symmetries == "none":
        # only add standart orientation
        symmetries = [TRANSFORMATION_INDICES["noop"]]
    else:
        # add specified symmetries
        symmetries = [TRANSFORMATION_INDICES[name] for name in arg_symmetries.strip().split(",")]

    if verbose:
        print("Used symmetries: " + arg_symmetries)

    # remove symmetries not used during current run
    if len(symmetries) != len(TRANSFORMATION_INDICES):
        train_indices = remove_unused_symmetries(train_indices, symmetries)
        test_indices = remove_unused_symmetries(test_indices, symmetries)
        val_indices = remove_unused_symmetries(val_indices, symmetries)

    # Need to make sure training data is dividable by minibatch size or get
    # warning mentioning accuracy from keras
    if len(train_indices) % batch_size != 0:
        # remove first len(train_indices) % args.minibatch rows
        train_indices = np.delete(train_indices, [row for row in range(len(train_indices)
                                                  % batch_size)], 0)

    if verbose:
        print("dataset loaded")
        print("\t%d total positions" % dataset_length)
        print("\t%d total samples" % (dataset_length * len(symmetries)))
        print("\t%d total samples check" % (len(train_indices) +
              len(val_indices) + len(test_indices)))
        print("\t%d training samples" % len(train_indices))
        print("\t%d validation samples" % len(val_indices))
        print("\t%d test samples" % len(test_indices))

    return train_indices, val_indices, test_indices


def create_and_save_shuffle_indices(n_total_data_size, max_validation,
                                    shuffle_file_train, shuffle_file_val, shuffle_file_test):
    """ create an array with all unique state and symmetry pairs,
        calculate test/validation/training set sizes,
        seperate those sets and save them to seperate files.
    """

    symmetries = TRANSFORMATION_INDICES.values()

    # Create an array with a unique row for each combination of a training example
    # and a symmetry.
    # shuffle_indices[i][0] is an index into training examples,
    # shuffle_indices[i][1] is the index (from 0 to 7) of the symmetry transformation to apply
    shuffle_indices = np.empty(shape=[n_total_data_size * len(symmetries), 2], dtype=int)
    for dataset_idx in range(n_total_data_size):
        for symmetry_idx in range(len(symmetries)):
            shuffle_indices[dataset_idx * len(symmetries) + symmetry_idx][0] = dataset_idx
            shuffle_indices[dataset_idx * len(symmetries) +
                            symmetry_idx][1] = symmetries[symmetry_idx]

    # shuffle rows without affecting x,y pairs
    np.random.shuffle(shuffle_indices)

    # validation set size
    n_val_data = int(FLAGS.validation_size * len(shuffle_indices))
    # limit validation set to --max-validation
    if n_val_data > max_validation:
        n_val_data = max_validation

    # test set size
    n_test_data = int(FLAGS.test_size * len(shuffle_indices))

    # train set size
    n_train_data = len(shuffle_indices) - n_val_data - n_test_data

    # create training set and save to file shuffle_file_train
    train_indices = shuffle_indices[0:n_train_data]
    save_indices_to_file(shuffle_file_train, train_indices)

    # create validation set and save to file shuffle_file_val
    val_indices = shuffle_indices[n_train_data:n_train_data + n_val_data]
    save_indices_to_file(shuffle_file_val, val_indices)

    # create test set and save to file shuffle_file_test
    test_indices = shuffle_indices[n_train_data + n_val_data:
                                   n_train_data + n_val_data + n_test_data]
    save_indices_to_file(shuffle_file_test, test_indices)


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

        states_placeholder = policy.get_states()
        actions_placeholder = policy.get_actions()

        lr = tf.train.exponential_decay(
                FLAGS.learning_rate,
                global_step,
                FLAGS.decay_step, FLAGS.decay)
        opt = tf.train.GradientDescentOptimizer(lr)

        # features of training data
        dataset = h5.File(FLAGS.train_data)
        dataset_length = len(dataset["states"])
        if FLAGS.epoch_length == 0:
            epoch_length = dataset_length
        else:
            epoch_length = FLAGS.epoch_length

        do_validation = bool(FLAGS.validation_size)

        # shuffle file locations for train/validation/test set
        shuffle_file_train = os.path.join(FLAGS.logdir, FILE_TRAIN)
        shuffle_file_val = os.path.join(FLAGS.logdir, FILE_VALIDATE)
        shuffle_file_test = os.path.join(FLAGS.logdir, FILE_TEST)

        # create and save new shuffle indices to file
        create_and_save_shuffle_indices(
                dataset_length, FLAGS.max_validation,
                shuffle_file_train, shuffle_file_val, shuffle_file_test)

        print("Created new data shuffling indices")

        # get train/validation/test indices
        train_indices, val_indices, test_indices \
            = load_train_val_test_indices(FLAGS.verbose, FLAGS.symmetries, dataset_length,
                                          FLAGS.batch_size, FLAGS.logdir)

        train_data_generator = threading_shuffled_hdf5_batch_generator(
            dataset['states'],
            dataset['actions'],
            train_indices,
            FLAGS.batch_size,
            )
        val_data_generator = threading_shuffled_hdf5_batch_generator(
            dataset["states"],
            dataset["actions"],
            val_indices,
            FLAGS.batch_size,
            validation=do_validation)

        # start generator thread storing batches into a queue
        train_data_gen_queue, _train_stop = generator_queue(train_data_generator)

        # Calculate the gradients for each model tower.
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
                        logits = policy.inference(states_placeholder)

                        probs = tf.nn.softmax(logits)

                        loss_op = tower_loss(scope, probs, actions_placeholder)
                        acc_op = tower_acc(scope, probs, actions_placeholder)

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
        wait_time = 0.01  # in seconds
        epoch = 0
        step = 0
        reports = 0
        callbacks.on_train_begin()
        while epoch < FLAGS.epoch:
            callbacks.on_epoch_begin(epoch)
            samples_seen = 0
            batch_index = 0
            while samples_seen < epoch_length:
                while not _train_stop.is_set():
                    if not train_data_gen_queue.empty():
                        states, actions = train_data_gen_queue.get()
                        break
                    else:
                        time.sleep(wait_time)

                batch_size = len(states[0])
                # build batch logs
                batch_logs = {"batch": batch_index, "size": batch_size}
                callbacks.on_batch_begin(batch_index, batch_logs)

                _, loss, acc, summary, step = sess.run(
                        [train_op, loss_op, acc_op, summary_op, global_step],
                        feed_dict={
                            states_placeholder: states,
                            actions_placeholder: actions
                        })

                batch_logs["loss"] = loss
                batch_logs["acc"] = acc
                callbacks.on_batch_end(batch_index, batch_logs)

                # construct epoch logs
                epoch_logs = {}
                batch_index += 1
                samples_seen += batch_size

                # epoch finished
                if samples_seen > epoch_length:
                    warnings.warn('Epoch comprised more than '
                                  '`epoch_length` samples, '
                                  'which might affect learning results. '
                                  'Set `epoch_length` correctly '
                                  'to avoid this warning.')

                # evaluate the model
                if samples_seen >= epoch_length and do_validation:
                    processed_samples = 0
                    val_losses = []
                    val_accs = []
                    val_data_gen_queue, _val_stop = generator_queue(val_data_generator)
                    while processed_samples < len(val_indices):
                        while not _val_stop.is_set():
                            if not val_data_gen_queue.empty():
                                states, actions = val_data_gen_queue.get()
                                break
                            else:
                                time.sleep(wait_time)
                        loss, acc = sess.run(
                            [loss_op, acc_op],
                            feed_dict={
                                states_placeholder: states,
                                actions_placeholder: actions
                            })
                        val_losses.append(loss)
                        val_accs.append(acc)
                        processed_samples += len(states[0])
                    _val_stop.set()
                    epoch_logs['val_loss'] = np.mean(val_losses)
                    epoch_logs['val_acc'] = np.mean(val_accs)

                try:
                    if step >= FLAGS.checkpoint * (reports+1):
                        reports += 1
                        summary_writer.add_summary(summary, global_step=step)
                        summary_writer.flush()
                except:
                    err, msg, _ = sys.exc_info()
                    sys.stderr.write("{} {}\n".format(err, msg))
                    sys.stderr.write(traceback.format_exc())

            checkpoint_file = os.path.join(FLAGS.logdir, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=step)

            callbacks.on_epoch_end(epoch, epoch_logs)
            epoch += 1


def main(argv=None):
    run_training()


if __name__ == '__main__':
    tf.app.run(main=main)
