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
import sys
import tensorflow as tf
import time
import traceback

from bamboo.train import nn_util

flags = tf.app.flags
flags.DEFINE_string("cluster_spec", "/cluster", "Cluster specification")
flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")

flags.DEFINE_string("train_data", "", "")

flags.DEFINE_integer("bsize", 19, "")
flags.DEFINE_integer("input_depth", 48, "")
flags.DEFINE_integer("filter_size", 192, "")
flags.DEFINE_integer("filter_width_1", 5, "")
flags.DEFINE_integer("filter_width_2_12", 3, "")

flags.DEFINE_string('logdir', '/tmp/logs',
                    'Directory where to save latest parameters for playout.')
flags.DEFINE_integer('checkpoint', 5, 'Interval steps to execute checkpoint.')

flags.DEFINE_integer("batch_size", 16, "")
flags.DEFINE_integer("epoch", 10, "")
flags.DEFINE_integer("epoch_length", 0, "")
flags.DEFINE_integer("max_validation", 1000000000, "")
flags.DEFINE_float("validation_size", .0, "")
flags.DEFINE_float("test_size", .0, "")

flags.DEFINE_float("learning_rate", 3e-4, "Learning rate.")
flags.DEFINE_float("decay", .5, "")
flags.DEFINE_integer("decay_step", 80000000, "")

flags.DEFINE_float('gpu_memory_fraction', 0.15,
                   'config.per_process_gpu_memory_fraction for training session')
flags.DEFINE_boolean('log_device_placement', False, '')

flags.DEFINE_string("symmetries", "all", "none, all or comma-separated list of transforms, subset of: noop,rot90,rot180,rot270,fliplr,flipud,diag1,diag2. Default: all")

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
        state_batch_shape = (self.batch_size,) + self.state_dataset.shape[1:]
        game_size = state_batch_shape[-1]
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

            Xbatch[batch_idx] = state_transform
            Ybatch[batch_idx] = action_transform.flatten()


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


def get_initial_weight(layer, wb, scope_name):
    if wb.lower() == 'w':
        if layer == 1:
            return nn_util.variable_with_weight_decay(
                scope_name + '_W',
                [FLAGS.filter_width_1, FLAGS.filter_width_1, FLAGS.input_depth, FLAGS.filter_size])
        elif layer <= 12:
            return nn_util.variable_with_weight_decay(
                scope_name + '_W',
                [FLAGS.filter_width_2_12, FLAGS.filter_width_2_12, FLAGS.filter_size, FLAGS.filter_size])
        elif layer == 13:
            return nn_util.variable_with_weight_decay(
                scope_name + '_W',
                [1, 1, FLAGS.filter_size, 1])
    elif wb.lower() == 'b':
        if 1 <= layer and layer <= 12:
            return nn_util.zero_variable(scope_name + '_b', [FLAGS.filter_size])
        elif layer == 13:
            return nn_util.zero_variable(scope_name + '_b', [1])
        elif layer == 14:
            return nn_util.zero_variable('Variable', [FLAGS.bsize**2])


def run_training(cluster, server, num_workers):

    # Assigns ops to the local worker by default.
    # Between-graph replication
    with tf.device(tf.train.replica_device_setter(
                   worker_device="/job:worker/task:{:d}".format(FLAGS.task_index),
                   cluster=cluster)):
        # count the number of updates
        global_step = tf.contrib.framework.get_or_create_global_step()

        states_placeholder = tf.placeholder(tf.float32,
                                            shape=(None, FLAGS.bsize, FLAGS.bsize, FLAGS.input_depth))
        actions_placeholder = tf.placeholder(tf.float32,
                                             shape=(None, FLAGS.bsize**2))

        # convolution2d_1
        with tf.variable_scope('convolution2d_1') as scope:
            weights = get_initial_weight(1, 'w', scope.name)
            biases = get_initial_weight(1, 'b', scope.name)
            conv = tf.nn.conv2d(states_placeholder, weights, [1, 1, 1, 1], padding='SAME')
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
            flatten = tf.reshape(conv13, [-1, FLAGS.bsize**2])
            logits = tf.add(flatten, bias, name=scope.name)

        # softmax
        with tf.variable_scope('softmax') as scope:
            probs = tf.nn.softmax(logits, name=scope.name)

        # loss
        with tf.variable_scope('loss') as scope:
            loss_op = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(
                            logits=logits, labels=actions_placeholder), name=scope.name)

        # accuracy
        with tf.variable_scope('accuracy') as scope:
            correct = tf.nn.in_top_k(probs, tf.argmax(actions_placeholder, 1), 1)
            acc_op = tf.reduce_mean(tf.cast(correct, tf.float32), name=scope.name)

        # specify replicas optimizer
        with tf.name_scope('dist_train'):
            learning_rate_op = tf.train.exponential_decay(
                    FLAGS.learning_rate,
                    global_step,
                    FLAGS.decay_step, FLAGS.decay)
            grad = tf.train.GradientDescentOptimizer(learning_rate_op)
            train_op = grad.minimize(loss_op, global_step=global_step)

        # create a summary for our cost and accuracy
        tf.summary.scalar("loss", loss_op)
        tf.summary.scalar("accuracy", acc_op)

        # merge all summaries into a single "operation" which we can execute in a session
        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()

        print("Variables initialized ...")

        # features of training data
        dataset = h5.File(FLAGS.train_data)
        dataset_length = len(dataset["states"])
        if FLAGS.epoch_length == 0:
            epoch_length = dataset_length
        else:
            epoch_length = dataset_length

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
        """
        val_data_generator = threading_shuffled_hdf5_batch_generator(
            dataset["states"],
            dataset["actions"],
            val_indices,
            FLAGS.batch_size,
            validation=True)
        """
        # start generator thread storing batches into a queue
        data_gen_queue, _stop = generator_queue(train_data_generator)

        is_chief = FLAGS.task_index == 0
        sv = tf.train.Supervisor(is_chief=is_chief,
                                 logdir=FLAGS.logdir,
                                 global_step=global_step,
                                 summary_op=None,
                                 saver=tf.train.Saver(max_to_keep=0),
                                 init_op=init_op)

        config = tf.ConfigProto(allow_soft_placement=True)

        print("Wait for session ...")
        """
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=is_chief,
                                               checkpoint_dir=FLAGS.logdir,
                                               config=config) as sess:
        """
        with sv.managed_session(server.target, config=config) as sess:
            print("Session initialized.")
            if is_chief:
                summary_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)

            while not sv.should_stop():
                # perform training cycles
                wait_time = 0.01  # in seconds
                epoch = 0
                step = 0
                reports = 0
                while epoch < FLAGS.epoch:
                    sample_seen = 0
                    while sample_seen < epoch_length:
                        generator_output = None
                        while not _stop.is_set():
                            if not data_gen_queue.empty():
                                generator_output = data_gen_queue.get()
                                break
                            else:
                                time.sleep(wait_time)

                        states, actions = generator_output
                        batch_size = len(states[0])
                        sample_seen += batch_size

                        _, loss, acc, summary, step = sess.run(
                                [train_op, loss_op, acc_op, summary_op, global_step],
                                feed_dict={
                                    states_placeholder: states,
                                    actions_placeholder: actions
                                })

                        if is_chief:
                            try:
                                if step >= FLAGS.checkpoint * (reports+1):
                                    print("Execute checkpoint at step {:d}.".format(step))
                                    reports += 1
                                    sv.saver.save(sess, sv.save_path, global_step=step)
                                    summary_writer.add_summary(summary, global_step=step)
                                    summary_writer.flush()
                            except:
                                err, msg, _ = sys.exc_info()
                                sys.stderr.write("{} {}\n".format(err, msg))
                                sys.stderr.write(traceback.format_exc())

                    epoch += 1

                _stop.set()
                break

            if is_chief:
                sv.request_stop()
            else:
                sv.stop()


def main(argv=None):
    from ast import literal_eval
    cluster_spec = literal_eval(open(FLAGS.cluster_spec).read())
    num_workers = len(cluster_spec['worker'])
    cluster = tf.train.ClusterSpec(cluster_spec)

    # start a server for a specific task
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement,
                            device_count={'GPU': 1})
    config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction

    server = tf.train.Server(cluster,
                             config=config,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == 'ps':
        server.join()
    elif FLAGS.job_name == 'worker':
        run_training(cluster, server, num_workers)


if __name__ == '__main__':
    tf.app.run(main=main)
