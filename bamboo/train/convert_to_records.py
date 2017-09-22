from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import dill
import h5py as h5
import numpy as np
import os
import sys

import tensorflow as tf

from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor

FLAGS = None


BOARD_TRANSFORMATIONS = {
    'noop': lambda feature: feature,
    'rot90': lambda feature: np.rot90(feature, 1),
    'rot180': lambda feature: np.rot90(feature, 2),
    'rot270': lambda feature: np.rot90(feature, 3),
    'fliplr': lambda feature: np.fliplr(feature),
    'flipud': lambda feature: np.flipud(feature),
    'diag1': lambda feature: np.transpose(feature),
    'diag2': lambda feature: np.fliplr(np.rot90(feature, 1))
}


def convert_to(data, output_dir, output_name):
    """Converts a dataset to tfrecords."""
    # Check num_examples
    data_set = h5.File(data)
    states = data_set['states']
    actions = data_set['actions']
    num_examples = states.shape[0]

    if num_examples != actions.shape[0]:
        raise ValueError('States size {:d} does not match actions size {:s}.'
                         .format(num_examples, actions.shape[0]))

    data_set.close()

    if FLAGS.symmetries:
        executor = ProcessPoolExecutor(max_workers=FLAGS.workers)
        for transform_name in BOARD_TRANSFORMATIONS.keys():
            executor.submit(write_tfrecords,
                            data,
                            output_dir,
                            output_name,
                            transform_name)
        try:
            executor.shutdown()
        except:
            executor.shutdown(wait=False)
    else:
        write_tfrecords(data, output_dir, output_name, 'noop')


def write_tfrecords(data, output_dir, output_name, transform_name):
    """Converts a dataset to tfrecords."""
    data_set = h5.File(data)
    states = data_set['states']
    actions = data_set['actions']
    bsize = states.shape[2]
    num_examples = states.shape[0]
    transform = BOARD_TRANSFORMATIONS[transform_name]

    if transform_name == 'noop':
        filename = os.path.join(output_dir, output_name + '.tfrecords')
    else:
        filename = os.path.join(output_dir, output_name + '_' + transform_name + '.tfrecords')
    print('Writing', filename)
    opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    with tf.python_io.TFRecordWriter(filename, options=opt) as writer:
        pbar = tqdm(total=num_examples)
        for index in xrange(num_examples):
            state = states[index]
            # transpose to TF input shape (samples, rows, cols, input_depth)
            state = state.transpose(1, 2, 0)
            state = state.astype(np.float32)

            action = np.zeros((bsize, bsize), dtype=np.float32)
            action_index = actions[index]
            action[action_index[0], action_index[1]] = 1

            if not transform_name == 'noop':
                state_transform = transform(state)
                if np.all(state_transform == state):
                    continue
                state = state_transform
                action = transform(action)

            d_feature = {}
            d_feature['state'] = tf.train.Feature(float_list=tf.train.FloatList(value=state.flatten()))
            d_feature['action'] = tf.train.Feature(float_list=tf.train.FloatList(value=action.flatten()))

            features = tf.train.Features(feature=d_feature)
            example = tf.train.Example(features=features)
            serialized = example.SerializeToString()
            writer.write(serialized)

            pbar.update(1)

    print("Writing {:s} done!".format(filename))


def read_test(file_path):
    opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    for serialized_example in tf.python_io.tf_record_iterator(file_path, options=opt):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        print(np.array(example.features.feature['state'].float_list.value))
        print(np.array(example.features.feature['action'].float_list.value))
        break


def apply_packed_function((dumped_function, item, args, kwargs),):
    target_function = dill.loads(dumped_function)
    res = target_function(item, *args, **kwargs)
    return res


def pack_function(target_function, item, *args, **kwargs):
    dumped_function = dill.dumps(target_function)
    return apply_packed_function, (dumped_function, item, args, kwargs)


def main(unused_argv):
    # Convert to Examples and write the result to TFRecords.
    output_dirname = os.path.dirname(FLAGS.data)
    output_name = os.path.basename(FLAGS.data)
    output_name = output_name[:output_name.find('.')]
    convert_to(FLAGS.data, output_dirname, output_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='State action pair extracted from SGF (generated by sgf2hdf5).'
    )
    parser.add_argument(
        '--symmetries',
        action='store_true',
        default=False,
        help='Add symmetric board state or not.'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help=''
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
