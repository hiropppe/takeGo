#!/usr/bin/env python
from __future__ import division

import h5py as h5
import numpy as np
import sys
import traceback

from tqdm import tqdm


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def start_testing(args):
    dataset = h5.File(args.test_data)
    states = dataset['states']
    actions = dataset['actions']
    if args.policy == 'rollout':
        n_features = 6
    else:
        n_features = 9

    board_max = states.shape[-1]

    n_test = len(states)

    with h5.File(args.weights) as weights_h5:
        params = {'W': weights_h5['W'].value}

    print('Test size: {:d}'.format(n_test))

    test_indices = np.random.permutation(n_test)

    n_test_acc = 0
    for j in tqdm(test_indices):
        try:
            onehot_index_array = states[j]

            # compute logits
            logits = np.zeros(board_max, dtype=np.float64)
            for k in range(n_features):
                onehot_index_position = np.where(onehot_index_array[k] != -1)[0]
                for onehot_index in onehot_index_position:
                    logits[onehot_index] += params['W'][onehot_index_array[k, onehot_index]]

            # one-hot
            t = np.zeros(board_max, dtype=np.float64)
            t[actions[j]] = 1

            y = softmax(logits)

            n_test_acc += t[np.argmax(y)]
        except KeyboardInterrupt:
            return 0
        except:
            sys.stderr.write('Unexpected error at test index {:d}\n'.format(test_indices[j]))
            err, msg, _ = sys.exc_info()
            sys.stderr.write("{} {}\n".format(err, msg))
            sys.stderr.write(traceback.format_exc())

    print('Acc. {:.3f} ({:.0f}/{:.0f}) '.format(n_test_acc/n_test, n_test_acc, n_test))


def handle_arguments(cmd_line_args=None):
    import argparse
    parser = argparse.ArgumentParser(description='Perform supervised training on a rollout policy.')
    parser.add_argument("test_data", help="A .h5 file of test data")

    parser.add_argument("--policy", "-p", type=str, default='rollout', choices=['rollout', 'tree'],
                        help="Choice policy to generate feature (Default: rollout)")
    parser.add_argument("--weights", "-w", default=None,
                        help="Name of a .h5 weights file (in the output directory) to load to resume training")
    parser.add_argument("--verbose", "-v", default=False, action="store_true",
                        help="Turn on verbose mode")

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    start_testing(args)


if __name__ == '__main__':
    handle_arguments()
