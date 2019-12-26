#!/usr/bin/env python

import h5py as h5
import numpy as np
import os
import sys
import traceback
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from tqdm import tqdm
from bamboo.train.rollout.optimizer import SGD, Momentum, AdaGrad, Adam, Nesterov, RMSprop

# default settings
DEFAULT_EPOCH = 10
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = .01
DEFAULT_DECAY = .9
DEFAULT_DECAY_EVERY = 10000000
DEFAULT_MOMENTUM = .9
DEFAULT_BETA1 = .9
DEFAULT_BETA2 = .999
DEFAULT_TRAIN_VAL_TEST = [.9, .0, .1]
REPORT_SIZE = 50000


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


def start_training(args):
    dataset = h5.File(args.train_data)

    states = dataset['states']
    actions = dataset['actions']
    feature_size = dataset['n_features'].value
    if args.policy == 'rollout':
        n_features = 6
    else:
        n_features = 9

    board_max = states.shape[-1]

    n_total = len(states)
    n_test = (int)(n_total*args.train_val_test[-1])
    n_train = n_total - n_test

    if args.weights is None:
        rgen = np.random.RandomState(1)
        params = {'W': rgen.normal(loc=0.0, scale=0.01, size=feature_size)}
    else:
        with h5.File(args.weights) as weights_h5:
            params = {'W': weights_h5['W'].value}

    if args.optimizer == 'sgd':
        optimizer = SGD(lr=args.learning_rate, decay_every=args.decay_every, decay_rate=args.decay)
        if not args.decay_every:
            print('Using SGD:' +
                  ' lr={:s}'.format(str(args.learning_rate)))
        else:
            print('Using SGD with step decay:' +
                  ' lr={:s}'.format(str(args.learning_rate)) +
                  ' decay_every={:s}'.format(str(args.decay_every)) +
                  ' decay={:s}'.format(str(args.decay)))
    elif args.optimizer == 'momentum':
        optimizer = Momentum(lr=args.learning_rate, momentum=args.momentum)
        print('Using Momentum SGD:' +
              ' lr={:s}'.format(str(args.learning_rate)) +
              ' momentum={:s}'.format(str(args.momentum)))
    elif args.optimizer == 'nesterov':
        optimizer = Nesterov(lr=args.learning_rate, momentum=args.momentum)
        print('Using Nesterov:' +
              ' lr={:s}'.format(str(args.learning_rate)) +
              ' momentum={:s}'.format(str(args.momentum)))
    elif args.optimizer == 'adagrad':
        optimizer = AdaGrad(lr=args.learning_rate)
        print('Using AdaGrad:' +
              ' lr={:s}'.format(str(args.learning_rate)))
    elif args.optimizer == 'rmsprop':
        optimizer = RMSprop(lr=args.learning_rate, decay_rate=args.decay)
        print('Using RMSprop:' +
              ' lr={:s}'.format(str(args.learning_rate)) +
              ' decay={:s}'.format(str(args.decay)))
    else:
        optimizer = Adam(lr=args.learning_rate, beta1=args.beta1, beta2=args.beta2)
        print('Using Adam:' +
              ' lr={:s}'.format(str(args.learning_rate)) +
              ' beta1={:s}'.format(str(args.beta1)) +
              ' beta2={:s}'.format(str(args.beta2)))

    print('Total size: {:d} Train size: {:d} Test size: {:d}'.format(n_total, n_train, n_test))

    train_acc_list, train_loss_list, test_acc_list = [], [], []
    for i in range(args.epochs):
        n_train_acc, n_train_total_loss = 0, 0.
        n_report_acc, n_report_total_loss = 0, 0.
        n_test_acc = 0

        indices = np.random.permutation(n_total)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        print('Epoch {:d}'.format(i))

        # train
        for j in tqdm(range(n_train)):
            try:
                onehot_index_array = states[train_indices[j]]
                onehot_index_position_by_feature = {}

                # compute logits
                logits = np.zeros(board_max, dtype=np.float64)
                for k in range(n_features):
                    onehot_index_position = np.where(onehot_index_array[k] != -1)[0]
                    # memorize onehot positions by feature
                    onehot_index_position_by_feature[k] = onehot_index_position
                    for onehot_index in onehot_index_position:
                        logits[onehot_index] += params['W'][onehot_index_array[k, onehot_index]]

                # one-hot
                t = np.zeros(board_max, dtype=np.float64)
                t[actions[train_indices[j]]] = 1

                y = softmax(logits)
                loss = cross_entropy_error(y, t)

                dy = y - t

                # compute grad
                grads = {'W': np.zeros(feature_size, dtype=np.float64)}
                for k in range(n_features):
                    for onehot_index in onehot_index_position_by_feature[k]:
                        grads['W'][onehot_index_array[k, onehot_index]] += dy[onehot_index]

                optimizer.update(params, grads)

                n_train_acc += t[np.argmax(y)]
                n_train_total_loss += loss
                n_report_acc += t[np.argmax(y)]
                n_report_total_loss += loss
            except KeyboardInterrupt:
                return 0
            except:
                sys.stderr.write('Unexpected error at train index {:d}\n'.format(train_indices[j]))
                err, msg, _ = sys.exc_info()
                sys.stderr.write("{} {}\n".format(err, msg))
                sys.stderr.write(traceback.format_exc())

            if (j+1) % REPORT_SIZE == 0:
                print('\nAcc. {:.3f} Loss. {:.3f}'.format(n_report_acc*100/REPORT_SIZE,
                                                          n_report_total_loss/REPORT_SIZE))
                n_report_acc = 0
                n_report_total_loss = 0.

        train_acc_list.append(n_train_acc*100/n_train)
        train_loss_list.append(n_train_total_loss/n_train)

        # test
        for j in range(n_test):
            try:
                onehot_index_array = states[test_indices[j]]

                # compute logits
                logits = np.zeros(board_max, dtype=np.float64)
                for k in range(n_features):
                    onehot_index_position = np.where(onehot_index_array[k] != -1)[0]
                    for onehot_index in onehot_index_position:
                        logits[onehot_index] += params['W'][onehot_index_array[k, onehot_index]]

                # one-hot
                t = np.zeros(board_max, dtype=np.float64)
                t[actions[test_indices[j]]] = 1

                y = softmax(logits)

                n_test_acc += t[np.argmax(y)]
            except KeyboardInterrupt:
                return 0
            except:
                sys.stderr.write('Unexpected error at test index {:d}\n'.format(test_indices[j]))
                err, msg, _ = sys.exc_info()
                sys.stderr.write("{} {}\n".format(err, msg))
                sys.stderr.write(traceback.format_exc())

        test_acc_list.append(n_test_acc*100/n_test)

        print('Acc. {:.3f} ({:.0f}/{:.0f}) '.format(train_acc_list[-1], n_train_acc, n_train) +
              'Loss. {:.3f} '.format(train_loss_list[-1]) +
              'Val Acc. {:.3f} ({:.0f}/{:.0f}) '.format(test_acc_list[-1], n_test_acc, n_test))

        if not os.path.exists(args.out_directory):
            os.mkdir(args.out_directory)

        params_filename = os.path.join(args.out_directory,
                                       'weights.{:s}.hdf5'.format(str(i).rjust(5, '0')))
        params_file = h5.File(params_filename, 'w')
        for key in params.keys():
            params_file.create_dataset(key, data=params[key])
        params_file.close()


def handle_arguments(cmd_line_args=None):
    import argparse
    parser = argparse.ArgumentParser(description='Perform supervised training on a rollout policy.')

    parser.add_argument("train_data", help="A .h5 file of training data")
    parser.add_argument("out_directory", help="Directory where metadata and weights will be saved")
    parser.add_argument("--policy", "-p", type=str, default='rollout', choices=['rollout', 'tree'],
                        help="Choice policy to generate feature (Default: rollout)")
    parser.add_argument("--minibatch", "-B", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Minibatch size. Default: " + str(DEFAULT_BATCH_SIZE))
    parser.add_argument("--epochs", "-E", type=int, default=DEFAULT_EPOCH,
                        help="Total number of iterations on the data. Default: " + str(DEFAULT_EPOCH))
    parser.add_argument("--optimizer", "-op", type=str, default='sgd',
                        choices=['sgd', 'momentum', 'nesterov', 'adagrad', 'adam', 'rmsprop'],
                        help="Choice optimizer type (Default: sgd)")
    parser.add_argument("--learning-rate", "-lr", type=float, default=DEFAULT_LEARNING_RATE,
                        help="Learning rate - how quickly the model learns at first. Default: " + str(DEFAULT_LEARNING_RATE))
    parser.add_argument("--momentum", "-m", type=float, default=DEFAULT_MOMENTUM,
                        help=("Hyper parameter of Momentum SGD and Nesterov. Default: " + str(DEFAULT_MOMENTUM)))
    parser.add_argument("--decay", "-d", type=float, default=DEFAULT_DECAY,
                        help=("The rate at which learning decreases. SGD and RMSprop Default: " + str(DEFAULT_DECAY)))
    parser.add_argument("--beta1", "-b1", type=float, default=DEFAULT_BETA1,
                        help="Hyper parameter of Adam. Default: " + str(DEFAULT_BETA1))
    parser.add_argument("--beta2", "-b2", type=float, default=DEFAULT_BETA2,
                        help="Hyper parameter of Adam. Default: " + str(DEFAULT_BETA2))
    parser.add_argument("--decay-every", "-de", type=int, default=None,
                        help="Batch size of decay learning rate. Default: None")
    parser.add_argument("--weights", "-w", default=None,
                        help="Name of a .h5 weights file (in the output directory) to load to resume training")
    parser.add_argument("--train-val-test", nargs=3, type=float, default=DEFAULT_TRAIN_VAL_TEST,
                        help="Fraction of data to use for training/val/test. Must sum to 1. Default: " + str(DEFAULT_TRAIN_VAL_TEST))
    parser.add_argument("--verbose", "-v", default=False, action="store_true",
                        help="Turn on verbose mode")

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    start_training(args)


if __name__ == '__main__':
    handle_arguments()
