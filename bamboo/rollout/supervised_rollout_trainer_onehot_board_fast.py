import h5py as h5
import numpy as np
import sys
import traceback

from tqdm import tqdm

data_file = sys.argv[1]

lr = 0.005
iter_num = 1000
test_size = .2
report_size = 10000


def softmax(x):
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


def run_training():
    dataset = h5.File(data_file)

    states = dataset['states']
    actions = dataset['actions']
    n_features = dataset['n_features'].value

    # W = 0.01 * np.random.randn(n_feature)
    rgen = np.random.RandomState(1)
    W = rgen.normal(loc=0.0, scale=0.01, size=n_features)

    board_max = states.shape[-1]

    n_total = len(states)
    n_test = (int)(n_total*test_size)
    n_train = n_total - n_test

    print('Total size: {:d} Train size: {:d} Test size: {:d}'.format(n_total, n_train, n_test))

    train_acc_list, train_loss_list, test_acc_list = [], [], []
    for i in range(iter_num):
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
                for k in range(6):
                    onehot_index_position = np.where(onehot_index_array[k] != -1)[0]
                    # memorize onehot positions by feature
                    onehot_index_position_by_feature[k] = onehot_index_position
                    for onehot_index in onehot_index_position:
                        logits[onehot_index] += W[onehot_index_array[k, onehot_index]]

                # one-hot
                t = np.zeros(board_max, dtype=np.float64)
                t[actions[train_indices[j]]] = 1

                y = softmax(logits)
                loss = cross_entropy_error(y, t)

                dy = y - t

                # compute grad
                grad = np.zeros(n_features, dtype=np.float64)
                for k in range(6):
                    for onehot_index in onehot_index_position_by_feature[k]:
                        grad[onehot_index_array[k, onehot_index]] += dy[onehot_index]

                W -= lr*grad

                n_train_acc += t[np.argmax(y)]
                n_train_total_loss += loss
                n_report_acc += t[np.argmax(y)]
                n_report_total_loss += loss
            except:
                sys.stderr.write('Unexpected error at train index {:d}\n'.format(train_indices[j]))
                err, msg, _ = sys.exc_info()
                sys.stderr.write("{} {}\n".format(err, msg))
                sys.stderr.write(traceback.format_exc())

            if (j+1) % report_size == 0:
                print('\nAcc. {:.3f} Loss. {:.3f}'.format(n_report_acc*100/report_size,
                                                          n_report_total_loss/report_size))
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
                for k in range(6):
                    onehot_index_position = np.where(onehot_index_array[k] != -1)[0]
                    for onehot_index in onehot_index_position:
                        logits[onehot_index] += W[onehot_index_array[k, onehot_index]]

                # one-hot
                t = np.zeros(board_max, dtype=np.float64)
                t[actions[test_indices[j]]] = 1

                y = softmax(logits)

                n_test_acc += t[np.argmax(y)]
            except:
                sys.stderr.write('Unexpected error at test index {:d}\n'.format(test_indices[j]))
                err, msg, _ = sys.exc_info()
                sys.stderr.write("{} {}\n".format(err, msg))
                sys.stderr.write(traceback.format_exc())

        test_acc_list.append(n_test_acc*100/n_test)

        print('Acc. {:.3f} ({:.0f}/{:.0f}) '.format(train_acc_list[-1], n_train_acc, n_train) +
              'Loss. {:.3f} '.format(train_loss_list[-1]) +
              'Val Acc. {:.3f} ({:.0f}/{:.0f}) '.format(test_acc_list[-1], n_test_acc, n_test))


if __name__ == '__main__':
    run_training()
