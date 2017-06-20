import h5py as h5
import numpy as np
import sys

from scipy.sparse import csr_matrix
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
    n_features = dataset['features']['n'].value[0]

    # W = 0.01 * np.random.randn(n_feature)
    rgen = np.random.RandomState(1)
    W = rgen.normal(loc=0.0, scale=0.01, size=n_features)

    board_max = 361

    n_total = len(actions)
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
            name = 'i' + str(train_indices[j])

            csr_data = states['data'][name].value
            csr_indices = states['indices'][name].value
            csr_indptr = states['indptr'][name].value

            X = csr_matrix((csr_data, csr_indices, csr_indptr), shape=(n_features+1, board_max)).toarray()
            # truncate dummy constant plane 1
            X = X[:-1].T

            # one-hot
            t = np.zeros(board_max)
            t[actions[name].value[0]] = 1

            y = softmax(np.dot(X, W))
            loss = cross_entropy_error(y, t)

            dy = y - t
            grad = np.dot(X.T, dy)

            W -= lr*grad

            n_train_acc += t[np.argmax(y)]
            n_train_total_loss += loss
            n_report_acc += t[np.argmax(y)]
            n_report_total_loss += loss

            if (j+1) % report_size == 0:
                print('\nAcc. {:.3f} Loss. {:.3f}'.format(n_report_acc*100/report_size,
                                                          n_report_total_loss/report_size))
                n_report_acc = 0
                n_report_total_loss = 0.

        train_acc_list.append(n_train_acc*100/n_train)
        train_loss_list.append(n_train_total_loss/n_train)

        # test
        for j in range(n_test):
            name = 'i' + str(test_indices[j])

            csr_data = states['data'][name].value
            csr_indices = states['indices'][name].value
            csr_indptr = states['indptr'][name].value

            X = csr_matrix((csr_data, csr_indices, csr_indptr), shape=(n_features+1, board_max)).toarray()
            # truncate dummy constant plane 1
            X = X[:-1].T

            # one-hot
            t = np.zeros(board_max)
            t[actions[name].value[0]] = 1

            y = softmax(np.dot(X, W))

            n_test_acc += t[np.argmax(y)]

        test_acc_list.append(n_test_acc*100/n_test)

        print('Acc. {:.3f} ({:.0f}/{:.0f}) '.format(train_acc_list[-1], n_train_acc, n_train) +
              'Loss. {:.3f} '.format(train_loss_list[-1]) +
              'Val Acc. {:.3f} ({:.0f}/{:.0f}) '.format(test_acc_list[-1], n_test_acc, n_test))


if __name__ == '__main__':
    run_training()
