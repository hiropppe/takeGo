import h5py as h5
import numpy as np
import sys

from tqdm import tqdm

data_file = sys.argv[1]

lr = 0.001
iter_num = 1000
test_size = .2


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

    n_total = len(states)
    n_test = (int)(n_total*test_size)
    n_train = n_total - n_test

    print('Total size: {:d} Train size: {:d} Test size: {:d}'.format(n_total, n_train, n_test))

    train_acc_list, train_loss_list, test_acc_list = [], [], []
    for i in range(iter_num):
        n_train_acc, n_train_total_loss = 0, 0.
        n_test_acc = 0

        indices = np.random.permutation(n_total)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        print('Epoch {:d}'.format(i))

        # train
        for j in tqdm(range(n_train)):
            X = np.zeros((board_max, n_features))
            onehot_index_array = states['i' + str(train_indices[j])]
            if onehot_index_array.shape != (1,):
                for pos, plane in onehot_index_array:
                    X[pos, plane] = 1
            # one-hot
            t = np.zeros(board_max)
            t[actions['i' + str(train_indices[j])].value[0]] = 1

            y = softmax(np.dot(X, W))
            loss = cross_entropy_error(y, t)

            dy = y - t
            grad = np.dot(X.T, dy)

            W -= lr*grad

            n_train_acc += t[np.argmax(y)]
            n_train_total_loss += loss

            if (j+1) % 100 == 0:
                print('Acc. {:.3f} ({:.0f}/{:.0f}) Loss. {:.3f}'.format(n_train_acc*100/j+1,
                                                                        n_train_acc,
                                                                        j+1,
                                                                        n_train_total_loss/j+1))

        train_acc_list.append(n_train_acc*100/n_train)
        train_loss_list.append(n_train_total_loss/n_train)

        print('Acc. {:.3f} ({:.0f}/{:.0f}) '.format(train_acc_list[-1], n_train_acc, n_train) +
              'Loss. {:.3f} '.format(train_loss_list[-1]))

        # test
        for j in range(n_test):
            X = np.zeros((board_max, n_features))
            onehot_index_array = states['i' + str(test_indices[j])]
            if onehot_index_array.shape != (1,):
                for pos, plane in onehot_index_array:
                    X[pos, plane] = 1

            # one-hot
            t = np.zeros(board_max)
            t[actions['i' + str(test_indices[j])].value[0]] = 1

            y = softmax(np.dot(X, W))

            n_test_acc += t[np.argmax(y)]

        test_acc_list.append(n_test_acc*100/n_test)

        print('Acc. {:.3f} ({:.0f}/{:.0f}) '.format(train_acc_list[-1], n_train_acc, n_train) +
              'Loss. {:.3f} '.format(train_loss_list[-1]) +
              'Val Acc. {:.3f} ({:.0f}/{:.0f}) '.format(test_acc_list[-1], n_test_acc, n_test))


if __name__ == '__main__':
    run_training()
