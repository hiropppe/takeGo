# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

import h5py as h5

from libc.math cimport exp
from libc.stdio cimport printf

from bamboo.go.board cimport PURE_BOARD_MAX


cdef class LinearSoftmax:

    def __cinit__(self, weights_hdf5=None, temperature=0.67):
        cdef int i
        for i in range(PURE_BOARD_MAX):
            self.probs[i] = .0
            self.logits[i] = .0
        self.logits_sum = .0

        if weights_hdf5:
            weights_data = h5.File(weights_hdf5, 'r')
            W = weights_data['W']
            for i in range(W.shape[0]):
                self.weights[i] = W[i]

        self.temperature = temperature

    def __dealloc__(self):
        pass

    cdef void softmax(self, int onehot_ix[6][361]) nogil:
        cdef int i, j
        for i in range(PURE_BOARD_MAX):
            for j in range(6):
                if onehot_ix[j][i] != -1:
                    self.logits[i] += self.weights[onehot_ix[j][i]]
            self.logits[i] = exp(self.logits[i]/self.temperature)
            self.logits_sum += self.logits[i]

        if self.logits_sum > .0:
            for i in range(PURE_BOARD_MAX):
                self.probs[i] = self.logits[i]/self.logits_sum

    cdef void update_softmax(self, int positions[361], int n, int onehot_ix[6][361]) nogil:
        cdef int pos
        cdef double updated_sum = .0
        cdef double updated_old_sum = .0
        cdef int i, j

        for i in range(n):
            pos = positions[i]
            updated_old_sum += self.logits[pos]
            self.logits[pos] = .0
            for j in range(6):
                if onehot_ix[j][pos] != -1:
                    self.logits[pos] += self.weights[onehot_ix[j][pos]]
            self.logits[pos] = exp(self.logits[pos]/self.temperature)
            updated_sum += self.logits[pos]

        self.logits_sum = self.logits_sum - updated_old_sum + updated_sum

        if self.logits_sum > .0:
            for i in range(PURE_BOARD_MAX):
                self.probs[i] = self.logits[i]/self.logits_sum


def test_update_speed():
    cdef LinearSoftmax ls = LinearSoftmax()
    cdef double weights[50000]
    cdef int onehot_ix[6][361]
    cdef int positions[361]
    cdef int n, n_max = 30
    cdef int i, j, k, l, p

    ls.temperature = 0.67

    import numpy as np
    import h5py as h5
    import time

    rgen = np.random.RandomState(1)
    W = rgen.normal(loc=0.0, scale=0.01, size=50000)
    for i in range(50000):
        ls.weights[i] = W[i]

    data = h5.File('kihuu_planes.h5')
    for j in range(6):
        for k in range(361):
            onehot_ix[j][k] = data['states'][0, j, k]
    ls.softmax(onehot_ix)

    speeds = []
    for i in range(1, 200):
        for j in range(6):
            for k in range(361):
                onehot_ix[j][k] = data['states'][i, j, k]
        for n, p in enumerate(np.where(data['states'][i, 5] != -1)[0]):
            positions[n] = p
            if n == n_max - 1:
                break
        s = time.time()
        #ls.softmax(onehot_ix)
        ls.update_softmax(positions, n+1, onehot_ix)
        speeds.append(time.time()-s)

    print('{:.3f} us'.format(np.mean(speeds)*1000*1000))
