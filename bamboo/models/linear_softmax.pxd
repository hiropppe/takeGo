cdef class LinearSoftmax:

    cdef:
        double weights[100000]
        double temperature
        double probs[361]
        double logits[361]
        double logits_sum

    cdef void softmax(self, int onehot_ix[6][361]) nogil
    cdef void update_softmax(self, int positions[529], int onehot_ix[6][361]) nogil
