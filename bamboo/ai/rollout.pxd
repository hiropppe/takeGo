from cpython.ref cimport PyObject

from libcpp.string cimport string

from bamboo.go.board cimport game_state_t
from bamboo.models.linear_softmax cimport LinearSoftmax
from bamboo.rollout.preprocess cimport RolloutFeature

cdef class RolloutPolicyPlayer:
    cdef:
        LinearSoftmax linear_softmax
        RolloutFeature rollout_feature
        int move_limit

    cdef int get_move(self, game_state_t *game)
