from cpython.ref cimport PyObject

from libcpp.string cimport string

from bamboo.go.board cimport game_state_t
from bamboo.go.policy_feature cimport policy_feature_t


cdef class GreedyPolicyPlayer:
    #cdef PyObject *policy_net
    cdef object sl_policy
    cdef policy_feature_t *feature
    cdef bint pass_when_offered
    cdef int move_limit

    cdef int get_move(self, game_state_t *game)
