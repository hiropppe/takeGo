import numpy as np

cimport numpy as np

from .board cimport game_state_t
from .tree_search cimport tree_node_t
from .policy_feature cimport PolicyFeature

ctypedef np.int32_t INT_t

cdef class PolicyPlayer:
    cdef object model
    cdef double temperature
    cdef int[:] genmove(self, np.ndarray[INT_t, ndim=4] tensors)
    cdef int[:] gen_masked_move(self, np.ndarray[INT_t, ndim=4] tensors, np.ndarray[np.npy_bool, ndim=2] mask)