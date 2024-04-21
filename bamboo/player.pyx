import numpy as np
import sys

cimport numpy as np

from libc.stdio cimport printf
from libc.stdlib cimport abort, malloc, free, rand

from . cimport policy_feature as pf

from .board cimport game_state_t, is_legal_not_eye, onboard_pos, PURE_BOARD_SIZE, PURE_BOARD_MAX, BOARD_MAX, MAX_RECORDS, MAX_MOVES, S_BLACK, S_WHITE, PASS, RESIGN
from .policy_feature cimport MAX_POLICY_PLANES, MAX_VALUE_PLANES
from .printer cimport print_board, print_PN
from .tree_search cimport tree_node_t
from .zobrist_hash cimport uct_hash_size


cdef class PolicyPlayer(object):

    def __cinit__(self, model, double temperature=0.1):
        self.model = model
        self.temperature = temperature

    cdef int[:] genmove(self, np.ndarray[INT_t, ndim=4] tensor):    
        cdef int[:] pos

        probs = self.model.eval_state(tensor)
        if any(np.abs(probs.sum(axis=1) - 1.0) > 0.01):
            print('>> Warnings. Sum of PN evaluation values {:.3f} != 1.0', probs.sum(), file=sys.stderr)
        
        ## argmax
        pos = np.argmax(probs, axis=1).astype(np.int32)
        
        return pos
        
    cdef int[:] gen_masked_move(self, np.ndarray[INT_t, ndim=4] tensor, np.ndarray[np.npy_bool, ndim=2] mask):    
        cdef int[:] pos

        probs = self.model.eval_state(tensor)
        if any(np.abs(probs.sum(axis=1) - 1.0) > 0.01):
            print('>> Warnings. Sum of PN evaluation values {:.3f} != 1.0', probs.sum(), file=sys.stderr)
        
        assert len(probs) == len(mask)
        assert len(probs[0]) == len(mask[0])

        #print(probs[0])
        #print(mask)
        probs = probs * mask
        #print(probs[0])
        probs = apply_temperature(probs, mask, temperature=self.temperature)
        # wa. nan prob
        probs = np.nan_to_num(probs)
        pos = np.array([np.random.choice(len(prob), p=prob) for prob in probs], dtype=np.int32)

        return pos

    def __dealloc__(self):
        pass
        #pf.free_feature(self.policy_feature)


def apply_temperature(x, mask, temperature):
    beta = 1.0/temperature
    if x.ndim == 2:
        x = x.T
        x = x * beta
        ex = np.exp(x - np.max(x, axis=0)) * mask.T
        y = ex / np.sum(ex, axis=0)
        return y.T
    
    x = x * beta
    ex = np.exp(x - np.max(x)) * mask
    return ex / np.sum(ex)
