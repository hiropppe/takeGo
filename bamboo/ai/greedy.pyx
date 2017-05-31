import numpy as np

cimport numpy as np

from cpython.ref cimport PyObject

from operator import itemgetter

from bamboo.go.board cimport PURE_BOARD_SIZE, PASS
from bamboo.go.board cimport game_state_t
from bamboo.go.board cimport is_legal 

from bamboo.go.policy_feature cimport allocate_feature, initialize_feature, free_feature, update

import time


cdef class GreedyPolicyPlayer(object):
    """A player that uses a greedy policy (i.e. chooses the highest probability
       move each turn)
    """
    def __cinit__(self, sl_policy, pass_when_offered=False, move_limit=500):
        self.sl_policy = sl_policy
        self.feature = allocate_feature()
        self.pass_when_offered = pass_when_offered
        self.move_limit = move_limit
        initialize_feature(self.feature)

    def __dealloc__(self):
        free_feature(self.feature)

    cdef int get_move(self, game_state_t *game):
        # check move limit
        if self.move_limit is not None and game.moves > self.move_limit:
            return PASS

        # check if pass was offered and we want to pass
        if self.pass_when_offered:
            if game.moves > 100 and game.record[game.moves-1].pos == PASS:
                return PASS
        s = time.time()
        # update feature planes
        update(self.feature, game)
        feature_tensor = np.asarray(self.feature.planes)
        feature_tensor = feature_tensor.reshape((1, 48, PURE_BOARD_SIZE, PURE_BOARD_SIZE))
        print('Feature Calculation Speed. {:.3f} us'.format((time.time()-s)*1000*1000))
        s = time.time()
        # generate and sort probs
        probs = self.sl_policy.eval_state(feature_tensor)
        print('Forward Speed. {:.3f} ms'.format((time.time()-s)*1000))
        pos = np.argmax(probs)

        if is_legal(game, pos, game.current_color):
            return pos
        else:
            # allow up to the 3rd candidate
            candidate = 1
            while candidate < 3:
                pos = np.argsort(probs)[::-1][candidate]
                if is_legal(game, pos, game.current_color):
                    return pos
                else:
                    candidate += 1

        # No 'sensible' moves available, so do pass move
        return PASS
