import numpy as np

cimport numpy as np

from operator import itemgetter

from bamboo.go.board cimport PURE_BOARD_SIZE, PASS
from bamboo.go.board cimport game_state_t, rollout_feature_t
from bamboo.go.board cimport is_legal
from bamboo.rollout.preprocess cimport update_planes, update_probs

import time


cdef class RolloutPolicyPlayer(object):

    def __cinit__(self, move_limit=500):
        self.move_limit = move_limit

    def __dealloc__(self):
        pass

    cdef int get_move(self, game_state_t *game):
        cdef rollout_feature_t *feature_t = &game.rollout_feature_planes[<int>game.current_color]

        # check move limit
        if self.move_limit is not None and game.moves > self.move_limit:
            return PASS

        s = time.time()
        # update feature planes
        update_planes(game)
        print('Feature Calculation Speed. {:.3f} us'.format((time.time()-s)*1000*1000))
        s = time.time()
        # generate and sort probs
        update_probs(game)
        print('Forward Speed. {:.3f} us'.format((time.time()-s)*1000*1000))
        probs = np.asarray(game.rollout_probs)
        pos = np.argmax(probs)

        if is_legal(game, pos, game.current_color):
            return pos
        else:
            # allow up to the 3rd candidate
            candidate = 1
            while candidate < 10:
                pos = np.argsort(probs)[::-1][candidate]
                if is_legal(game, pos, game.current_color):
                    return pos
                else:
                    print('Candidate number {:d} is not legal. {:d}'.format(candidate, pos))
                    candidate += 1

        # No 'sensible' moves available, so do pass move
        return PASS
