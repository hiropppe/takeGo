import numpy as np

cimport numpy as np

from operator import itemgetter

from bamboo.go.board cimport PURE_BOARD_SIZE, PASS
from bamboo.go.board cimport game_state_t
from bamboo.go.board cimport is_legal 
from bamboo.models.linear_softmax cimport LinearSoftmax
from bamboo.rollout.preprocess cimport RolloutFeature
from bamboo.rollout.preprocess cimport rollout_feature_t

import time


cdef class RolloutPolicyPlayer(object):

    def __cinit__(self,
                  linear_softmax,
                  rollout_feature,
                  move_limit=500):
        self.linear_softmax = linear_softmax
        self.rollout_feature = rollout_feature
        self.move_limit = move_limit

    def __dealloc__(self):
        pass

    cdef int get_move(self, game_state_t *game):
        cdef rollout_feature_t *feature_t = &self.rollout_feature.feature_planes[<int>game.current_color]

        # check move limit
        if self.move_limit is not None and game.moves > self.move_limit:
            return PASS

        s = time.time()
        # update feature planes
        self.rollout_feature.update(game)
        print('Feature Calculation Speed. {:.3f} us'.format((time.time()-s)*1000*1000))
        s = time.time()
        # generate and sort probs
        self.linear_softmax.softmax(feature_t.tensor)
        print('Forward Speed. {:.3f} ms'.format((time.time()-s)*1000))
        probs = np.asarray(self.linear_softmax.probs)
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
