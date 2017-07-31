# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

import msgpackrpc

from libc.stdio cimport printf

from bamboo.go.board cimport FLIP_COLOR 
from bamboo.go.board cimport game_state_t
from bamboo.go.board cimport allocate_game, initialize_board, put_stone
from bamboo.mcts.tree_search cimport MCTS


cdef class MCTSServer(object):

    def __cinit__(self,
                  object policy,
                  double temperature=0.67,
                  int playout_limit=1000,
                  int n_threads=1):
        self.mcts = MCTS(policy, temperature, playout_limit, n_threads)

        self.game = allocate_game()
        initialize_board(self.game)

    def start_pondering(self):
        self.mcts.start_search_thread(self.game)

    def stop_pondering(self):
        self.mcts.stop_search_thread()

    def genmove(self, color):
        pos = self.mcts.genmove(self.game)

    def play(self, pos, color):
        cdef bint legal
        legal = put_stone(self.game, pos, color)
        if legal:
            self.game.current_color = FLIP_COLOR(self.game.current_color)
            return True
        else:
            return False
