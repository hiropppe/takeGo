from bamboo.go.board cimport game_state_t
from bamboo.mcts.tree_search cimport MCTS


cdef class MCTSServer:
    cdef:
        MCTS mcts
        game_state_t *game
