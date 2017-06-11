from bamboo.go.board cimport game_state_t


cdef class SGFMoveIterator:
    cdef game_state_t *game
    cdef int i 
    cdef list moves
    cdef tuple next_move

    cdef int sgf_init_game(self, object sgf_root) except? -1
