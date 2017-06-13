from bamboo.go.board cimport game_state_t


cpdef convert_to_simple_sgf(sgf_string)


cdef class SGFMoveIterator:
    cdef int bsize
    cdef game_state_t *game
    cdef int i 
    cdef list moves
    cdef tuple next_move
    cdef bint ignore_not_legal
    cdef bint verbose

    cdef int sgf_init_game(self, object sgf_root) except? -1
