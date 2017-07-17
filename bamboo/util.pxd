from bamboo.go.board cimport game_state_t


cpdef min_sgf_extract(sgf_string)


cdef class SGFMoveIterator:
    cdef int bsize
    cdef game_state_t *game
    cdef int i 
    cdef list moves
    cdef tuple next_move
    cdef int too_few_moves_threshold
    cdef int too_many_moves_threshold
    cdef bint ignore_not_legal
    cdef bint verbose

    cdef int sgf_init_game(self, object sgf_root) except? -1
