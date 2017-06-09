from bamboo.go.board cimport game_state_t

cdef class SGFIterator:
    cdef int bsize
    cdef game_state_t *game
    cdef int current_moves
    cdef list moves
    cdef int next_move

    cdef int read(self, object sgf_string) except? -1
    cdef bint has_next(self)
    cdef game_state_t *move_next(self)
    cdef int sgf_init_gamestate(self, object sgf_root) except? -1
