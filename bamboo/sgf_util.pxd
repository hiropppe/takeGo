from bamboo.board cimport game_state_t


cpdef min_sgf_extract(sgf_string)


cdef class SGFMoveIterator:
    cdef int bsize
    cdef game_state_t *game
    cdef int i 
    cdef list moves
    cdef tuple next_move
    cdef float komi
    cdef int winner
    cdef int too_few_moves_threshold
    cdef int too_many_moves_threshold
    cdef bint ignore_not_legal
    cdef bint ignore_no_result
    cdef bint verbose

    cdef int sgf_init_game(self, object sgf_root) except? -1


cdef void save_gamestate_to_sgf(game_state_t *game,
                                path,
                                filename,
                                black_player_name,
                                white_player_name)
