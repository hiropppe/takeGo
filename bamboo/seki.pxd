from bamboo.board cimport game_state_t

cdef void check_seki(game_state_t *game, bint *seki)
cdef bint is_self_atari(game_state_t *game, int pos, int color)
