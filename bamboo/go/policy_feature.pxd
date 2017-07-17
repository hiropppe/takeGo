# -*- coding: utf-8 -*-

cimport board

cdef enum:
    MAX_POLICY_PLANES = 48
    MAX_LADDER_DEPTH = 80
    MAX_LADDER_MOVES = 1000

ctypedef struct policy_feature_t:
    int n_planes
    int[:, ::1] planes
    board.game_state_t search_games[80]


cdef policy_feature_t *allocate_feature()
cdef void initialize_feature(policy_feature_t *feature)
cdef void free_feature(policy_feature_t *feature)

cdef void update(policy_feature_t *feature, board.game_state_t *game)

cdef int is_ladder_capture(board.game_state_t *game, int string_id, int pos, int othre_pos,
                           board.game_state_t search_games[80], int depth, int *ladder_moves)

cdef int is_ladder_escape(board.game_state_t *game, int string_id, int pos, bint is_atari_pos,
                          board.game_state_t search_games[80], int depth, int *ladder_moves)
