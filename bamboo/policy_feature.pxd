# -*- coding: utf-8 -*-

from . cimport board

from .tree_search cimport tree_node_t

cdef enum:
    MAX_POLICY_PLANES = 48
    MAX_VALUE_PLANES = 49
    MAX_LADDER_DEPTH = 80
    MAX_LADDER_MOVES = 1000


cdef class PolicyFeature:
    cdef:
        int n_planes
        int[:, ::1] planes
        board.game_state_t *search_games


cdef PolicyFeature allocate_feature(int n_planes)
cdef void initialize_feature(PolicyFeature feature)
cdef void free_feature(PolicyFeature feature)

cdef void update(PolicyFeature feature, tree_node_t *node)

cdef int is_ladder_capture(board.game_state_t *game, int string_id, int pos, int othre_pos,
                           board.game_state_t *search_games, int depth, int *ladder_moves)

cdef int is_ladder_escape(board.game_state_t *game, int string_id, int pos, bint is_atari_pos,
                          board.game_state_t *search_games, int depth, int *ladder_moves)
