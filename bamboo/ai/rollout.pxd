from bamboo.go.board cimport game_state_t

cdef class RolloutPolicyPlayer:
    cdef:
        int move_limit

    cdef int get_move(self, game_state_t *game)
