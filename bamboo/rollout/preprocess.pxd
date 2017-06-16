from bamboo.go.board cimport game_state_t, string_t

cdef enum:
    FEATURE_MAX = 109747

cdef enum:
    RESPONSE
    SAVE_ATARI
    NEIGHBOR
    NAKADE
    RESPONSE_PAT
    NON_RESPONSE_PAT

cdef struct rollout_feature_t:
    int color
    float tensor[361][6] # hold one-hot index for each feature
    bint is_neighbor8_set
    int prev_neighbor8[8]


cdef class RolloutFeature:
    cdef rollout_feature_t feature_planes[3]
    cdef int feature_size
    cdef int response_size, save_atari_size, neighbor_size, nakade_size, x33_size, d12_size
    cdef int response_start, save_atari_start, neighbor_start, nakade_start, x33_start, d12_start

    """ function to generate carray feature for mcts
    """
    cdef void rebase(self, game_state_t *game) nogil

    cdef void update(self, game_state_t *game) nogil

    cdef void update_save_atari(self, game_state_t *game, string_t *string) nogil

    cdef void update_neighbor(self, game_state_t *game, int pos) nogil
    
    cdef void update_3x3(self, game_state_t *game, int pos) nogil

    cdef void update_d12(self, game_state_t *game, int pos) nogil

    cdef void clear_onehot_index(self, game_state_t *game, int pos) nogil

    """ function to generate sparse feature for training
    """
    cdef void update_lil(self, game_state_t *game, object lil_matrix)

    cdef void update_save_atari_lil(self, game_state_t *game, string_t *string, object lil_matrix)

    cdef void update_neighbor_lil(self, game_state_t *game, int pos, object lil_matrix)
    
    cdef void update_3x3_lil(self, game_state_t *game, int pos, object lil_matrix)

    cdef void update_d12_lil(self, game_state_t *game, int pos, object lil_matrix)

    cdef void clear_updated_string_cache(self, game_state_t *game) nogil

    cdef void clear_planes(self) nogil

    cdef object get_tensor_as_csr_matrix(self, int color)
