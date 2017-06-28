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
    int tensor[6][361] # hold one-hot index for each feature
    int prev_neighbor8[8]
    int prev_neighbor8_num
    int prev_d12[12]
    int prev_d12_num


cdef class RolloutFeature:
    cdef rollout_feature_t feature_planes[3]
    cdef int feature_size
    cdef int response_size, save_atari_size, neighbor_size, nakade_size, x33_size, d12_size
    cdef int response_start, save_atari_start, neighbor_start, nakade_start, x33_start, d12_start

    cdef void update_all(self, game_state_t *game) nogil

    cdef void update(self, game_state_t *game) nogil

    cdef void update_save_atari(self, game_state_t *game, string_t *string) nogil

    cdef void update_neighbor(self, game_state_t *game, int pos) nogil
    
    cdef void update_3x3(self, game_state_t *game, int pos) nogil

    cdef void update_d12(self, game_state_t *game, int prev_pos, int prev_color) nogil

    cdef void clear_neighbor(self, game_state_t *game) nogil

    cdef void clear_d12(self, game_state_t *game) nogil

    cdef void clear_onehot_index(self, game_state_t *game, int pos) nogil

    cdef void clear_updated_string_cache(self, game_state_t *game) nogil

    cdef void clear_planes(self) nogil
