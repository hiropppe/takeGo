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
    int updated[529]
    int updated_num

cdef int feature_size
cdef int response_size
cdef int save_atari_size
cdef int neighbor_size
cdef int nakade_size
cdef int x33_size
cdef int d12_size
cdef int response_start
cdef int save_atari_start
cdef int neighbor_start
cdef int nakade_start
cdef int x33_start
cdef int d12_start

cdef void initialize_feature(int nakade_size, int x33_size, int d12_size) nogil
cdef void update_all(game_state_t *game) nogil
cdef void update(game_state_t *game) nogil
cdef void update_save_atari(rollout_feature_t *feature, game_state_t *game, string_t *string) nogil
cdef void update_neighbor(rollout_feature_t *feature, game_state_t *game, int pos) nogil
cdef void update_d12(rollout_feature_t *feature, game_state_t *game, int prev_pos, int prev_color) nogil
cdef void update_3x3(rollout_feature_t *feature, game_state_t *game, int pos, int color) nogil
cdef void clear_neighbor(rollout_feature_t *feature) nogil
cdef void clear_d12(rollout_feature_t *feature) nogil
cdef void clear_onehot_index(rollout_feature_t *feature, int pos) nogil
cdef void clear_updated_string_cache(game_state_t *game) nogil
cdef void clear_planes(game_state_t *game) nogil
cdef bint memorize_updated(rollout_feature_t *feature, int pos) nogil


cdef class RolloutFeature:
    cdef:
        rollout_feature_t feature_planes[3]
        int feature_size
        int response_size
        int save_atari_size
        int neighbor_size
        int nakade_size
        int x33_size
        int d12_size
        int response_start
        int save_atari_start
        int neighbor_start
        int nakade_start
        int x33_start
        int d12_start

    cdef void update_all(self, game_state_t *game) nogil

    cdef void update(self, game_state_t *game) nogil

    cdef void update_save_atari(self, rollout_feature_t *feature, game_state_t *game, string_t *string) nogil

    cdef void update_neighbor(self, rollout_feature_t *feature, game_state_t *game, int pos) nogil

    cdef void update_d12(self, rollout_feature_t *feature, game_state_t *game, int prev_pos, int prev_color) nogil

    cdef void update_3x3(self, rollout_feature_t *feature, game_state_t *game, int pos, int color) nogil

    cdef void clear_neighbor(self, rollout_feature_t *feature) nogil

    cdef void clear_d12(self, rollout_feature_t *feature) nogil

    cdef void clear_onehot_index(self, rollout_feature_t *feature, int pos) nogil

    cdef void clear_updated_string_cache(self, game_state_t *game) nogil

    cdef void clear_planes(self) nogil

    cdef bint memorize_updated(self, rollout_feature_t *feature, int pos) nogil
