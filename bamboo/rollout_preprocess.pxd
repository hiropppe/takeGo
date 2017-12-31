from bamboo.board cimport game_state_t, rollout_feature_t, string_t

cdef extern from "ray.h":
    ctypedef enum:
        F_RESPONSE
        F_SAVE_ATARI
        F_NEIGHBOR
        F_NAKADE
        F_D12_RSP_PAT
        F_X33_PAT
        F_SELF_ATARI
        F_LAST_MOVE_DISTANCE
        F_D12_PAT

cdef enum:
    MOVE_DISTANCE_MAX = 17

cdef bint debug

cdef int rollout_feature_size
cdef int tree_feature_size

cdef int response_size
cdef int save_atari_size
cdef int neighbor_size
cdef int nakade_size
cdef int x33_size
cdef int d12_rsp_size
cdef int self_atari_size
cdef int last_move_distance_size
cdef int d12_size

cdef int response_start
cdef int save_atari_start
cdef int neighbor_start
cdef int nakade_start
cdef int x33_start
cdef int d12_rsp_start
cdef int self_atari_start
cdef int last_move_distance_start
cdef int d12_start

cdef double rollout_weights[100000]
cdef double tree_weights[100000]

cdef bint use_pos_aware_d12 = False

cpdef void initialize_rollout_const(int nakade_size, int x33_size, int d12_rsp_size, int d12_size, bint pos_aware_d12=?)
cpdef void set_rollout_parameter(object weights_hdf5) 
cpdef void set_tree_parameter(object weights_hdf5)

cdef void initialize_rollout(game_state_t *game) nogil
cdef void update_rollout(game_state_t *game) nogil

cdef void initialize_planes(game_state_t *game) nogil
cdef void update_planes(game_state_t *game) nogil
cdef void update_save_atari(rollout_feature_t *feature, game_state_t *game, string_t *string) nogil
cdef void update_neighbor(rollout_feature_t *feature, game_state_t *game, int pos) nogil
cdef void update_nakade(rollout_feature_t *feature, game_state_t *game, int prev_color) nogil
cdef void update_d12_rsp(rollout_feature_t *feature, game_state_t *game, int prev_pos, int prev_color) nogil
cdef void update_3x3(rollout_feature_t *feature, game_state_t *game, int pos, int color) nogil
cdef void clear_neighbor(rollout_feature_t *feature) nogil
cdef void clear_nakade(rollout_feature_t *feature) nogil
cdef void clear_d12_rsp(rollout_feature_t *feature) nogil
cdef void clear_onehot_index(rollout_feature_t *feature, int pos) nogil
cdef void clear_updated_string_cache(game_state_t *game) nogil
cdef bint memorize_updated(rollout_feature_t *feature, int pos) nogil

cdef void initialize_probs(game_state_t *game) nogil
cdef void update_probs(game_state_t *game) nogil
cdef void update_all_probs(game_state_t *game) nogil
cdef void norm_probs(double *probs, double *row_probs, double *logits, double logits_sum) nogil
cdef void set_illegal(game_state_t *game, int pos) nogil
cdef int choice_rollout_move(game_state_t *game) nogil
cdef void get_rollout_probs(game_state_t *game, double *probs) nogil

cdef void update_tree_planes_all(game_state_t *game) nogil
cdef void get_tree_probs(game_state_t *game, double *probs) nogil

cdef void set_debug(bint debug) nogil
