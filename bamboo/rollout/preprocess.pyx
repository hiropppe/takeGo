# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
from __future__ import division

import h5py as h5
import numpy as np

cimport numpy as np

from libc.math cimport exp as cexp
from libc.math cimport round as cround
from libc.stdlib cimport rand, RAND_MAX
from libc.stdio cimport printf

from bamboo.go.board cimport PURE_BOARD_SIZE, BOARD_MAX, PURE_BOARD_MAX, S_EMPTY, S_BLACK, S_WHITE, S_OB, PASS, STRING_EMPTY_END, OB_SIZE
from bamboo.go.board cimport FLIP_COLOR, POS, Y
from bamboo.go.board cimport game_state_t, rollout_feature_t, pure_board_max, onboard_index, onboard_pos
from bamboo.go.board cimport is_legal, get_neighbor4, get_neighbor8, get_neighbor8_in_order, get_md12

from bamboo.rollout.nakade cimport NOT_NAKADE, get_nakade_index, get_nakade_id, get_nakade_pos
from bamboo.rollout.pattern cimport x33_hash, x33_hashmap
from bamboo.rollout.pattern cimport d12_hash, d12_hashmap, d12_pos_mt
from bamboo.go.printer cimport print_board


cpdef void initialize_const(int nakade_feature_size,
                            int x33_feature_size,
                            int d12_feature_size):
    global response_size, save_atari_size, neighbor_size, nakade_size, x33_size, d12_size
    global response_start, save_atari_start, neighbor_start, nakade_start, x33_start, d12_start
    global feature_size

    response_size = 1
    save_atari_size = 1
    neighbor_size = 8
    nakade_size = nakade_feature_size
    x33_size = x33_feature_size
    d12_size = d12_feature_size

    response_start = 0
    save_atari_start = response_start + response_size
    neighbor_start = save_atari_start + save_atari_size
    nakade_start = neighbor_start + neighbor_size
    x33_start = nakade_start + nakade_size
    d12_start = x33_start + x33_size

    feature_size = d12_start + d12_size


cdef void initialize_rollout(game_state_t *game) nogil:
    initialize_planes(game)
    initialize_probs(game)


cdef void initialize_planes(game_state_t *game) nogil:
    cdef rollout_feature_t *black
    cdef rollout_feature_t *white
    cdef int i, j

    black = &game.rollout_feature_planes[<int>S_BLACK]
    white = &game.rollout_feature_planes[<int>S_WHITE]

    black.color = <int>S_BLACK
    white.color = <int>S_WHITE

    for i in range(6):
        for j in range(PURE_BOARD_MAX):
            black.tensor[i][j] = -1
            white.tensor[i][j] = -1

    black.prev_d12_num = 0
    white.prev_d12_num = 0

    black.prev_nakade = NOT_NAKADE
    white.prev_nakade = NOT_NAKADE

    black.updated[0] = BOARD_MAX
    black.updated_num = 0
    white.updated[0] = BOARD_MAX
    white.updated_num = 0


cdef void initialize_probs(game_state_t *game) nogil:
    cdef int i, j, k

    for i in range(S_OB):
        game.rollout_logits_sum[i] = .0
        for j in range(PURE_BOARD_MAX):
            game.rollout_probs[i][j] = .0
            game.rollout_logits[i][j] = .0
        for k in range(PURE_BOARD_SIZE):
            game.rollout_row_probs[i][k] = .0


cdef void update_rollout(game_state_t *game) nogil:
    update_planes(game)
    update_probs(game)


cdef void update_planes_all(game_state_t *game) nogil:
    cdef rollout_feature_t *current_feature
    cdef int current_color = <int>game.current_color
    cdef int prev_pos, prev_color
    cdef string_t *string
    cdef int pos
    cdef int i

    initialize_rollout(game)

    if game.moves == 0:
        return

    current_feature = &game.rollout_feature_planes[current_color]

    prev_pos = game.record[game.moves - 1].pos
    prev_color = game.record[game.moves - 1].color

    if prev_pos != PASS:
        update_neighbor(current_feature, game, prev_pos)
        update_d12(current_feature, game, prev_pos, prev_color)

    for i in range(pure_board_max):
        pos = onboard_pos[i]
        if game.board[pos] == S_EMPTY:
            update_3x3(current_feature, game, pos, current_color)
        else:
            string = &game.string[game.string_id[pos]]
            update_save_atari(current_feature, game, string)


cdef void update_planes(game_state_t *game) nogil:
    cdef rollout_feature_t *current_feature
    cdef int current_color = <int>game.current_color
    cdef int prev_pos, prev_color
    cdef int prev2_pos
    cdef int updated_string_num
    cdef int *updated_string_id
    cdef string_t *updated_string
    cdef int update_pos
    cdef int i

    if game.moves == 0:
        return

    current_feature = &game.rollout_feature_planes[current_color]

    current_feature.updated[0] = BOARD_MAX
    current_feature.updated_num = 0

    prev_pos = game.record[game.moves - 1].pos
    prev_color = game.record[game.moves - 1].color
    
    clear_onehot_index(current_feature, prev_pos)
    memorize_updated(current_feature, prev_pos)

    if game.moves > 1:
        prev2_pos = game.record[game.moves - 2].pos

        clear_onehot_index(current_feature, prev2_pos)
        memorize_updated(current_feature, prev2_pos)

    if prev_pos == PASS:
        clear_neighbor(current_feature)
        clear_nakade(current_feature)
        clear_d12(current_feature)
    else:
        update_neighbor(current_feature, game, prev_pos)
        update_nakade(current_feature, game, prev_color)
        update_d12(current_feature, game, prev_pos, prev_color)

    updated_string_num = game.updated_string_num[current_color]
    updated_string_id = game.updated_string_id[current_color]
    for i in range(updated_string_num):
        updated_string = &game.string[updated_string_id[i]]
        if updated_string.flag:
            update_save_atari(current_feature, game, updated_string)
            update_pos = updated_string.empty[0]
            while update_pos != STRING_EMPTY_END:
                update_3x3(current_feature, game, update_pos, current_color)
                update_pos = updated_string.empty[update_pos]


    # clear updated string memo for next feature calculation
    clear_updated_string_cache(game)


cdef void update_save_atari(rollout_feature_t *feature, game_state_t *game, string_t *string) nogil:
    """ Save atari 1 Move saves stone(s) from capture
    """
    cdef int last_lib
    cdef int neighbor4[4]
    cdef int neighbor_pos, neighbor_string_id
    cdef string_t *neighbor_string
    cdef int libs_after_move
    cdef int i
    cdef bint flag = False

    global save_atari_start

    libs_after_move = 0
    if string.libs == 1 and string.color == game.current_color:
        last_lib = string.lib[0]
        get_neighbor4(neighbor4, last_lib)
        for i in range(4):
            neighbor_pos = neighbor4[i]
            neighbor_string_id = game.string_id[neighbor_pos]
            if neighbor_string_id:
                neighbor_string = &game.string[neighbor_string_id]
                if neighbor_string.libs > 1 and neighbor_string.color == game.current_color:
                    libs_after_move += neighbor_string.libs - 1
            elif game.board[neighbor_pos] != S_OB:
                libs_after_move += 1

            if libs_after_move >= 2:
                flag = True
                break

        if flag:
            feature.tensor[F_SAVE_ATARI][onboard_index[last_lib]] = save_atari_start
            memorize_updated(feature, last_lib)


cdef void update_neighbor(rollout_feature_t *feature, game_state_t *game, int pos) nogil:
    """ Move is 8-connected to previous move
    """
    cdef int neighbor8[8]
    cdef int neighbor_pos, empty_neighbor_ix
    cdef int i

    global neighbor_start

    for i in range(feature.prev_neighbor8_num):
        feature.tensor[F_NEIGHBOR][feature.prev_neighbor8[i]] = -1

    get_neighbor8_in_order(neighbor8, pos)

    feature.prev_neighbor8_num = 0
    for i in range(8):
        neighbor_pos = neighbor8[i]
        if game.board[neighbor_pos] == S_EMPTY:
            empty_neighbor_ix = onboard_index[neighbor_pos]
            feature.tensor[F_NEIGHBOR][empty_neighbor_ix] = neighbor_start + i
            # memorize previous neighbor position
            feature.prev_neighbor8[feature.prev_neighbor8_num] = empty_neighbor_ix
            feature.prev_neighbor8_num += 1
            memorize_updated(feature, neighbor_pos)


cdef void update_nakade(rollout_feature_t *feature, game_state_t *game, int prev_color) nogil:
    cdef int capture_num
    cdef int *capture_pos
    cdef int nakade_index, nakade_id, nakade_pos, nakade_pure_pos

    global nakade_start

    clear_nakade(feature)

    capture_num = game.capture_num[prev_color]
    capture_pos = game.capture_pos[prev_color]

    if capture_num < 3 or 6 < capture_num:
        return

    # index for each number of capture
    nakade_index = get_nakade_index(capture_num, capture_pos)
    if nakade_index != NOT_NAKADE:
        nakade_id = get_nakade_id(capture_num, nakade_index)
        nakade_pos = get_nakade_pos(capture_num, capture_pos, nakade_index)
        nakade_pure_pos = onboard_index[nakade_pos]
        feature.tensor[F_NAKADE][nakade_pure_pos] = nakade_start + nakade_id
        feature.prev_nakade = nakade_pure_pos 
        memorize_updated(feature, nakade_pos)


cdef void update_d12(rollout_feature_t *feature, game_state_t *game, int prev_pos, int prev_color) nogil:
    """ Move matches 12-point diamond pattern near previous move
    """
    cdef int i, pos
    cdef int empty_ix[12]
    cdef int empty_pos[12]
    cdef int each_empty_pos
    cdef int n_empty_val = 0
    cdef int *n_empty = &n_empty_val
    cdef unsigned long long hash, positional_hash
    cdef int pax_ix
    cdef int empty_onboard_ix

    global response_start, d12_start

    # clear previous d12 positions
    for i in range(feature.prev_d12_num):
        pos = feature.prev_d12[i]
        feature.tensor[F_RESPONSE][pos] = -1
        feature.tensor[F_RESPONSE_PAT][pos] = -1

    feature.prev_d12_num = 0
    hash = d12_hash(game, prev_pos, prev_color, empty_ix, empty_pos, n_empty)
    for i in range(n_empty_val):
        positional_hash = hash ^ d12_pos_mt[1 << empty_ix[i]] 
        each_empty_pos = empty_pos[i]
        if d12_hashmap.find(positional_hash) == d12_hashmap.end():
            empty_onboard_ix = onboard_index[each_empty_pos]
            feature.tensor[F_RESPONSE][empty_onboard_ix] = -1 
            feature.tensor[F_RESPONSE_PAT][empty_onboard_ix] = -1
        else:
            pat_ix = d12_start + d12_hashmap[positional_hash]
            empty_onboard_ix = onboard_index[each_empty_pos]
            # set response(?) and response pattern
            feature.tensor[F_RESPONSE][empty_onboard_ix] = response_start
            feature.tensor[F_RESPONSE_PAT][empty_onboard_ix] = pat_ix
            # memorize previous d12 position
            feature.prev_d12[feature.prev_d12_num] = empty_onboard_ix
            feature.prev_d12_num += 1

        memorize_updated(feature, each_empty_pos)


cdef void update_3x3(rollout_feature_t *feature, game_state_t *game, int pos, int color) nogil:
    """ Move matches 3 × 3 pattern around move
    """
    cdef unsigned long long hash
    cdef int pat_ix

    global x33_start

    hash = x33_hash(game, pos, color)
    if x33_hashmap.find(hash) == x33_hashmap.end():
        feature.tensor[F_NON_RESPONSE_PAT][onboard_index[pos]] = -1
    else:
        pat_ix = x33_start + x33_hashmap[hash]
        feature.tensor[F_NON_RESPONSE_PAT][onboard_index[pos]] = pat_ix

    memorize_updated(feature, pos)


cdef void clear_neighbor(rollout_feature_t *feature) nogil:
    cdef int i, pos

    for i in range(feature.prev_neighbor8_num):
        pos = feature.prev_neighbor8[i]
        feature.tensor[F_NEIGHBOR][pos] = -1
        memorize_updated(feature, onboard_pos[pos])


cdef void clear_d12(rollout_feature_t *feature) nogil:
    cdef int i, pos

    for i in range(feature.prev_d12_num):
        pos = feature.prev_d12[i]
        feature.tensor[F_RESPONSE][pos] = -1
        feature.tensor[F_RESPONSE_PAT][pos] = -1
        memorize_updated(feature, onboard_pos[pos])


cdef void clear_nakade(rollout_feature_t *feature) nogil:
    if feature.prev_nakade != NOT_NAKADE:
        feature.tensor[F_NAKADE][feature.prev_nakade] = -1
        memorize_updated(feature, onboard_pos[feature.prev_nakade])


cdef void clear_onehot_index(rollout_feature_t *feature, int pos) nogil:
    if pos != PASS:
        feature.tensor[F_RESPONSE][onboard_index[pos]] = -1 
        feature.tensor[F_SAVE_ATARI][onboard_index[pos]] = -1 
        feature.tensor[F_NAKADE][onboard_index[pos]] = -1 
        feature.tensor[F_RESPONSE_PAT][onboard_index[pos]] = -1 
        feature.tensor[F_NON_RESPONSE_PAT][onboard_index[pos]] = -1 


cdef void clear_updated_string_cache(game_state_t *game) nogil:
    cdef int *updated_string_num

    updated_string_num = &game.updated_string_num[<int>game.current_color]
    updated_string_num[0] = 0


cdef bint memorize_updated(rollout_feature_t *feature, int pos) nogil:
    if feature.updated[pos] != 0:
        return False

    feature.updated[pos] = feature.updated[0]
    feature.updated[0] = pos

    feature.updated_num += 1

    return True


cpdef void set_rollout_parameter(object weights_hdf5):
    cdef int i

    global rollout_weights

    weights_data = h5.File(weights_hdf5, 'r')
    W = weights_data['W']
    for i in range(W.shape[0]):
        rollout_weights[i] = W[i]


cdef void update_probs_all(game_state_t *game) nogil:
    cdef int color
    cdef rollout_feature_t *feature
    cdef double *probs
    cdef double *row_probs
    cdef double *logits
    cdef int pos
    cdef int i, j

    color = <int>game.current_color
    feature = &game.rollout_feature_planes[color]
    probs = game.rollout_probs[color]
    row_probs = game.rollout_row_probs[color]
    logits = game.rollout_logits[color]

    for i in range(PURE_BOARD_MAX):
        if is_legal(game, onboard_pos[i], color):
            for j in range(6):
                if feature.tensor[j][i] != -1:
                    logits[i] += rollout_weights[feature.tensor[j][i]]
            logits[i] = cexp(logits[i])
            game.rollout_logits_sum[color] += logits[i]
        else:
            logits[i] = .0

    if game.rollout_logits_sum[color] > .0:
        norm_probs(probs, row_probs, logits, game.rollout_logits_sum[color])


cdef void update_probs(game_state_t *game) nogil:
    cdef int color
    cdef rollout_feature_t *feature
    cdef double *probs
    cdef double *row_probs
    cdef double *logits
    cdef int pos, tmp_pos
    cdef int pure_pos, pure_row
    cdef double updated_sum = .0
    cdef double updated_old_sum = .0
    cdef int i, j

    color = <int>game.current_color
    feature = &game.rollout_feature_planes[color]
    probs = game.rollout_probs[color]
    row_probs = game.rollout_row_probs[color]
    logits = game.rollout_logits[color]

    pos = feature.updated[0]
    while pos != BOARD_MAX:
        if is_legal(game, pos, color):
            pure_pos = onboard_index[pos]
            updated_old_sum += logits[pure_pos]
            logits[pure_pos] = .0
            for j in range(6):
                if feature.tensor[j][pure_pos] != -1:
                    logits[pure_pos] += rollout_weights[feature.tensor[j][pure_pos]]
            logits[pure_pos] = cexp(logits[pure_pos])
            updated_sum += logits[pure_pos]
        # Must be cleared for next feature calculation
        tmp_pos = feature.updated[pos]
        feature.updated[pos] = 0
        pos = tmp_pos

    game.rollout_logits_sum[color] = game.rollout_logits_sum[color] - updated_old_sum + updated_sum

    if game.rollout_logits_sum[color] > .0:
        norm_probs(probs, row_probs, logits, game.rollout_logits_sum[color])


cdef void norm_probs(double *probs, double *row_probs, double *logits, double logits_sum) nogil:
    cdef int pure_row
    cdef int i

    for i in range(PURE_BOARD_MAX):
        pure_row = Y(i, PURE_BOARD_SIZE)
        row_probs[pure_row] -= probs[i]
        probs[i] = logits[i]/logits_sum
        row_probs[pure_row] += probs[i]


cdef void set_illegal(game_state_t *game, int pos) nogil:
    cdef int color
    cdef rollout_feature_t *feature
    cdef double *probs
    cdef double *row_probs
    cdef double *logits
    cdef int pure_pos
    cdef int i, j

    color = <int>game.current_color
    probs = game.rollout_probs[color]
    row_probs = game.rollout_row_probs[color]
    logits = game.rollout_logits[color]

    pure_pos = onboard_index[pos]
    game.rollout_logits_sum[color] -= logits[pure_pos]
    logits[pure_pos] = .0

    norm_probs(probs, row_probs, logits, game.rollout_logits_sum[color])


cdef int choice_rollout_move(game_state_t *game) nogil:
    cdef int color
    cdef double *probs
    cdef double *row_probs
    cdef double random_number
    cdef int row, pos

    color = <int>game.current_color
    probs = game.rollout_probs[color]
    row_probs = game.rollout_row_probs[color]

    if game.rollout_logits_sum[color] <= .001:
        return PASS

    random_number = <double>rand() / RAND_MAX
    row = 0
    while random_number > row_probs[row]:
        random_number -= row_probs[row]
        row += 1

    pos = row * PURE_BOARD_SIZE
    while True:
        random_number -= probs[pos]
        if random_number <= 0:
            break
        pos += 1

    return onboard_pos[pos]


cdef void set_debug(bint dbg) nogil:
    global debug
    debug = dbg
