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

from bamboo.board cimport PURE_BOARD_SIZE, BOARD_MAX, PURE_BOARD_MAX, S_EMPTY, S_BLACK, S_WHITE, S_OB, PASS, STRING_EMPTY_END
from bamboo.board cimport FLIP_COLOR, POS, Y, DIS, NORTH, WEST, EAST, SOUTH
from bamboo.board cimport game_state_t, rollout_feature_t, pure_board_max
from bamboo.board cimport board_size, onboard_index, onboard_pos, board_x, board_y, move_dis, liberty_end
from bamboo.board cimport neighbor4_pos, neighbor8_pos, neighbor8_seq_pos
from bamboo.board cimport is_legal, is_legal_not_eye

from bamboo.nakade cimport NOT_NAKADE, get_nakade_index, get_nakade_id, get_nakade_pos
from bamboo.local_pattern cimport x33_hash, x33_hashmap
from bamboo.local_pattern cimport d12_rsp_hash, d12_rspos_hash, d12_pos_mt, d12_rsp_hashmap
from bamboo.local_pattern cimport d12_hash, d12_hashmap


cpdef void initialize_rollout_const(int nakade_feature_size,
                                    int x33_feature_size,
                                    int d12_rsp_feature_size,
                                    int d12_feature_size,
                                    bint pos_aware_d12=False):
    global rollout_feature_size
    global response_size, save_atari_size, neighbor_size, nakade_size, x33_size, d12_size
    global response_start, save_atari_start, neighbor_start, nakade_start, x33_start, d12_rsp_start
    global tree_feature_size
    global self_atari_size, last_move_distance_size, d12_size
    global self_atari_start, last_move_distance_start, d12_start
    global use_pos_aware_d12

    response_size = 1
    save_atari_size = 1
    neighbor_size = 8
    nakade_size = nakade_feature_size
    x33_size = x33_feature_size
    d12_rsp_size = d12_rsp_feature_size
    self_atari_size = 1
    last_move_distance_size = 34
    d12_size = d12_feature_size

    response_start = 0
    save_atari_start = response_start + response_size
    neighbor_start = save_atari_start + save_atari_size
    nakade_start = neighbor_start + neighbor_size
    x33_start = nakade_start + nakade_size
    d12_rsp_start = x33_start + x33_size
    self_atari_start = d12_rsp_start + d12_rsp_size
    last_move_distance_start = self_atari_start + self_atari_size
    d12_start = last_move_distance_start + last_move_distance_size

    rollout_feature_size = d12_rsp_start + d12_rsp_size
    tree_feature_size = d12_start + d12_size

    use_pos_aware_d12 = pos_aware_d12


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

    for i in range(9):
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
        clear_d12_rsp(current_feature)
    else:
        update_neighbor(current_feature, game, prev_pos)
        update_nakade(current_feature, game, prev_color)
        update_d12_rsp(current_feature, game, prev_pos, prev_color)

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
    cdef int neighbor_pos
    cdef string_t *neighbor_string
    cdef int libs_after_move
    cdef int i
    cdef bint flag = False

    global save_atari_start

    libs_after_move = 0
    if string.libs == 1 and string.color == game.current_color:
        last_lib = string.lib[0]
        for i in range(4):
            neighbor_pos = neighbor4_pos[last_lib][i]
            neighbor_string = &game.string[game.string_id[neighbor_pos]]
            if neighbor_string.flag:
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
    cdef int neighbor_pos, empty_neighbor_ix
    cdef int i

    global neighbor_start

    for i in range(feature.prev_neighbor8_num):
        feature.tensor[F_NEIGHBOR][feature.prev_neighbor8[i]] = -1

    feature.prev_neighbor8_num = 0
    for i in range(8):
        neighbor_pos = neighbor8_seq_pos[pos][i]
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


cdef void update_d12_rsp(rollout_feature_t *feature, game_state_t *game, int prev_pos, int prev_color) nogil:
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

    global response_start, d12_rsp_start

    # clear previous d12 positions
    for i in range(feature.prev_d12_num):
        pos = feature.prev_d12[i]
        feature.tensor[F_RESPONSE][pos] = -1
        feature.tensor[F_D12_RSP_PAT][pos] = -1

    feature.prev_d12_num = 0
    if use_pos_aware_d12:
        hash = d12_rspos_hash(game, prev_pos, prev_color, empty_ix, empty_pos, n_empty)
        for i in range(n_empty_val):
            positional_hash = hash ^ d12_pos_mt[1 << empty_ix[i]] 
            each_empty_pos = empty_pos[i]
            if d12_rsp_hashmap.find(positional_hash) == d12_rsp_hashmap.end():
                empty_onboard_ix = onboard_index[each_empty_pos]
                feature.tensor[F_RESPONSE][empty_onboard_ix] = -1 
                feature.tensor[F_D12_RSP_PAT][empty_onboard_ix] = -1
            else:
                pat_ix = d12_rsp_start + d12_rsp_hashmap[positional_hash]
                empty_onboard_ix = onboard_index[each_empty_pos]
                # set response(?) and response pattern
                feature.tensor[F_RESPONSE][empty_onboard_ix] = response_start
                feature.tensor[F_D12_RSP_PAT][empty_onboard_ix] = pat_ix
                # memorize previous d12 position
                feature.prev_d12[feature.prev_d12_num] = empty_onboard_ix
                feature.prev_d12_num += 1

            memorize_updated(feature, each_empty_pos)
    else:
        hash = d12_rsp_hash(game, prev_pos, prev_color, empty_ix, empty_pos, n_empty)
        for i in range(n_empty_val):
            each_empty_pos = empty_pos[i]
            if d12_rsp_hashmap.find(hash) == d12_rsp_hashmap.end():
                empty_onboard_ix = onboard_index[each_empty_pos]
                feature.tensor[F_RESPONSE][empty_onboard_ix] = -1 
                feature.tensor[F_D12_RSP_PAT][empty_onboard_ix] = -1
            else:
                pat_ix = d12_rsp_start + d12_rsp_hashmap[hash]
                empty_onboard_ix = onboard_index[each_empty_pos]
                # set response(?) and response pattern
                feature.tensor[F_RESPONSE][empty_onboard_ix] = response_start
                feature.tensor[F_D12_RSP_PAT][empty_onboard_ix] = pat_ix
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
        feature.tensor[F_X33_PAT][onboard_index[pos]] = -1
    else:
        pat_ix = x33_start + x33_hashmap[hash]
        feature.tensor[F_X33_PAT][onboard_index[pos]] = pat_ix

    memorize_updated(feature, pos)


cdef void clear_neighbor(rollout_feature_t *feature) nogil:
    cdef int i, pos

    for i in range(feature.prev_neighbor8_num):
        pos = feature.prev_neighbor8[i]
        feature.tensor[F_NEIGHBOR][pos] = -1
        memorize_updated(feature, onboard_pos[pos])


cdef void clear_d12_rsp(rollout_feature_t *feature) nogil:
    cdef int i, pos

    for i in range(feature.prev_d12_num):
        pos = feature.prev_d12[i]
        feature.tensor[F_RESPONSE][pos] = -1
        feature.tensor[F_D12_RSP_PAT][pos] = -1
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
        feature.tensor[F_D12_RSP_PAT][onboard_index[pos]] = -1
        feature.tensor[F_X33_PAT][onboard_index[pos]] = -1


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


cpdef void set_tree_parameter(object weights_hdf5):
    cdef int i

    global tree_weights

    weights_data = h5.File(weights_hdf5, 'r')
    W = weights_data['W']
    for i in range(W.shape[0]):
        tree_weights[i] = W[i]


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


cdef void get_rollout_probs(game_state_t *game, double *probs) nogil:
    cdef int i
    cdef int color = <int>game.current_color

    for i in range(PURE_BOARD_MAX):
        probs[i] = game.rollout_probs[color][i]


cdef void update_tree_planes_all(game_state_t *game) nogil:
    cdef int current_color
    cdef rollout_feature_t *current_feature
    cdef int pos
    cdef int i, j

    current_color = <int>game.current_color
    current_feature = &game.rollout_feature_planes[current_color]

    for i in range(pure_board_max):
        pos = onboard_pos[i]
        if is_legal_not_eye(game, pos, current_color):
            update_self_atari(current_feature, game, pos, current_color)
            update_last_move_distance(current_feature, game, pos)
            update_d12(current_feature, game, pos, current_color) 
        else:
            current_feature.tensor[F_SELF_ATARI][i] = -1
            current_feature.tensor[F_LAST_MOVE_DISTANCE][i] = -1
            current_feature.tensor[F_D12_PAT][i] = -1


cdef void get_tree_probs(game_state_t *game, double probs[361]) nogil:
    cdef int color
    cdef rollout_feature_t *feature
    cdef double logits[361]
    cdef double logits_sum
    cdef int i

    color = <int>game.current_color
    feature = &game.rollout_feature_planes[color]
    logits_sum = .0

    for i in range(PURE_BOARD_MAX):
        logits[i] = .0
        if is_legal_not_eye(game, onboard_pos[i], color):
            for j in range(9):
                if feature.tensor[j][i] != -1:
                    logits[i] += tree_weights[feature.tensor[j][i]]
            logits[i] = cexp(logits[i])
            logits_sum += logits[i]

    for i in range(PURE_BOARD_MAX):
        probs[i] = logits[i]/logits_sum


cdef void update_self_atari(rollout_feature_t *feature, game_state_t *game, int pos, int color) nogil:
    cdef char *board = game.board
    cdef string_t *string = game.string
    cdef int *string_id = game.string_id
    cdef int other = FLIP_COLOR(color)
    cdef int already[4]
    cdef int already_num = 0
    cdef int lib, count = 0, libs = 0
    cdef int lib_candidate[10]
    cdef int i
    cdef int id
    cdef int north, west, east, south
    cdef bint checked

    global self_atari_start

    # 上下左右が空点なら呼吸点の候補に入れる
    north = NORTH(pos, board_size)
    west = WEST(pos)
    east = EAST(pos)
    south = SOUTH(pos, board_size)

    if board[north] == S_EMPTY:
        lib_candidate[libs] = north 
        libs += 1
    if board[west] == S_EMPTY:
        lib_candidate[libs] = west
        libs += 1
    if board[east] == S_EMPTY:
        lib_candidate[libs] = east
        libs += 1
    if board[south] == S_EMPTY:
        lib_candidate[libs] = south
        libs += 1

    # 空点
    if libs >= 2:
        feature.tensor[F_SELF_ATARI][onboard_index[pos]] = -1
        return

    # 上を調べる
    if board[north] == color:
        id = string_id[north]
        if string[id].libs > 2:
            feature.tensor[F_SELF_ATARI][onboard_index[pos]] = -1
            return
        lib = string[id].lib[0]
        count = 0
        while lib != liberty_end:
            if lib != pos:
                checked = False
                for i in range(libs):
                    if lib_candidate[i] == lib:
                        checked = True
                        break
                if not checked:
                    lib_candidate[libs + count] = lib
                    count += 1
            lib = string[id].lib[lib]
        libs += count
        already[already_num] = id
        already_num += 1
        if libs >= 2:
            feature.tensor[F_SELF_ATARI][onboard_index[pos]] = -1
            return
    elif board[north] == other and string[string_id[north]].libs == 1:
        feature.tensor[F_SELF_ATARI][onboard_index[pos]] = -1
        return

    # 左を調べる
    if board[west] == color:
        id = string_id[west]
        if already[0] != id:
            if string[id].libs > 2:
                feature.tensor[F_SELF_ATARI][onboard_index[pos]] = -1
                return
            lib = string[id].lib[0]
            count = 0
            while lib != liberty_end:
                if lib != pos:
                    checked = False
                    for i in range(libs):
                        if lib_candidate[i] == lib:
                            checked = True
                            break
                    if not checked:
                        lib_candidate[libs + count] = lib
                        count += 1
                lib = string[id].lib[lib]
            libs += count
            already[already_num] = id
            already_num += 1
            if libs >= 2:
                feature.tensor[F_SELF_ATARI][onboard_index[pos]] = -1
                return
    elif board[west] == other and string[string_id[west]].libs == 1:
        feature.tensor[F_SELF_ATARI][onboard_index[pos]] = -1
        return

    # 右を調べる
    if board[east] == color:
        id = string_id[east];
        if already[0] != id and already[1] != id:
            if string[id].libs > 2:
                feature.tensor[F_SELF_ATARI][onboard_index[pos]] = -1
                return
            lib = string[id].lib[0]
            count = 0
            while lib != liberty_end:
                if lib != pos:
                    checked = False
                    for i in range(libs):
                        if lib_candidate[i] == lib:
                            checked = True
                            break
                    if not checked:
                        lib_candidate[libs + count] = lib
                        count += 1
                lib = string[id].lib[lib]
            libs += count
            already[already_num] = id
            already_num += 1
            if libs >= 2:
                feature.tensor[F_SELF_ATARI][onboard_index[pos]] = -1
                return
    elif board[east] == other and string[string_id[east]].libs == 1:
        feature.tensor[F_SELF_ATARI][onboard_index[pos]] = -1
        return

    # 下を調べる
    if board[south] == color:
        id = string_id[south]
        if already[0] != id and already[1] != id and already[2] != id:
            if string[id].libs > 2:
                feature.tensor[F_SELF_ATARI][onboard_index[pos]] = -1
                return
            lib = string[id].lib[0]
            count = 0
            while lib != liberty_end:
                if lib != pos:
                    checked = False
                    for i in range(libs):
                        if lib_candidate[i] == lib:
                            checked = True
                            break
                    if not checked:
                        lib_candidate[libs + count] = lib
                        count += 1
                lib = string[id].lib[lib]
            libs += count
            already[already_num] = id
            already_num += 1
            if libs >= 2:
                feature.tensor[F_SELF_ATARI][onboard_index[pos]] = -1
                return
    elif board[south] == other and string[string_id[south]].libs == 1:
        feature.tensor[F_SELF_ATARI][onboard_index[pos]] = -1
        return

    feature.tensor[F_SELF_ATARI][onboard_index[pos]] = self_atari_start


cdef void update_last_move_distance(rollout_feature_t *feature, game_state_t *game, int pos) nogil:
    cdef int prev_pos, prev2_pos
    cdef int prev_dis = 0, prev2_dis = 0

    global last_move_distance_start

    if game.moves == 0:
        return
    elif game.moves > 0:
        prev_pos = game.record[game.moves - 1].pos
        if prev_pos != PASS:
            prev_dis = DIS(pos, prev_pos, board_x, board_y, move_dis)

        if game.moves > 1:
            prev2_pos = game.record[game.moves - 2].pos
            if prev2_pos != PASS:
                prev2_dis = DIS(pos, prev2_pos, board_x, board_y, move_dis)

        feature.tensor[F_LAST_MOVE_DISTANCE][onboard_index[pos]] = last_move_distance_start + prev_dis + prev2_dis


cdef void update_d12(rollout_feature_t *feature, game_state_t *game, int pos, int color) nogil:
    """ Move matches 12-point diamond pattern centred around move 
    """
    cdef unsigned long long hash
    cdef int pat_ix

    global d12_start

    hash = d12_hash(game, pos, color)
    if d12_hashmap.find(hash) == d12_hashmap.end():
        feature.tensor[F_D12_PAT][onboard_index[pos]] = -1
    else:
        pat_ix = d12_start + d12_hashmap[hash]
        feature.tensor[F_D12_PAT][onboard_index[pos]] = pat_ix


cdef void set_debug(bint dbg) nogil:
    global debug
    debug = dbg
