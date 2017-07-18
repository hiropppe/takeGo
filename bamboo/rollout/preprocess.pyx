# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

import numpy as np

cimport numpy as np

from libc.stdio cimport printf

from bamboo.go.board cimport BOARD_MAX, PURE_BOARD_MAX, S_EMPTY, S_BLACK, S_WHITE, S_OB, PASS, STRING_EMPTY_END 
from bamboo.go.board cimport FLIP_COLOR
from bamboo.go.board cimport game_state_t, pure_board_max, onboard_index, onboard_pos
from bamboo.go.board cimport get_neighbor4, get_neighbor8, get_neighbor8_in_order, get_md12

from bamboo.rollout.pattern cimport x33_hash, x33_hashmap
from bamboo.rollout.pattern cimport d12_hash, d12_hashmap, d12_pos_mt


cdef void initialize_feature(int nakade_size, int x33_size, int d12_size) nogil:
    pass


cdef void update_all(game_state_t *game) nogil:
    pass


cdef void update(game_state_t *game) nogil:
    pass


cdef void update_save_atari(rollout_feature_t *feature, game_state_t *game, string_t *string) nogil:
    pass


cdef void update_neighbor(rollout_feature_t *feature, game_state_t *game, int pos) nogil:
    pass


cdef void update_d12(rollout_feature_t *feature, game_state_t *game, int prev_pos, int prev_color) nogil:
    pass


cdef void update_3x3(rollout_feature_t *feature, game_state_t *game, int pos, int color) nogil:
    pass


cdef void clear_neighbor(rollout_feature_t *feature) nogil:
    pass


cdef void clear_d12(rollout_feature_t *feature) nogil:
    pass


cdef void clear_onehot_index(rollout_feature_t *feature, int pos) nogil:
    pass


cdef void clear_updated_string_cache(game_state_t *game) nogil:
    pass


cdef void clear_planes(game_state_t *game) nogil:
    pass


cdef bint memorize_updated(rollout_feature_t *feature, int pos) nogil:
    pass


cdef class RolloutFeature:

    def __cinit__(self, int nakade_size=0, int x33_size=0, int d12_size=0):
        self.response_size = 1
        self.save_atari_size = 1
        self.neighbor_size = 8
        self.nakade_size = nakade_size
        self.x33_size = x33_size
        self.d12_size = d12_size

        self.response_start = 0
        self.save_atari_start = self.response_start + self.response_size
        self.neighbor_start = self.save_atari_start + self.save_atari_size
        self.nakade_start = self.neighbor_start + self.neighbor_size
        self.x33_start = self.nakade_start + self.nakade_size
        self.d12_start = self.x33_start + self.x33_size

        self.feature_size = self.d12_start + self.d12_size

        self.clear_planes()

    def __dealloc__(self):
        pass

    cdef void update_all(self, game_state_t *game) nogil:
        cdef rollout_feature_t *current_feature
        cdef int current_color = <int>game.current_color
        cdef int prev_pos, prev_color
        cdef string_t *string
        cdef int pos
        cdef int i

        if game.moves == 0:
            return

        current_feature = &self.feature_planes[<int>game.current_color]

        prev_pos = game.record[game.moves - 1].pos
        prev_color = game.record[game.moves - 1].color

        if prev_pos != PASS:
            self.update_neighbor(current_feature, game, prev_pos)
            self.update_d12(current_feature, game, prev_pos, prev_color)

        for i in range(pure_board_max):
            pos = onboard_pos[i]
            if game.board[pos] == S_EMPTY:
                self.update_3x3(current_feature, game, pos, current_color)
            else:
                string = &game.string[game.string_id[pos]]
                self.update_save_atari(current_feature, game, string)

    cdef void update(self, game_state_t *game) nogil:
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

        current_feature = &self.feature_planes[current_color]

        current_feature.updated[0] = BOARD_MAX
        current_feature.updated_num = 0

        prev_pos = game.record[game.moves - 1].pos
        prev_color = game.record[game.moves - 1].color
        self.clear_onehot_index(current_feature, prev_pos)
        self.memorize_updated(current_feature, prev_pos)

        if game.moves > 1:
            prev2_pos = game.record[game.moves - 2].pos
            self.clear_onehot_index(current_feature, prev2_pos)
            self.memorize_updated(current_feature, prev2_pos)

        if prev_pos == PASS:
            self.clear_neighbor(current_feature)
            self.clear_d12(current_feature)
        else:
            self.update_neighbor(current_feature, game, prev_pos)
            self.update_d12(current_feature, game, prev_pos, prev_color)

        updated_string_num = game.updated_string_num[current_color]
        updated_string_id = game.updated_string_id[current_color]
        for i in range(updated_string_num):
            updated_string = &game.string[updated_string_id[i]]
            self.update_save_atari(current_feature, game, updated_string)
            update_pos = updated_string.empty[0]
            while update_pos != STRING_EMPTY_END:
                self.update_3x3(current_feature, game, update_pos, current_color)
                update_pos = updated_string.empty[update_pos]

        # clear updated string memo for next feature calculation
        self.clear_updated_string_cache(game)

    cdef void clear_onehot_index(self, rollout_feature_t *feature, int pos) nogil:
        if pos != PASS:
            feature.tensor[RESPONSE][onboard_index[pos]] = -1 
            feature.tensor[SAVE_ATARI][onboard_index[pos]] = -1 
            feature.tensor[NAKADE][onboard_index[pos]] = -1 
            feature.tensor[RESPONSE_PAT][onboard_index[pos]] = -1 
            feature.tensor[NON_RESPONSE_PAT][onboard_index[pos]] = -1 

    cdef void update_3x3(self, rollout_feature_t *feature, game_state_t *game, int pos, int color) nogil:
        """ Move matches 3 × 3 pattern around move
        """
        cdef unsigned long long hash
        cdef int pat_ix

        hash = x33_hash(game, pos, color)
        if x33_hashmap.find(hash) == x33_hashmap.end():
            feature.tensor[NON_RESPONSE_PAT][onboard_index[pos]] = -1
        else:
            pat_ix = self.x33_start + x33_hashmap[hash]
            feature.tensor[NON_RESPONSE_PAT][onboard_index[pos]] = pat_ix

        self.memorize_updated(feature, pos)

    cdef void update_d12(self, rollout_feature_t *feature, game_state_t *game, int prev_pos, int prev_color) nogil:
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

        # clear previous d12 positions
        for i in range(feature.prev_d12_num):
            pos = feature.prev_d12[i]
            feature.tensor[RESPONSE][pos] = -1
            feature.tensor[RESPONSE_PAT][pos] = -1

        feature.prev_d12_num = 0
        hash = d12_hash(game, prev_pos, prev_color, empty_ix, empty_pos, n_empty)
        for i in range(n_empty_val):
            positional_hash = hash ^ d12_pos_mt[1 << empty_ix[i]] 
            each_empty_pos = empty_pos[i]
            if d12_hashmap.find(positional_hash) == d12_hashmap.end():
                empty_onboard_ix = onboard_index[each_empty_pos]
                feature.tensor[RESPONSE][empty_onboard_ix] = -1 
                feature.tensor[RESPONSE_PAT][empty_onboard_ix] = -1
            else:
                pat_ix = self.d12_start + d12_hashmap[positional_hash]
                empty_onboard_ix = onboard_index[each_empty_pos]
                # set response(?) and response pattern
                feature.tensor[RESPONSE][empty_onboard_ix] = self.response_start
                feature.tensor[RESPONSE_PAT][empty_onboard_ix] = pat_ix
                # memorize previous d12 position
                feature.prev_d12[feature.prev_d12_num] = empty_onboard_ix
                feature.prev_d12_num += 1

            self.memorize_updated(feature, each_empty_pos)

    cdef void update_save_atari(self, rollout_feature_t *feature, game_state_t *game, string_t *string) nogil:
        """ Save atari 1 Move saves stone(s) from capture
        """
        cdef int last_lib
        cdef int neighbor4[4]
        cdef int neighbor_pos, neighbor_string_id
        cdef string_t *neighbor_string
        cdef int libs_after_move
        cdef int i
        cdef bint flag = False

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
                feature.tensor[SAVE_ATARI][onboard_index[last_lib]] = self.save_atari_start
                self.memorize_updated(feature, last_lib)

    cdef void update_neighbor(self, rollout_feature_t *feature, game_state_t *game, int pos) nogil:
        """ Move is 8-connected to previous move
        """
        cdef int neighbor8[8]
        cdef int neighbor_pos, empty_neighbor_ix
        cdef int i

        for i in range(feature.prev_neighbor8_num):
            feature.tensor[NEIGHBOR][feature.prev_neighbor8[i]] = -1

        get_neighbor8_in_order(neighbor8, pos)

        feature.prev_neighbor8_num = 0
        for i in range(8):
            neighbor_pos = neighbor8[i]
            if game.board[neighbor_pos] == S_EMPTY:
                empty_neighbor_ix = onboard_index[neighbor_pos]
                feature.tensor[NEIGHBOR][empty_neighbor_ix] = self.neighbor_start + i
                # memorize previous neighbor position
                feature.prev_neighbor8[feature.prev_neighbor8_num] = empty_neighbor_ix
                feature.prev_neighbor8_num += 1
                self.memorize_updated(feature, neighbor_pos)

    cdef void clear_neighbor(self, rollout_feature_t *feature) nogil:
        cdef int i, pos

        for i in range(feature.prev_neighbor8_num):
            pos = feature.prev_neighbor8[i]
            feature.tensor[NEIGHBOR][pos] = -1
            self.memorize_updated(feature, pos)

    cdef void clear_d12(self, rollout_feature_t *feature) nogil:
        cdef int i, pos

        for i in range(feature.prev_d12_num):
            pos = feature.prev_d12[i]
            feature.tensor[RESPONSE][pos] = -1
            feature.tensor[RESPONSE_PAT][pos] = -1
            self.memorize_updated(feature, pos)

    cdef void clear_updated_string_cache(self, game_state_t *game) nogil:
        cdef int *updated_string_num

        updated_string_num = &game.updated_string_num[<int>game.current_color]
        updated_string_num[0] = 0

    cdef void clear_planes(self) nogil:
        cdef rollout_feature_t *black = &self.feature_planes[<int>S_BLACK]
        cdef rollout_feature_t *white = &self.feature_planes[<int>S_WHITE]
        cdef int i, j

        black.color = <int>S_BLACK
        white.color = <int>S_WHITE

        for i in range(6):
            for j in range(PURE_BOARD_MAX):
                black.tensor[i][j] = -1
                white.tensor[i][j] = -1

        black.prev_d12_num = 0
        white.prev_d12_num = 0

        black.updated[0] = BOARD_MAX
        black.updated_num = 0
        white.updated[0] = BOARD_MAX
        white.updated_num = 0

    cdef bint memorize_updated(self, rollout_feature_t *feature, int pos) nogil:
        if feature.updated[pos] != 0:
            return False

        feature.updated[pos] = feature.updated[0]
        feature.updated[0] = pos

        feature.updated_num += 1

        return True
