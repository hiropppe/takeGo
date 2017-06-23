# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

from libc.stdio cimport printf

import numpy as np

cimport numpy as np

from scipy.sparse import lil_matrix, csr_matrix

from bamboo.go.board cimport PURE_BOARD_MAX, S_EMPTY, S_BLACK, S_WHITE, S_OB, PASS, STRING_EMPTY_END 
from bamboo.go.board cimport FLIP_COLOR
from bamboo.go.board cimport game_state_t, pure_board_max, onboard_index, onboard_pos, liberty_end
from bamboo.go.board cimport get_neighbor4, get_neighbor8, get_md12
from bamboo.go.pattern cimport N, S, W, E, NN, NW, NE, SS, SW, SE, WW, EE

from bamboo.rollout.pattern cimport x33_hash, x33_hashmap
from bamboo.rollout.pattern cimport d12_hash, d12_hashmap, d12_pos_mt


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

    cdef void rebase(self, game_state_t *game) nogil:
        pass

    cdef void update(self, game_state_t *game) nogil:
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

        prev_pos = game.record[game.moves - 1].pos
        prev_color = game.record[game.moves - 1].color
        self.clear_onehot_index(game, prev_pos)

        if game.moves > 1:
            prev2_pos = game.record[game.moves - 2].pos
            self.clear_onehot_index(game, prev2_pos)

        # clear neighbor and response if passed ?
        if prev_pos != PASS:
            self.update_neighbor(game, prev_pos)
            self.update_d12(game, prev_pos, prev_color)

        updated_string_num = game.updated_string_num[current_color]
        updated_string_id = game.updated_string_id[current_color]
        for i in range(updated_string_num):
            updated_string = &game.string[updated_string_id[i]]
            self.update_save_atari(game, updated_string)
            update_pos = updated_string.empty[0]
            while update_pos != STRING_EMPTY_END:
                self.update_3x3(game, update_pos)
                update_pos = updated_string.empty[update_pos]

        # clear updated string memo for next feature calculation
        self.clear_updated_string_cache(game)

    cdef void clear_onehot_index(self, game_state_t *game, int pos) nogil:
        cdef rollout_feature_t *feature = &self.feature_planes[<int>game.current_color]
        if pos != PASS:
            feature.tensor[RESPONSE][onboard_index[pos]] = -1 
            feature.tensor[SAVE_ATARI][onboard_index[pos]] = -1 
            feature.tensor[NAKADE][onboard_index[pos]] = -1 
            feature.tensor[RESPONSE_PAT][onboard_index[pos]] = -1 
            feature.tensor[NON_RESPONSE_PAT][onboard_index[pos]] = -1 

    cdef void update_3x3(self, game_state_t *game, int pos) nogil:
        """ Move matches 3 × 3 pattern around move
        """
        cdef rollout_feature_t *feature = &self.feature_planes[<int>game.current_color]
        cdef unsigned long long hash
        cdef int pat_ix

        hash = x33_hash(game, pos, <int>game.current_color)
        if x33_hashmap.find(hash) != x33_hashmap.end():
            pat_ix = self.x33_start + x33_hashmap[hash]
            feature.tensor[NON_RESPONSE_PAT][onboard_index[pos]] = pat_ix

    cdef void update_d12(self, game_state_t *game, int prev_pos, int prev_color) nogil:
        """ Move matches 12-point diamond pattern near previous move
        """
        cdef rollout_feature_t *feature = &self.feature_planes[<int>game.current_color]
        cdef int i
        cdef int empty_ix[12], empty_pos[12]
        cdef int n_empty_val = 0
        cdef int *n_empty = &n_empty_val
        cdef unsigned long long hash, positional_hash
        cdef int pax_ix
        cdef int empty_onboard_ix
        cdef int prev_d12_num = 0

        # clear previous d12 positions
        for i in range(feature.prev_d12_num):
            feature.tensor[RESPONSE_PAT][feature.prev_d12[i]] = -1

        hash = d12_hash(game, prev_pos, prev_color, empty_ix, empty_pos, n_empty)
        for i in range(n_empty_val):
            positional_hash = hash ^ d12_pos_mt[1 << empty_ix[i]] 
            if d12_hashmap.find(positional_hash) != d12_hashmap.end():
                pat_ix = self.d12_start + d12_hashmap[positional_hash]
                empty_onboard_ix = onboard_index[empty_pos[i]]
                feature.tensor[RESPONSE_PAT][empty_onboard_ix] = pat_ix
                # memorize previous d12 position
                feature.prev_d12[prev_d12_num] = empty_onboard_ix
                feature.prev_d12_num += 1

    cdef void update_save_atari(self, game_state_t *game, string_t *string) nogil:
        """ Save atari 1 Move saves stone(s) from capture
        """
        cdef rollout_feature_t *feature = &self.feature_planes[<int>game.current_color]
        cdef int neighbor4[4]
        cdef int neighbor_pos, neighbor_string_id
        cdef string_t *neighbor_string
        cdef int libs_after_move
        cdef int i
        cdef bint flag = False

        libs_after_move = 0
        if string.libs == 1 and string.color == game.current_color:
            get_neighbor4(neighbor4, string.lib[0])
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
                feature.tensor[SAVE_ATARI][onboard_index[string.lib[0]]] = self.save_atari_start 

    cdef void update_neighbor(self, game_state_t *game, int pos) nogil:
        """ Move is 8-connected to previous move
        """
        cdef rollout_feature_t *feature = &self.feature_planes[<int>game.current_color]
        cdef int neighbor8[8]
        cdef int neighbor_pos, neighbor_i
        cdef int i

        get_neighbor8(neighbor8, pos)

        for i in range(8):
            if feature.is_neighbor8_set:
                # unset previous neighbor8
                if feature.prev_neighbor8[i] != -1:
                    feature.tensor[NEIGHBOR][feature.prev_neighbor8[i]] = -1

            neighbor_pos = neighbor8[i]
            if game.board[neighbor_pos] != S_OB:
                neighbor_i = onboard_index[neighbor_pos]
                feature.tensor[NEIGHBOR][neighbor_i] = self.neighbor_start + i
                feature.prev_neighbor8[i] = neighbor_i
            else:
                feature.prev_neighbor8[i] = -1

        feature.is_neighbor8_set = True

    cdef void update_lil(self, game_state_t *game, object lil_matrix):
        """ Direct update sparse feature.
            !! Too slow
        """
        cdef int current_color = <int>game.current_color
        cdef int prev_pos, prev_color
        cdef int prev2_pos, prev2_color
        cdef int updated_string_num
        cdef int *updated_string_id
        cdef string_t *updated_string
        cdef int update_pos
        cdef int i

        if game.moves == 0:
            return

        pos = game.record[game.moves - 1].pos
        self.clear_onehot_lil_index(game, pos, lil_matrix)

        if game.moves > 1:
            prev2_pos = game.record[game.moves - 2].pos
            self.clear_onehot_lil_index(game, prev2_pos, lil_matrix)

        # clear neighbor and response if passed ?
        """
        if pos != PASS:
            self.update_neighbor_lil(game, pos, lil_matrix)
            self.update_d12_lil(game, pos, lil_matrix)
        """
        updated_string_num = game.updated_string_num[current_color]
        updated_string_id = game.updated_string_id[current_color]
        for i in range(updated_string_num):
            updated_string = &game.string[updated_string_id[i]]
            """
            self.update_save_atari_lil(game, updated_string, lil_matrix)
            """
            update_pos = updated_string.empty[0]
            while update_pos != STRING_EMPTY_END:
                self.update_3x3_lil(game, update_pos, lil_matrix)
                update_pos = updated_string.empty[update_pos]

        # clear updated string memo for next feature calculation
        self.clear_updated_string_cache(game)

    cdef void clear_onehot_lil_index(self, game_state_t *game, int pos, object lil_matrix):
        if pos != PASS:
            lil_matrix[:, onboard_index[pos]] = 0

    cdef void update_3x3_lil(self, game_state_t *game, int pos, object lil_matrix):
        """ Move matches 3 × 3 pattern around move
        """
        cdef unsigned long long hash
        cdef int pat_ix

        hash = x33_hash(game, pos, <int>game.current_color)
        if x33_hashmap.find(hash) != x33_hashmap.end():
            pat_ix = self.x33_start + x33_hashmap[hash]
            # clear old then set new index
            lil_matrix[self.x33_start:self.d12_start, onboard_index[pos]] = 0
            lil_matrix[pat_ix, onboard_index[pos]] = 1

    cdef void update_d12_lil(self, game_state_t *game, int pos, object lil_matrix):
        """ Move matches 12-point diamond pattern near previous move
        cdef int i
        cdef int d12[12]
        cdef int pat_id

        d12[0] = pos + NN;
        d12[1] = pos + NW; d12[2] = pos + N; d12[3] = pos + NE;
        d12[4] = pos + WW; d12[5] = pos + W; d12[6] = pos + E; d12[7] = pos + EE;
        d12[8] = pos + SW; d12[9] = pos + S; d12[10] = pos + SE;
        d12[11] = pos + SS;

        # lookup pattern index
        pat_id = self.d12_start

        for i in range(12):
            if d12[i] == S_EMPTY:
                lil_matrix[pat_id, onboard_index[d12[i]]] = 1
        """
        pass

    cdef void update_save_atari_lil(self, game_state_t *game, string_t *string, object lil_matrix):
        """ Save atari 1 Move saves stone(s) from capture
        """
        cdef int neighbor4[4]
        cdef int neighbor_pos, neighbor_string_id
        cdef string_t *neighbor_string
        cdef int libs_after_move
        cdef int i
        cdef bint flag = False

        libs_after_move = 0
        if string.libs == 1 and string.color == game.current_color:
            get_neighbor4(neighbor4, string.lib[0])
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
                lil_matrix[self.save_atari_start, onboard_index[string.lib[0]]] = 1

    cdef void update_neighbor_lil(self, game_state_t *game, int pos, object lil_matrix):
        """ Move is 8-connected to previous move
        """
        cdef rollout_feature_t *feature = &self.feature_planes[<int>game.current_color]
        cdef int neighbor8[8]
        cdef int neighbor_pos, neighbor_i
        cdef int i

        get_neighbor8(neighbor8, pos)

        for i in range(8):
            if feature.is_neighbor8_set:
                # unset previous neighbor8
                if feature.prev_neighbor8[i] != -1:
                    lil_matrix[self.neighbor_start + i, feature.prev_neighbor8[i]] = 0

            neighbor_pos = neighbor8[i]
            if game.board[neighbor_pos] != S_OB:
                neighbor_i = onboard_index[neighbor_pos]
                lil_matrix[self.neighbor_start + i, neighbor_i] = 1
                feature.prev_neighbor8[i] = neighbor_i
            else:
                feature.prev_neighbor8[i] = 0

        feature.is_neighbor8_set = True

    cdef void clear_updated_string_cache(self, game_state_t *game) nogil:
        cdef int *updated_string_num

        updated_string_num = &game.updated_string_num[<int>game.current_color]
        updated_string_num[0] = 0

    cdef void clear_planes(self) nogil:
        cdef rollout_feature_t *black_feature = &self.feature_planes[<int>S_BLACK]
        cdef rollout_feature_t *white_feature = &self.feature_planes[<int>S_WHITE]
        cdef int i, j

        black_feature.color = <int>S_BLACK
        white_feature.color = <int>S_WHITE

        for i in range(6):
            for j in range(PURE_BOARD_MAX):
                black_feature.tensor[i][j] = -1
                white_feature.tensor[i][j] = -1

        black_feature.is_neighbor8_set = False
        white_feature.is_neighbor8_set = False

        black_feature.prev_d12_num = 0
        white_feature.prev_d12_num = 0

    cdef object get_tensor_as_csr_matrix(self, int color):
        # too slow to use
        cdef int i, j
        cdef rollout_feature_t *feature = &self.feature_planes[color]
        sparse_tensor = lil_matrix((self.feature_size, pure_board_max))
        for i in range(6):
            for j in range(pure_board_max):
                if feature.tensor[i][j] != -1:
                    sparse_tensor[i, feature.tensor[i][j]] = 1
        return sparse_tensor.tocsr()
