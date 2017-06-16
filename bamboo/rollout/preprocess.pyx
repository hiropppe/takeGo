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
from bamboo.go.board cimport get_neighbor4, get_neighbor8
from bamboo.go.pattern cimport N, S, W, E, NN, NW, NE, SS, SW, SE, WW, EE

from bamboo.rollout.pattern cimport x33_hash, x33_hashmap


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
        cdef int pos, color
        cdef int updated_string_num
        cdef int *updated_string_id
        cdef string_t *updated_string
        cdef int update_pos
        cdef int i

        if game.moves == 0:
            return

        pos = game.record[game.moves - 1].pos
        color = game.record[game.moves - 1].color

        # clear neighbor and response if passed ?
        if pos != PASS:
            self.update_neighbor(game, pos)
            self.update_d12(game, pos)

        updated_string_num = game.updated_string_num[color]
        updated_string_id = game.updated_string_id[color]
        for i in range(updated_string_num):
            updated_string = &game.string[updated_string_id[i]]
            self.update_save_atari(game, updated_string)
            update_pos = updated_string.empty[0]
            while update_pos != STRING_EMPTY_END:
                self.update_3x3(game, update_pos)
                update_pos = updated_string.empty[update_pos]

        # clear updated string memo for next feature calculation
        self.clear_updated_string_cache(game)

    cdef void update_3x3(self, game_state_t *game, int pos) nogil:
        """ Move matches 3 × 3 pattern around move
        """
        cdef rollout_feature_t *feature = &self.feature_planes[<int>game.current_color]
        cdef unsigned long long hash
        cdef int pat_ix

        hash = x33_hash(game, pos, <int>game.current_color)
        if x33_hashmap.find(hash) != x33_hashmap.end():
            pat_ix = self.x33_start + x33_hashmap[hash]
            feature.tensor[onboard_index[pos]][NON_RESPONSE_PAT] = pat_ix

    cdef void update_d12(self, game_state_t *game, int pos) nogil:
        """ Move matches 12-point diamond pattern near previous move
        """
        cdef rollout_feature_t *feature = &self.feature_planes[<int>game.current_color]
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
                feature.tensor[onboard_index[d12[i]]][RESPONSE_PAT] = pat_id

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
                feature.tensor[onboard_index[string.lib[0]]][SAVE_ATARI] = self.save_atari_start 

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
                    feature.tensor[feature.prev_neighbor8[i]][NEIGHBOR] = self.neighbor_start + i

            neighbor_pos = neighbor8[i]
            if game.board[neighbor_pos] != S_OB:
                neighbor_i = onboard_index[neighbor_pos]
                feature.tensor[neighbor_i][NEIGHBOR] = self.neighbor_start + i
                feature.prev_neighbor8[i] = neighbor_i
            else:
                feature.prev_neighbor8[i] = -1

        feature.is_neighbor8_set = True

    cdef void update_lil(self, game_state_t *game, object lil_matrix):
        cdef int current_color = <int>game.current_color
        cdef int pos, color
        cdef int updated_string_num
        cdef int *updated_string_id
        cdef string_t *updated_string
        cdef int update_pos
        cdef int i

        if game.moves == 0:
            return

        pos = game.record[game.moves - 1].pos
        color = game.record[game.moves - 1].color

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
            update_pos = updated_string.lib[0]
            while update_pos != liberty_end:
                self.update_3x3_lil(game, update_pos, lil_matrix)
                update_pos = updated_string.lib[update_pos]

        # clear updated string memo for next feature calculation
        self.clear_updated_string_cache(game)

    cdef void update_3x3_lil(self, game_state_t *game, int pos, object lil_matrix):
        """ Move matches 3 × 3 pattern around move
        """
        cdef unsigned long long hash
        cdef int pat_id

        hash = x33_hash(game, pos, <int>game.current_color)
        if x33_hashmap.find(hash) != x33_hashmap.end():
            pat_id = self.x33_start + x33_hashmap[hash]
            lil_matrix[onboard_index[pos], pat_id] = 1

    cdef void update_d12_lil(self, game_state_t *game, int pos, object lil_matrix):
        """ Move matches 12-point diamond pattern near previous move
        """
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
                lil_matrix[onboard_index[d12[i]], pat_id] = 1

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
                lil_matrix[onboard_index[string.lib[0]], self.save_atari_start] = 1

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
                    lil_matrix[feature.prev_neighbor8[i], self.neighbor_start + i] = 0

            neighbor_pos = neighbor8[i]
            if game.board[neighbor_pos] != S_OB:
                neighbor_i = onboard_index[neighbor_pos]
                lil_matrix[neighbor_i, self.neighbor_start + i] = 1
                feature.prev_neighbor8[i] = neighbor_i
            else:
                feature.prev_neighbor8[i] = -1

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

        for i in range(PURE_BOARD_MAX):
            for j in range(6):
                black_feature.tensor[i][j] = -1.0
                white_feature.tensor[i][j] = -1.0

        black_feature.is_neighbor8_set = False
        white_feature.is_neighbor8_set = False

    cdef object get_tensor_as_csr_matrix(self, int color):
        # too slow to use
        cdef int i, j
        cdef rollout_feature_t *feature = &self.feature_planes[color]
        sparse_tensor = lil_matrix((pure_board_max, self.feature_size))
        for i in range(pure_board_max):
            for j in range(6):
                if feature.tensor[i][j] != -1.0:
                    sparse_tensor[i, feature.tensor[i][j]] = 1
        return sparse_tensor.tocsr()
