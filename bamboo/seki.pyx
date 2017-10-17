# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

import numpy as np
cimport numpy as np

from libc.string cimport memset
from libc.stdio cimport printf

from bamboo.board cimport S_EMPTY, S_BLACK, S_WHITE, BOARD_MAX, PURE_BOARD_MAX, MAX_STRING, E_NOT_EYE
from bamboo.board cimport FLIP_COLOR, NORTH, WEST, EAST, SOUTH
from bamboo.board cimport onboard_pos, eye_condition, pure_board_max, board_size, liberty_end, string_end
from bamboo.board cimport game_state_t, string_t
from bamboo.board cimport get_neighbor4, is_true_eye, fill_n_int, fill_n_bint
from bamboo.pattern cimport pat3, print_input_pat3


cdef void check_seki(game_state_t *game, int *seki) nogil:
    cdef int i
    cdef string_t *string = game.string
    cdef bint seki_candidate[529]
    cdef int lib1, lib2
    cdef int string_pos

    fill_n_int(seki, BOARD_MAX, 0)
    fill_n_bint(seki_candidate, BOARD_MAX, 0)

    for i in range(pure_board_max):
        pos = onboard_pos[i]
        if (is_self_atari(game, pos, S_BLACK) and
            is_self_atari(game, pos, S_WHITE)):
            seki_candidate[pos] = True

    for i in range(MAX_STRING):
        if string[i].flag == False or string[i].libs != 2:
            continue

        lib1 = string[i].lib[0]
        lib2 = string[i].lib[lib1]

        if seki_candidate[lib1] and seki_candidate[lib2]: 
            seki[lib1] = seki[lib2] = 1 
        elif seki_candidate[lib1]:
            if eye_condition[pat3(game.pat, lib2)] != E_NOT_EYE:
                seki[lib1] = seki[lib2] = 1
        elif seki_candidate[lib2]:
            if eye_condition[pat3(game.pat, lib1)] != E_NOT_EYE:
                seki[lib1] = seki[lib2] = 1

        if seki[lib1]:
            string_pos = string[i].origin
            while string_pos != string_end:
                seki[string_pos] = 1
                string_pos = game.string_next[string_pos]


cdef bint is_self_atari(game_state_t *game, int pos, int color) nogil:
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
        return False

    # 上を調べる
    if board[north] == color:
        id = string_id[north]
        if string[id].libs > 2:
            return False
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
            return False
    elif board[north] == other and string[string_id[north]].libs == 1:
        return False

    # 左を調べる
    if board[west] == color:
        id = string_id[west]
        if already[0] != id:
            if string[id].libs > 2:
                return False
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
                return False
    elif board[west] == other and string[string_id[west]].libs == 1:
        return False

    # 右を調べる
    if board[east] == color:
        id = string_id[east];
        if already[0] != id and already[1] != id:
            if string[id].libs > 2:
                return False
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
                return False
    elif board[east] == other and string[string_id[east]].libs == 1:
        return False

    # 下を調べる
    if board[south] == color:
        id = string_id[south]
        if already[0] != id and already[1] != id and already[2] != id:
            if string[id].libs > 2:
                return False
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
                return False
    elif board[south] == other and string[string_id[south]].libs == 1:
        return False

    return True
