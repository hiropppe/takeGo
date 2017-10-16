# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

import numpy as np
cimport numpy as np

from libc.string cimport memset
from libc.stdio cimport printf

from bamboo.board cimport S_EMPTY, S_BLACK, S_WHITE, BOARD_MAX, PURE_BOARD_MAX, MAX_STRING, E_NOT_EYE, E_COMPLETE_HALF_EYE
from bamboo.board cimport FLIP_COLOR, NORTH, WEST, EAST, SOUTH
from bamboo.board cimport onboard_pos, eye_condition, pure_board_max, board_size, liberty_end
from bamboo.board cimport game_state_t, string_t
from bamboo.board cimport get_neighbor4, is_true_eye
from bamboo.pattern cimport pat3, print_input_pat3


cdef void check_seki(game_state_t *game, bint *seki):
    cdef int i, j, k, pos, id
    cdef char *board = game.board
    cdef int *string_id = game.string_id
    cdef string_t *string = game.string
    cdef bint seki_candidate[529]
    cdef int lib1, lib2
    cdef int lib1_id[4]
    cdef int lib2_id[4]
    cdef int lib1_ids, lib2_ids
    cdef int neighbor1_lib, neighbor2_lib
    cdef int neighbor4[4]
    cdef int tmp_id1 = 0, tmp_id2 = 0
    cdef bint already_checked
    cdef int empty_diagonal_stack[200]
    cdef int empty_diagonal_top = 0

    memset(seki_candidate, 0, sizeof(bint)*BOARD_MAX)

    # 双方が自己アタリになっている座標を抽出
    for i in range(pure_board_max):
        pos = onboard_pos[i]
        if (is_self_atari(game, pos, S_BLACK) and
            is_self_atari(game, pos, S_WHITE)):
            seki_candidate[pos] = True
            printf('candidate_pos=%d\n', pos)

    for i in range(MAX_STRING):
        # 連が存在しない,
        # または連の呼吸点数が2個でなければ次を調べる
        if string[i].flag == False or string[i].libs != 2:
            continue

        lib1 = string[i].lib[0]
        lib2 = string[i].lib[lib1]

        # 連の持つ呼吸点がともにセキの候補
        if seki_candidate[lib1] and seki_candidate[lib2]:
            # 呼吸点1の周囲の連のIDを取り出す
            get_neighbor4(neighbor4, lib1)
            lib1_ids = 0
            for j in range(4):
                if (board[neighbor4[j]] == S_BLACK or
                    board[neighbor4[j]] == S_WHITE):
                    id = string_id[neighbor4[j]]
                    if id != i:
                        already_checked = False
                        for k in range(lib1_ids):
                            if lib1_id[k] == id:
                                already_checked = True
                                break
                        if not already_checked:
                            lib1_id[lib1_ids] = id
                            lib1_ids += 1

            # 呼吸点2の周囲の連のIDを取り出す
            get_neighbor4(neighbor4, lib2)
            lib2_ids = 0
            for j in range(4):
                if (board[neighbor4[j]] == S_BLACK or
                    board[neighbor4[j]] == S_WHITE):
                    id = string_id[neighbor4[j]]
                    if id != i:
                        already_checked = False
                        for k in range(lib2_ids):
                            if lib2_id[k] == id:
                                already_checked = True
                                break

                        if not already_checked:
                            lib2_id[lib2_ids] = id
                            lib2_ids += 1

            printf('lib1=%d\n', lib1)
            printf('lib2=%d\n', lib2)
            printf('lib1_ids=%d, lib2_ids=%d\n', lib1_ids, lib2_ids)
            print_input_pat3(pat3(game.pat, lib1))
            print_input_pat3(pat3(game.pat, lib2))
            if lib1_ids == 1 and lib2_ids == 1:
                seki[lib1] = seki[lib2] = True
            elif eye_condition[pat3(game.pat, lib2)] != E_NOT_EYE:
                seki[lib1] = True
            elif eye_condition[pat3(game.pat, lib1)] != E_NOT_EYE:
                seki[lib2] = True
            else:
                pass

            """
            if lib1_ids == 1 and lib2_ids == 1:
                printf('lib1=%d lib2=%d\n', lib1, lib2)
                neighbor1_lib = string[lib1_id[0]].lib[0]
                if (neighbor1_lib == lib1 or
                    neighbor1_lib == lib2):
                    neighbor1_lib = string[lib1_id[0]].lib[neighbor1_lib]

                neighbor2_lib = string[lib2_id[0]].lib[0]
                if (neighbor2_lib == lib1 or
                    neighbor2_lib == lib2):
                    neighbor2_lib = string[lib2_id[0]].lib[neighbor2_lib]

                if neighbor1_lib == neighbor2_lib:
                    #print_input_pat3(pat3(game.pat, neighbor1_lib))
                    #print eye_condition[pat3(game.pat, neighbor1_lib)], E_NOT_EYE
                    #if eye_condition[pat3(game.pat, neighbor1_lib)] != E_NOT_EYE:
                    #    seki[lib1] = seki[lib2] = True
                    #    seki[neighbor1_lib] = True
                    seki[lib1] = seki[lib2] = True
                    seki[neighbor1_lib] = True
                elif (eye_condition[pat3(game.pat, neighbor1_lib)] == E_COMPLETE_HALF_EYE and
                      eye_condition[pat3(game.pat, neighbor2_lib)] == E_COMPLETE_HALF_EYE):
                    tmp_id1 = 0
                    tmp_id2 = 0
                    get_neighbor4(neighbor4, neighbor1_lib)
                    for j in range(4):
                        if (board[neighbor4[j]] == S_BLACK or
                            board[neighbor4[j]] == S_WHITE):
                            id = string_id[neighbor4[j]]
                            if (id != lib1_id[0] and
                                id != lib2_id[0] and
                                id != tmp_id1):
                                tmp_id1 = id
                    
                    get_neighbor4(neighbor4, neighbor2_lib)
                    for j in range(4):
                        if (board[neighbor4[j]] == S_BLACK or
                            board[neighbor4[j]] == S_WHITE):
                            id = string_id[neighbor4[j]]      
                            if (id != lib1_id[0] and
                                id != lib2_id[0] and
                                id != tmp_id2):
                                tmp_id2 = id;

                    if tmp_id1 == tmp_id2:
                        seki[lib1] = seki[lib2] = True
                        seki[neighbor1_lib] = seki[neighbor2_lib] = True
            """

cdef bint is_self_atari(game_state_t *game, int pos, int color):
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
