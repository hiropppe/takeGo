# -*- coding: utf-8 -*-
#cython: boundscheck=False
#cython: wraparound=False

import numpy as np

cimport numpy as np

from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy
from libc.math cimport exp as cexp
from libc.stdio cimport printf

from libcpp.string cimport string as cppstring

cimport point
cimport pattern as pat
cimport printer 
cimport policy_feature

from bamboo.go.zobrist_hash cimport HASH_PASS, HASH_BLACK, HASH_WHITE, HASH_KO
from bamboo.go.zobrist_hash cimport hash_bit


pure_board_size = PURE_BOARD_SIZE
pure_board_max = PURE_BOARD_MAX

board_size = BOARD_SIZE
board_max = BOARD_MAX

max_string = MAX_STRING
max_neighbor = MAX_STRING

board_start = BOARD_START
board_end = BOARD_END

string_lib_max = STRING_LIB_MAX
string_pos_max = STRING_POS_MAX

string_end = STRING_END
liberty_end = LIBERTY_END

max_records = MAX_RECORDS
max_moves = MAX_MOVES

default_komi = KOMI


cdef void fill_n_char (char *arr, int size, char v) nogil:
    cdef int i
    for i in range(size):
        arr[i] = v


cdef void fill_n_unsigned_char (unsigned char *arr, int size, unsigned char v) nogil:
    cdef int i
    for i in range(size):
        arr[i] = v


cdef void fill_n_short (short *arr, int size, short v) nogil:
    cdef int i
    for i in range(size):
        arr[i] = v


cdef void fill_n_int (int *arr, int size, int v) nogil:
    cdef int i
    for i in range(size):
        arr[i] = v


cdef game_state_t *allocate_game() nogil:
    cdef game_state_t *game

    game = <game_state_t *>malloc(sizeof(game_state_t))
    memset(game, 0, sizeof(game_state_t))

    return game


cdef void free_game(game_state_t *game) nogil:
    if game:
        free(game)


cdef void copy_game(game_state_t *dst, game_state_t *src) nogil:
    #memcpy(dst.record, src.record, sizeof(move_t) * max_records)
    #memcpy(dst.prisoner, src.prisoner, sizeof(int) * S_MAX)
    memcpy(dst.board, src.board, sizeof(char) * board_max)
    #memcpy(dst.birth_move, src.birth_move, sizeof(int) * board_max)
    memcpy(dst.pat, src.pat, sizeof(pattern_t) * board_max)
    memcpy(dst.string_id, src.string_id, sizeof(int) * string_pos_max)
    memcpy(dst.string_next, src.string_next, sizeof(int) * string_pos_max)
    #memcpy(dst.candidates, src.candidates, sizeof(bint) * board_max)
    #memcpy(dst.capture_num, src.capture_num, sizeof(int) * S_OB)

    for i in range(max_string):
        if src.string[i].flag:
            memcpy(&dst.string[i], &src.string[i], sizeof(string_t))
        else:
            dst.string[i].flag = False

    dst.current_color = src.current_color
    #dst.pass_count = src.pass_count
    dst.moves = src.moves
    dst.ko_move = src.ko_move
    dst.ko_pos = src.ko_pos


cdef void initialize_board(game_state_t *game, bint rollout):
    cdef int i, x, y, pos

    memset(game.record, 0, sizeof(move_t) * max_records)
    memset(game.pat, 0, sizeof(pattern_t) * board_max)

    game.current_color = S_BLACK
    game.moves = 0
    game.ko_pos = 0
    game.ko_move = 0
    game.pass_count = 0
    game.current_hash = 0
    game.rollout = rollout

    fill_n_char(game.board, BOARD_MAX, 0)
    fill_n_int(game.birth_move, BOARD_MAX, 0)
    fill_n_int(game.capture_num, S_OB, 0)
    fill_n_int(game.candidates, BOARD_MAX, 0)

    for y in range(board_size):
        for x in range(OB_SIZE):
            game.board[POS(x, y, board_size)] = S_OB
            game.board[POS(y, x, board_size)] = S_OB
            game.board[POS(y, board_size - 1 - x, board_size)] = S_OB
            game.board[POS(board_size - 1 - x, y, board_size)] = S_OB

    for y in range(board_start, board_end + 1):
        for x in range(board_start, board_end + 1):
            pos = POS(x, y, board_size)
            game.candidates[pos] = 1 

    for i in range(max_string):
        game.string[i].flag = False

    pat.clear_pattern(game.pat)

    initialize_neighbor()
    initialize_eye()


cdef bint do_move(game_state_t *game, int pos) nogil:
    cdef bint is_legal
    is_legal = put_stone(game, pos, game.current_color)
    if is_legal:
        game.current_color = FLIP_COLOR(game.current_color)
    return is_legal


cdef bint put_stone(game_state_t *game, int pos, char color) nogil:
    cdef char other = FLIP_COLOR(color)
    cdef int connection = 0
    cdef int connect[4]
    cdef int prisoner = 0
    cdef int neighbor8[8]
    cdef int neighbor_pos, neighbor_string_id
    cdef string_t *neighbor_string
    cdef int i

    if not is_legal(game, pos, color):
        return False

    if game.moves < max_records:
        game.record[game.moves].color = color
        game.record[game.moves].pos = pos

    if pos == PASS:
        game.current_hash ^= hash_bit[game.pass_count][<int>HASH_PASS]
        game.pass_count += 1
        game.moves += 1
        return True

    fill_n_int(connect, 4, 0)

    game.capture_num[<int>color] = 0

    game.board[pos] = color

    game.current_hash ^= hash_bit[pos][<int>color]

    game.candidates[pos] = 0

    pat.update_md2_stone(game.pat, pos, color)

    get_neighbor8(neighbor8, pos)

    for i in range(8):
        neighbor_pos = neighbor8[i]
        neighbor_string_id = game.string_id[neighbor_pos]
        neighbor_string = &game.string[neighbor_string_id]
        if i < 4:
            if game.board[neighbor_pos] == color:
                remove_liberty(neighbor_string, pos)
                connect[connection] = neighbor_string_id
                connection += 1
                memorize_updated_string(game, neighbor_string_id)
            elif game.board[neighbor_pos] == other:
                remove_liberty(neighbor_string, pos)
                if game.string[game.string_id[neighbor_pos]].libs == 0:
                    prisoner += remove_string(game, neighbor_string)
                else:
                    memorize_updated_string(game, neighbor_string_id)

        if neighbor_string.flag:
            remove_empty(neighbor_string, pos)

    game.prisoner[<int>color] += prisoner

    if connection == 0:
        make_string(game, pos, color)
        if prisoner == 1 and game.string[game.string_id[pos]].libs == 1:
            game.ko_move = game.moves
            game.ko_pos = game.string[game.string_id[pos]].lib[0]
            game.current_hash ^= hash_bit[game.ko_pos][<int>HASH_KO]
    elif connection == 1:
        add_stone(game, pos, color, connect[0])
    else:
        connect_string(game, pos, color, connection, connect)

    game.moves += 1

    if not game.rollout:
        game.birth_move[pos] = game.moves

    return True


cdef void connect_string(game_state_t *game, int pos, char color, int connection, int string_id[4]) nogil:
    cdef int min_string_id = string_id[0]
    cdef string_t *string[3]
    cdef int connections = 0
    cdef bint flag = True
    cdef int i, j

    for i in range(1, connection):
        flag = True
        for j in range(i):
            if string_id[j] == string_id[i]:
                flag = False
                break
        if flag:
            if min_string_id > string_id[i]:
                string[connections] = &game.string[min_string_id]
                min_string_id = string_id[i]
            else:
                string[connections] = &game.string[string_id[i]]
            connections += 1

    add_stone(game, pos, color, min_string_id)

    if connections > 0:
        merge_string(game, &game.string[min_string_id], string, connections)


cdef void merge_string(game_state_t *game, string_t *dst, string_t *src[3], int n) nogil:
    cdef int tmp, pos, prev, neighbor
    cdef int string_id = game.string_id[dst.origin]
    cdef int removed_string_id
    cdef int i

    for i in range(n):
        removed_string_id = game.string_id[src[i].origin]

        prev = 0
        pos = src[i].lib[0]
        while pos != liberty_end:
            prev = add_liberty(dst, pos, prev)
            pos = src[i].lib[pos]

        prev = 0
        pos = src[i].origin
        while pos != string_end:
            game.string_id[pos] = string_id
            tmp = game.string_next[pos]
            add_stone_to_string(game, dst, pos, prev)
            prev = pos
            pos = tmp

        prev = 0
        neighbor = src[i].neighbor[0]
        while neighbor != NEIGHBOR_END:
            remove_neighbor_string(&game.string[neighbor], removed_string_id)
            add_neighbor(dst, neighbor, prev)
            add_neighbor(&game.string[neighbor], string_id, prev)
            prev = neighbor
            neighbor = src[i].neighbor[neighbor]

        prev = 0
        pos = src[i].empty[0]
        while pos != STRING_EMPTY_END:
            prev = add_empty(dst, pos, prev)
            pos = src[i].empty[pos]

        src[i].flag = False


cdef void add_stone(game_state_t *game, int pos, char color, int string_id) nogil:
    cdef string_t *add_string
    cdef int lib_add = 0
    cdef int empty_add = 0
    cdef int other = FLIP_COLOR(color)
    cdef int neighbor8[8]
    cdef int neighbor_pos, neighbor_string_id
    cdef int i

    game.string_id[pos] = string_id

    add_string = &game.string[string_id]

    add_stone_to_string(game, add_string, pos, 0)

    get_neighbor8(neighbor8, pos)

    for i in range(8):
        neighbor_pos = neighbor8[i]
        if i < 4:
            if game.board[neighbor_pos] == S_EMPTY:
                lib_add = add_liberty(add_string, neighbor_pos, lib_add)
            elif game.board[neighbor_pos] == other:
                neighbor_string_id = game.string_id[neighbor_pos]
                add_neighbor(&game.string[neighbor_string_id], string_id, 0)
                add_neighbor(&game.string[string_id], neighbor_string_id, 0)

        if game.board[neighbor_pos] == S_EMPTY:
            empty_add = add_empty(add_string, neighbor_pos, empty_add)

    memorize_updated_string(game, string_id)


cdef void add_stone_to_string(game_state_t *game, string_t *string, int pos, int head) nogil:
    cdef int string_pos

    if pos == string_end:
        return

    if string.origin > pos:
        game.string_next[pos] = string.origin
        string.origin = pos
    else:
        if head != 0:
            string_pos = head
        else:
            string_pos = string.origin

        while game.string_next[string_pos] < pos:
            string_pos = game.string_next[string_pos]

        game.string_next[pos] = game.string_next[string_pos]
        game.string_next[string_pos] = pos

    string.size += 1


cdef void make_string(game_state_t *game, int pos, char color) nogil:
    cdef string_t *new_string
    cdef int string_id = 1
    cdef int lib_add = 0
    cdef int empty_add = 0
    cdef int other = FLIP_COLOR(color)
    cdef int neighbor8[8]
    cdef int neighbor_pos
    cdef int neighbor_string_id
    cdef int i

    while game.string[string_id].flag:
        string_id += 1

    new_string = &game.string[string_id]

    fill_n_short(new_string.lib, string_lib_max, 0)
    fill_n_short(new_string.neighbor, max_neighbor, 0)
    fill_n_short(new_string.empty, STRING_EMPTY_MAX, 0)

    new_string.libs = 0
    new_string.lib[0] = liberty_end
    new_string.neighbors = 0
    new_string.neighbor[0] = NEIGHBOR_END
    new_string.empties = 0
    new_string.empty[0] = STRING_EMPTY_END
    new_string.color = color
    new_string.origin = pos
    new_string.size = 1

    game.string_id[pos] = string_id
    game.string_next[pos] = string_end

    get_neighbor8(neighbor8, pos)

    for i in range(8):
        neighbor_pos = neighbor8[i]
        if i < 4:
            if game.board[neighbor_pos] == S_EMPTY:
                lib_add = add_liberty(new_string, neighbor_pos, lib_add)
            elif game.board[neighbor_pos] == other:
                neighbor_string_id = game.string_id[neighbor_pos]
                add_neighbor(&game.string[neighbor_string_id], string_id, 0)
                add_neighbor(&game.string[string_id], neighbor_string_id, 0)

        if game.board[neighbor_pos] == S_EMPTY:
            empty_add = add_empty(new_string, neighbor_pos, empty_add)

    new_string.flag = True

    memorize_updated_string(game, string_id)


cdef int remove_string(game_state_t *game, string_t *string) nogil:
    cdef int next
    cdef int neighbor
    cdef int pos = string.origin
    cdef int color = <int>game.current_color
    cdef int remove_string_id = game.string_id[pos]
    cdef int remove_color = game.board[pos]
    cdef int *capture_num = &game.capture_num[color]
    cdef int *capture_pos = game.capture_pos[color]
    cdef int lib

    neighbor = string.neighbor[0]
    while neighbor != NEIGHBOR_END:
        memorize_updated_string(game, neighbor)
        neighbor = string.neighbor[neighbor]

    while True:
        game.board[pos] = S_EMPTY

        if not game.rollout:
            game.birth_move[pos] = 0

        capture_num[0] += 1
        capture_pos[capture_num[0]] = pos 

        pat.update_md2_empty(game.pat, pos)

        north_string_id = game.string_id[NORTH(pos, board_size)]
        west_string_id = game.string_id[WEST(pos)]
        east_string_id = game.string_id[EAST(pos)]
        south_string_id = game.string_id[SOUTH(pos, board_size)]

        if game.string[north_string_id].flag:
            add_liberty(&game.string[north_string_id], pos, 0)
        if game.string[west_string_id].flag:
            add_liberty(&game.string[west_string_id], pos, 0)
        if game.string[east_string_id].flag:
            add_liberty(&game.string[east_string_id], pos, 0)
        if game.string[south_string_id].flag:
            add_liberty(&game.string[south_string_id], pos, 0)

        next = game.string_next[pos]

        game.string_next[pos] = 0
        game.string_id[pos] = 0

        pos = next

        if pos == string_end:
            break

    neighbor = string.neighbor[0]
    while neighbor != NEIGHBOR_END:
        remove_neighbor_string(&game.string[neighbor], remove_string_id)
        neighbor = string.neighbor[neighbor]

    string.flag = False

    return string.size


cdef int add_liberty(string_t *string, int pos, int head) nogil:
    cdef int lib

    if string.lib[pos] != 0:
        return pos

    lib = head

    while string.lib[lib] < pos:
        lib = string.lib[lib]

    string.lib[pos] = string.lib[lib]
    string.lib[lib] = <short>pos

    string.libs += 1

    return pos


cdef int add_empty(string_t *string, int pos, int head) nogil:
    cdef int empty

    if string.empty[pos] != 0:
        return pos

    empty = head

    while string.empty[empty] < pos:
        empty = string.empty[empty]

    string.empty[pos] = string.empty[empty]
    string.empty[empty] = <short>pos

    string.empties += 1

    return pos


cdef void remove_liberty(string_t *string, int pos) nogil:
    cdef int lib = 0

    if string.lib[pos] == 0:
        return

    while string.lib[lib] != pos:
        lib = string.lib[lib]

    string.lib[lib] = string.lib[string.lib[lib]]
    string.lib[pos] = <short>0

    string.libs -= 1


cdef void remove_empty(string_t *string, int pos) nogil:
    cdef int empty = 0

    if string.empty[pos] == 0:
        return

    while string.empty[empty] != pos:
        empty = string.empty[empty]

    string.empty[empty] = string.empty[string.empty[empty]]
    string.empty[pos] = <short>0

    string.empties -= 1


cdef void add_neighbor(string_t *string, int string_id, int head) nogil:
    cdef int neighbor = 0

    if string.neighbor[string_id] != 0:
        return

    neighbor = head

    while string.neighbor[neighbor] < string_id:
        neighbor = string.neighbor[neighbor]

    string.neighbor[string_id] = string.neighbor[neighbor]
    string.neighbor[neighbor] = <short>string_id

    string.neighbors += 1


cdef void remove_neighbor_string(string_t *string, int string_id) nogil:
    cdef int neighbor = 0

    if string.neighbor[string_id] == 0:
        return

    while string.neighbor[neighbor] != string_id:
        neighbor = string.neighbor[neighbor]

    string.neighbor[neighbor] = string.neighbor[string.neighbor[neighbor]]
    string.neighbor[string_id] = 0

    string.neighbors -= 1


cdef void get_neighbor4(int neighbor4[4], int pos) nogil:
    neighbor4[0] = NORTH(pos, board_size)
    neighbor4[1] = WEST(pos)
    neighbor4[2] = EAST(pos)
    neighbor4[3] = SOUTH(pos, board_size)


cdef void get_neighbor8(int neighbor8[8], int pos) nogil:
    neighbor8[0] = NORTH(pos, board_size)
    neighbor8[1] = WEST(pos)
    neighbor8[2] = EAST(pos)
    neighbor8[3] = SOUTH(pos, board_size)
    neighbor8[4] = NORTH_WEST(pos, board_size)
    neighbor8[5] = NORTH_EAST(pos, board_size)
    neighbor8[6] = SOUTH_WEST(pos, board_size)
    neighbor8[7] = SOUTH_EAST(pos, board_size)


cdef void get_neighbor8_in_order(int neighbor8[8], int pos) nogil:
    neighbor8[0] = NORTH_WEST(pos, board_size)
    neighbor8[1] = NORTH(pos, board_size)
    neighbor8[2] = NORTH_EAST(pos, board_size)
    neighbor8[3] = WEST(pos)
    neighbor8[4] = EAST(pos)
    neighbor8[5] = SOUTH_WEST(pos, board_size)
    neighbor8[6] = SOUTH(pos, board_size)
    neighbor8[7] = SOUTH_EAST(pos, board_size)


cdef void get_md12(int md12[12], int pos) nogil:
    md12[0] = pos + pat.NN
    md12[1] = pos + pat.NW
    md12[2] = pos + pat.N
    md12[3] = pos + pat.NE
    md12[4] = pos + pat.WW
    md12[5] = pos + pat.W
    md12[6] = pos + pat.E
    md12[7] = pos + pat.EE
    md12[8] = pos + pat.SW
    md12[9] = pos + pat.S
    md12[10] = pos + pat.SE
    md12[11] = pos + pat.SS


cdef void init_board_position():
    cdef int i, x, y, p,
    cdef int neighbor4[4]
    cdef int n, nx, ny, n_pos, n_size

    global onboard_index, onboard_pos, board_x, board_y

    free(onboard_index)
    free(onboard_pos)
    free(board_x)
    free(board_y)

    onboard_index = <int *>malloc(board_max * sizeof(int))
    onboard_pos = <int *>malloc(pure_board_max * sizeof(int))
    board_x = <int *>malloc(board_max * sizeof(int))
    board_y = <int *>malloc(board_max * sizeof(int))

    i = 0
    for y in range(board_start, board_end + 1):
        for x in range(board_start, board_end + 1):
            p = POS(x, y, board_size)
            onboard_index[p] = i
            onboard_pos[i] = p
            board_x[p] = x
            board_y[p] = y
            i += 1


cdef void init_line_number():
    cdef int x, y, p, d

    global board_dis_x, board_dis_y

    free(board_dis_x)
    free(board_dis_y)

    board_dis_x = <int *>malloc(board_max * sizeof(int))
    board_dis_y = <int *>malloc(board_max * sizeof(int))
    for y in range(board_start, board_end + 1):
        for x in range(board_start, board_start + pure_board_size / 2 + 1):
            d = x - (OB_SIZE - 1)
            board_dis_x[POS(x, y, board_size)] = d
            board_dis_x[POS(board_end + OB_SIZE - x, y, board_size)] = d
            board_dis_y[POS(y, x, board_size)] = d
            board_dis_y[POS(y, board_end + OB_SIZE - x, board_size)] = d 


cdef void init_move_distance():
    cdef int x, y

    global move_dis

    move_dis = np.zeros((pure_board_size, pure_board_size), dtype=np.int32)
    for y in range(pure_board_size):
        for x in range(pure_board_size):
            move_dis[x, y] = x + y + MAX(x, y)
            if move_dis[x, y] > MOVE_DISTANCE_MAX:
                move_dis[x, y] = MOVE_DISTANCE_MAX


cdef void init_board_position_id():
    cdef int i, x, y

    global board_pos_id

    free(board_pos_id)

    board_pos_id = <int *>malloc(board_max * sizeof(int))
    fill_n_int(board_pos_id, board_max, 0)
    i = 1
    for y in range(board_start, board_start + pure_board_size / 2 + 1):
        for x in range(board_start, y + 1):
            board_pos_id[POS(x, y, board_size)] = i
            board_pos_id[POS(board_end + OB_SIZE - x, y, board_size)] = i
            board_pos_id[POS(y, x, board_size)] = i
            board_pos_id[POS(y, board_end + OB_SIZE - x, board_size)] = i
            board_pos_id[POS(x, board_end + OB_SIZE - y, board_size)] = i
            board_pos_id[POS(board_end + OB_SIZE - x, board_end + OB_SIZE - y, board_size)] = i
            board_pos_id[POS(board_end + OB_SIZE - y, x, board_size)] = i
            board_pos_id[POS(board_end + OB_SIZE - y, board_end + OB_SIZE - x, board_size)] = i
            i+=1


cdef void init_corner():
    global corner, corner_neighbor

    free(corner)

    corner = <int *>malloc(4 * sizeof(int))
    corner[0] = POS(board_start, board_start, board_size)
    corner[1] = POS(board_start, board_end, board_size)
    corner[2] = POS(board_end, board_start, board_size)
    corner[3] = POS(board_end, board_end, board_size)

    corner_neighbor = np.zeros((4, 2), dtype=np.int32)
    corner_neighbor[0, 0] = EAST(corner[0])
    corner_neighbor[0, 1] = SOUTH(corner[0], board_size)
    corner_neighbor[1, 0] = NORTH(corner[1], board_size)
    corner_neighbor[1, 1] = EAST(corner[1])
    corner_neighbor[2, 0] = WEST(corner[2])
    corner_neighbor[2, 1] = SOUTH(corner[2], board_size)
    corner_neighbor[3, 0] = NORTH(corner[3], board_size)
    corner_neighbor[3, 1] = WEST(corner[3])


cdef void initialize_neighbor():
    cdef int i
    cdef char empty = 0

    for i in range(pat.PAT3_MAX):
        empty = 0
        if ((i >> 2) & 0x3) == S_EMPTY:
            empty += 1
        if ((i >> 6) & 0x3) == S_EMPTY:
            empty += 1
        if ((i >> 8) & 0x3) == S_EMPTY:
            empty += 1
        if ((i >> 12) & 0x3) == S_EMPTY:
            empty += 1

        nb4_empty[i] = empty


cdef void initialize_eye():
    cdef unsigned int eye_pat3[16]
    cdef unsigned int false_eye_pat3[4]
    cdef unsigned int complete_half_eye[12]
    cdef unsigned int half_3_eye[2]
    cdef unsigned int half_2_eye[4]
    cdef unsigned int half_1_eye[6]
    cdef unsigned int complete_1_eye[5]

    cdef unsigned int transp[8]
    cdef unsigned int pat3_transp16[16]

    cdef int i, j

    """
      眼のパターンはそれぞれ1か所あたり最大2ビットで表現
        012
        3*4
        567
      それぞれの番号×2ビットだけシフトさせる
        +:空点      0
        O:自分の石  1
        X:相手の石 10
        #:盤外     11
    """
    eye_pat3[:] = [
      # +OO     XOO     +O+     XO+
      # O*O     O*O     O*O     O*O
      # OOO     OOO     OOO     OOO
      0x5554, 0x5556, 0x5544, 0x5546,

      # +OO     XOO     +O+     XO+
      # O*O     O*O     O*O     O*O
      # OO+     OO+     OO+     OO+
      0x1554, 0x1556, 0x1544, 0x1546,

      # +OX     XO+     +OO     OOO
      # O*O     O*O     O*O     O*O
      # OO+     +O+     ###     ###
      0x1564, 0x1146, 0xFD54, 0xFD55,

      # +O#     OO#     XOX     XOX
      # O*#     O*#     O+O     O+O
      # ###     ###     OOO     ###
      0xFF74, 0xFF75, 0x5566, 0xFD66,
    ]

    false_eye_pat3[:] = [
      # OOX     OOO     XOO     XO#
      # O*O     O*O     O*O     O*#
      # XOO     XOX     ###     ###
      0x5965, 0x9955, 0xFD56, 0xFF76,
    ]

    complete_half_eye[:] = [
      # XOX     OOX     XOX     XOX     XOX
      # O*O     O*O     O*O     O*O     O*O
      # OOO     XOO     +OO     XOO     +O+
      0x5566, 0x5965, 0x5166, 0x5966, 0x1166,
      # +OX     XOX     XOX     XOO     XO+
      # O*O     O*O     O*O     O*O     O*O
      # XO+     XO+     XOX     ###     ###
      0x1964, 0x1966, 0x9966, 0xFD56, 0xFD46,
      # XOX     XO#
      # O*O     O*#
      # ###     ###
      0xFD66, 0xFF76
    ]

    half_3_eye[:] = [
      # +O+     XO+
      # O*O     O*O
      # +O+     +O+
      0x1144, 0x1146
    ]

    half_2_eye[:] = [
      # +O+     XO+     +OX     +O+
      # O*O     O*O     O*O     O*O
      # +OO     +OO     +OO     ###
      0x5144, 0x5146, 0x5164, 0xFD44,
    ]

    half_1_eye[:] = [
      # +O+     XO+     OOX     OOX     +OO
      # O*O     O*O     O*O     O*O     O*O
      # OOO     OOO     +OO     +OO     ###
      0x5544, 0x5564, 0x5145, 0x5165, 0xFD54,
      # +O#
      # O*#
      # ###
      0xFF74,
    ]

    complete_1_eye[:] = [
      # OOO     +OO     XOO     OOO     OO#
      # O*O     O*O     O*O     O*O     O*#
      # OOO     OOO     OOO     ###     ###
      0x5555, 0x5554, 0x5556, 0xFD55, 0xFF75,
    ]

    fill_n_unsigned_char(eye_condition, E_NOT_EYE, 0);

    for i in range(12):
        pat.pat3_transpose16(complete_half_eye[i], pat3_transp16)
        for j in range(16):
            eye_condition[pat3_transp16[j]] = E_COMPLETE_HALF_EYE

    for i in range(2):
        pat.pat3_transpose16(half_3_eye[i], pat3_transp16);
        for j in range(16):
            eye_condition[pat3_transp16[j]] = E_HALF_3_EYE;

    for i in range(4):
        pat.pat3_transpose16(half_2_eye[i], pat3_transp16);
        for j in range(16):
            eye_condition[pat3_transp16[j]] = E_HALF_2_EYE;

    for i in range(6):
        pat.pat3_transpose16(half_1_eye[i], pat3_transp16);
        for j in range(16):
            eye_condition[pat3_transp16[j]] = E_HALF_1_EYE;

    for i in range(5):
        pat.pat3_transpose16(complete_1_eye[i], pat3_transp16);
        for j in range(16):
            eye_condition[pat3_transp16[j]] = E_COMPLETE_ONE_EYE;

    # BBB
    # B*B
    # BBB
    eye[0x5555] = S_BLACK;

    # WWW
    # W*W
    # WWW
    eye[pat.pat3_reverse(0x5555)] = S_WHITE;

    # +B+
    # B*B
    # +B+
    eye[0x1144] = S_BLACK;

    # +W+
    # W*W
    # +W+
    eye[pat.pat3_reverse(0x1144)] = S_WHITE;

    for i in range(14):
      pat.pat3_transpose8(eye_pat3[i], transp);
      for j in range(8):
        eye[transp[j]] = S_BLACK;
        eye[pat.pat3_reverse(transp[j])] = S_WHITE;

    for i in range(4):
      pat.pat3_transpose8(false_eye_pat3[i], transp);
      for j in range(8):
        false_eye[transp[j]] = S_BLACK;
        false_eye[pat.pat3_reverse(transp[j])] = S_WHITE;


cdef void initialize_const():
    #global komi

    #komi = <double *>malloc(S_OB * sizeof(double))
    #komi[0] = default_komi
    #komi[<int>S_BLACK] = default_komi + 1.0
    #komi[<int>S_WHITE] = default_komi - 1.0

    init_board_position()

    #init_line_number()

    #init_move_distance()

    #init_board_position_id()

    #init_corner()


cdef void clear_const():
    global komi
    global onboard_pos, board_x, board_y

    if komi:
        free(komi)

    if onboard_pos:
        free(onboard_pos)

    if board_x:
        free(board_x)

    if board_y:
        free(board_y)


cdef void set_board_size(int size):
    global pure_board_size, pure_board_max
    global board_size, board_max
    global max_string, max_neighbor
    global board_start, board_end
    global string_lib_max, string_pos_max
    global string_end, liberty_end
    global max_records, max_moves

    pure_board_size = size
    pure_board_max = size ** 2

    board_size = size + 2 * OB_SIZE
    board_max = board_size ** 2

    max_string = pure_board_max * 4 / 5
    max_neighbor = max_string

    board_start = OB_SIZE
    board_end = pure_board_size + OB_SIZE - 1

    string_lib_max = board_size * (size + OB_SIZE)
    string_pos_max = board_size * (size + OB_SIZE)

    string_end = string_pos_max - 1
    liberty_end = string_lib_max - 1

    max_records = pure_board_max * 3
    max_moves = max_records - 1

    pat.N = -board_size
    pat.S = board_size
    pat.W = -1
    pat.E = 1
    pat.NN = pat.N + pat.N
    pat.NW = pat.N + pat.W
    pat.NE = pat.N + pat.E
    pat.SS = pat.S + pat.S
    pat.SW = pat.S + pat.W
    pat.SE = pat.S + pat.E
    pat.WW = pat.W + pat.W
    pat.EE = pat.E + pat.E

    initialize_const()


cdef int get_neighbor4_empty(game_state_t *game, int pos) nogil:
    return nb4_empty[pat.pat3(game.pat, pos)]


cdef bint is_legal(game_state_t *game, int pos, char color) nogil:
    if pos == PASS:
        return True

    if game.board[pos] != S_EMPTY:
        return False

    if nb4_empty[pat.pat3(game.pat, pos)] == 0 and is_suicide(game, pos, color):
        return False

    if game.ko_pos == pos and game.ko_move == (game.moves - 1):
        return False

    return True


cdef bint is_legal_not_eye(game_state_t *game, int pos, char color) nogil:
    if pos == PASS:
        return True

    if game.board[pos] != S_EMPTY:
        game.candidates[pos] = False
        return False

    if (eye[pat.pat3(game.pat, pos)] != <int>color or
        game.string[game.string_id[NORTH(pos, board_size)]].libs == 1 or
        game.string[game.string_id[WEST(pos)]].libs == 1 or
        game.string[game.string_id[EAST(pos)]].libs == 1 or
        game.string[game.string_id[SOUTH(pos, board_size)]].libs == 1):

        if nb4_empty[pat.pat3(game.pat, pos)] == 0 and is_suicide(game, pos, color):
            return False

        if game.ko_pos == pos and game.ko_move == (game.moves - 1):
            return False

        return True

    game.candidates[pos] = False

    return False


cdef bint is_suicide(game_state_t *game, int pos, char color) nogil:
    cdef int neighbor4[4]
    cdef int other = FLIP_COLOR(color)
    cdef int i, neighbor_pos, neighbor_id

    get_neighbor4(neighbor4, pos)

    for i in range(4):
        neighbor_pos = neighbor4[i]
        neighbor_id = game.string_id[neighbor_pos]
        if (game.board[neighbor_pos] == other and
            game.string[neighbor_id].libs == 1):
            return False
        elif (game.board[neighbor_pos] == color and
              game.string[neighbor_id].libs > 1):
            return False

    return True


cdef int calculate_score(game_state_t *game) nogil:
    return 0


cdef void memorize_updated_string(game_state_t *game, int string_id) nogil:
    """ Memorized string_id for incremental rollout feature calculation.
        Number of memorized string is cleared after feature calculation.
    """
    cdef int *num_for_black
    cdef int *id_for_black
    cdef int *num_for_white
    cdef int *id_for_white

    num_for_black = &game.updated_string_num[<int>S_BLACK]
    num_for_white = &game.updated_string_num[<int>S_WHITE]

    if num_for_black[0] < MAX_RECORDS:
        game.updated_string_id[<int>S_BLACK][num_for_black[0]] = string_id
        num_for_black[0] += 1

    if num_for_white[0] < MAX_RECORDS:
        game.updated_string_id[<int>S_WHITE][num_for_white[0]] = string_id
        num_for_white[0] += 1


cpdef test_playout(int n_playout=1, int move_limit=500):
    """ benchmark playout speed
    """
    cdef int i, n
    cdef char color = S_BLACK
    cdef list empties = []
    cdef list move_speeds = [0]
    cdef list policy_feature_speeds = [0]
    cdef list copy_speeds = [0]
    cdef float total_po_sec
    cdef list po_overheads = [0]
    cdef game_state_t *game
    cdef game_state_t *clone
    cdef policy_feature.policy_feature_t *feature

    import itertools
    import random
    import time

    set_board_size(19)

    game = allocate_game()
    #clone = allocate_game()
    #feature = policy_feature.allocate_feature()

    ps = time.time()
    for n in range(n_playout):
        poos = time.time()

        initialize_board(game, False)
        #policy_feature.initialize_feature(feature)

        #policy_feature.update(feature, game)

        pos_generator = itertools.product(range(board_start, board_end + 1), repeat=2)
        empties = [POS(p[0], p[1], board_size) for p in pos_generator]

        po_overheads.append(time.time() - poos)
        for i in range(move_limit):
            if not empties:
                break
            pos = random.choice(empties)
            empties.remove(pos)
            ms = time.time()
            if is_legal_not_eye(game, pos, game.current_color):
                do_move(game, pos)
                move_speeds.append(time.time() - ms)
                """
                Fps = time.time()
                policy_feature.update(feature, game)
                np.asarray(feature.planes).reshape((48, pure_board_size, pure_board_size))
                policy_feature_speeds.append(time.time() - Fps)

                Cps = time.time()
                copy_game(clone, game)
                copy_speeds.append(time.time() - Cps)
                """
        printer.print_board(game)

    total_po_sec = time.time() - ps - np.sum(po_overheads) - np.sum(policy_feature_speeds) - np.sum(copy_speeds)

    free_game(game)
    #free_game(clone)
    #policy_feature.free_feature(feature)

    print("Avg PO. {:.3f} us. ".format(total_po_sec*1000**2/n_playout) +
          "Avg Move. {:.3f} ns. ".format(np.mean(move_speeds)*1000**3) +
          "Avg Copy. {:.3f} us. ".format(np.mean(copy_speeds)*1000**2) +
          "Avg Feature Calc (Policy). {:.3f} us. ".format(np.mean(policy_feature_speeds)*1000**2))

