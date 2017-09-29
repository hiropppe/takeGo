# -*- coding: utf-8 -*-
#cython: boundscheck=False
#cython: wraparound=False

import numpy as np

cimport numpy as np

from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy
from libc.stdint cimport intptr_t
from libc.stdio cimport printf

cimport board 
cimport printer

from bamboo.board cimport LIBERTY_END, NEIGHBOR_END
from bamboo.board cimport NORTH, WEST, EAST, SOUTH
from bamboo.board cimport board_size, liberty_end, string_end, onboard_index, board_dis_x, board_dis_y
from bamboo.board cimport is_true_eye, get_neighbor4, get_neighbor4_empty
from bamboo.pattern cimport pat3


cdef policy_feature_t *allocate_feature(int n_planes):
    cdef policy_feature_t *feature
    cdef int i

    feature = <policy_feature_t *>malloc(sizeof(policy_feature_t))
    memset(feature, 0, sizeof(policy_feature_t))

    feature.planes = np.zeros((n_planes, board.pure_board_max), dtype=np.int32)
    feature.n_planes = feature.planes.shape[0]

    return feature


cdef void initialize_feature(policy_feature_t *feature):
    feature.planes[...] = 0
    for i in range(80):
        board.initialize_board(&feature.search_games[i])


cdef void free_feature(policy_feature_t *feature):
    if feature:
        free(feature)


cdef void update(policy_feature_t *feature, board.game_state_t *game):
    cdef int[:, ::1] F = feature.planes

    cdef char current_color = game.current_color
    cdef char other_color = board.FLIP_COLOR(game.current_color)
    cdef int neighbor4[4]
    cdef int pos, npos, nlib
    cdef char color, ncolor
    cdef int string_id, nstring_id
    cdef board.string_t *string
    cdef board.string_t *nstring
    cdef int capture_size, self_atari_size, libs_after_move
    cdef short neighbor_checked[361][288]   # PURE_BOARD_MAX:MAX_STRING
    cdef board.string_t *capture_string
    cdef int capture_pos
    cdef int neighbor[361][288]
    cdef int north_string_id, west_string_id, east_string_id, south_string_id
    cdef int libpos_after_move[361][483]    # PURE_BOARD_MAX:STRING_LIB_MAX
    cdef int libpos
    cdef int empty_diagonal_stack[200]
    cdef int empty_diagonal_top
    cdef short ladder_checked[288]          # MAX_STRING
    cdef int ladder_capture, ladder_escape, ladder_x, ladder_y
    cdef int escape_options[4]
    cdef int escape_options_num
    cdef int ladder_moves_value = 0
    cdef int *ladder_moves = &ladder_moves_value
    cdef int first_ladder_capture, first_ladder_escape
    cdef int second_ladder_capture, second_ladder_escape
    cdef int ladder_result
    cdef int i, j

    F[...] = 0
    # Ones: A constant plane filled with 1
    F[3, :] = 1

    for i in range(361):
        if i < 288:
            ladder_checked[i] = False
        for j in range(483):
            if j < 288:
                neighbor_checked[i][j] = False
                neighbor[i][j] = 0
            libpos_after_move[i][j] = 0
        neighbor[i][0] = NEIGHBOR_END
        libpos_after_move[i][0] = LIBERTY_END

    for i in range(board.pure_board_max):
        capture_size = 0
        self_atari_size = 1
        libs_after_move = 0
        pos = board.onboard_pos[i]
        color = game.board[pos]

        empty_diagonal_top = 0

        # Stone colour(3): Player stone(0) / opponent stone(1) / empty(2)
        if color == current_color:
            F[0, i] = 1
        elif color == other_color:
            F[1, i] = 1
        else:
            F[2, i] = 1

        if board.is_legal(game, pos, current_color):
            board.get_neighbor4(neighbor4, pos)

            for n in range(4):
                npos = neighbor4[n]
                nstring_id = game.string_id[npos]
                nstring = &game.string[nstring_id]
                if nstring.flag:
                    if nstring.color == current_color:
                        if not neighbor_checked[i][nstring_id]:
                            self_atari_size += nstring.size
                            nlib = nstring.lib[0]
                            while nlib != liberty_end:
                                if nlib != pos and libpos_after_move[i][nlib] == 0:
                                    libpos_after_move[i][nlib] = libpos_after_move[i][0]
                                    libpos_after_move[i][0] = nlib
                                nlib = nstring.lib[nlib]
                            if neighbor[i][nstring_id] == 0:
                                neighbor[i][nstring_id] = neighbor[i][0]
                                neighbor[i][0] = nstring_id
                            neighbor_checked[i][nstring_id] = True
                elif game.board[npos] == board.S_EMPTY:
                    if libpos_after_move[i][npos] == 0:
                        libpos_after_move[i][npos] = libpos_after_move[i][0]
                        libpos_after_move[i][0] = npos

            for n in range(4):
                npos = neighbor4[n]
                nstring_id = game.string_id[npos]
                nstring = &game.string[nstring_id]
                if nstring.flag:
                    if nstring.color != current_color and nstring.libs == 1 and nstring.lib[pos] != 0:
                        # add neighbor pos
                        if libpos_after_move[i][npos] == 0:
                            libpos_after_move[i][npos] = libpos_after_move[i][0]
                            libpos_after_move[i][0] = npos
                        # add captured pos at own string neighbor 
                        if not neighbor_checked[i][nstring_id]:
                            capture_size += nstring.size
                            capture_string = &game.string[nstring_id]
                            capture_pos = capture_string.origin
                            while capture_pos != string_end:
                                if capture_pos != npos and libpos_after_move[i][capture_pos] == 0:
                                    north_string_id = game.string_id[NORTH(capture_pos, board_size)]
                                    if north_string_id and neighbor[i][north_string_id] != 0:
                                        libpos_after_move[i][capture_pos] = libpos_after_move[i][0]
                                        libpos_after_move[i][0] = capture_pos
                                        capture_pos = game.string_next[capture_pos]
                                        continue
                                    west_string_id = game.string_id[WEST(capture_pos)]
                                    if west_string_id and neighbor[i][west_string_id] != 0:
                                        libpos_after_move[i][capture_pos] = libpos_after_move[i][0]
                                        libpos_after_move[i][0] = capture_pos
                                        capture_pos = game.string_next[capture_pos]
                                        continue
                                    east_string_id = game.string_id[EAST(capture_pos)]
                                    if east_string_id and neighbor[i][east_string_id] != 0:
                                        libpos_after_move[i][capture_pos] = libpos_after_move[i][0]
                                        libpos_after_move[i][0] = capture_pos
                                        capture_pos = game.string_next[capture_pos]
                                        continue
                                    south_string_id = game.string_id[SOUTH(capture_pos, board_size)]
                                    if south_string_id and neighbor[i][south_string_id] != 0:
                                        libpos_after_move[i][capture_pos] = libpos_after_move[i][0]
                                        libpos_after_move[i][0] = capture_pos
                                        capture_pos = game.string_next[capture_pos]
                                        continue
                                capture_pos = game.string_next[capture_pos]
                            neighbor_checked[i][nstring_id] = True
            # count libs_after_move
            libpos = libpos_after_move[i][0]
            while libpos != LIBERTY_END:
                libs_after_move += 1
                libpos = libpos_after_move[i][libpos]

            # Capture size(8): How many opponent stones would be captured
            F[20 + board.MIN(capture_size, 7), i] = 1
            # Self-atari size(8): How many of own stones would be captured
            if libs_after_move == 1:
                F[28 + board.MIN(self_atari_size, 8) - 1, i] = 1
            # Liberties after move(8): Number of liberties after this move is played
            F[36 + board.MIN(libs_after_move, 8) - 1, i] = 1
            # Whether a move is legal and does not fill its own eyes
            if not is_true_eye(game, pos, color, other_color, empty_diagonal_stack, empty_diagonal_top):
                F[46, i] = 1
            # Player color(1): Whether current player is black (value network only)
            if feature.n_planes == MAX_VALUE_PLANES:
                F[48, i] = (current_color == board.S_BLACK)
        else:
            if color:
                string_id = game.string_id[pos]
                string = &game.string[string_id]
                # Turns since(8): How many turns since a move was played
                if game.birth_move[pos]:
                    F[4 + board.MIN(game.moves - game.birth_move[pos], 7), i] = 1

                # Liberties(8): Number of liberties (empty adjacent points)
                F[12 + board.MIN(string.libs, 8) - 1, i] = 1

                if not ladder_checked[string_id]:
                    # Ladder capture(1): Whether a move at this point is a successful ladder capture
                    if string.libs == 2 and string.color == other_color:
                        # 1st candidate
                        ladder_moves[0] = 0
                        first_ladder_capture = string.lib[0]
                        first_ladder_escape = string.lib[string.lib[0]]
                        capture_result = is_ladder_capture(game,
                                            string_id,
                                            first_ladder_capture,
                                            first_ladder_escape,
                                            feature.search_games,
                                            0,
                                            ladder_moves)
                        if capture_result == 1:
                            F[44, onboard_index[first_ladder_capture]] = 1

                        # 2nd candidate
                        ladder_moves[0] = 0
                        second_ladder_capture = first_ladder_escape
                        second_ladder_escape = first_ladder_capture
                        capture_result = is_ladder_capture(game,
                                            string_id,
                                            second_ladder_capture,
                                            second_ladder_escape,
                                            feature.search_games,
                                            0,
                                            ladder_moves)
                        if capture_result == 1:
                            F[44, onboard_index[second_ladder_capture]] = 1

                    # Ladder escape(1): Whether a move at this point is a successful ladder escape
                    elif string.libs == 1 and string.color == current_color:
                        ladder_moves[0] = 0
                        ladder_result =  is_ladder_escape(game,
                                            string_id,
                                            string.lib[0],
                                            True, # is atari-pos
                                            feature.search_games,
                                            0,
                                            ladder_moves)
                        if ladder_result == 1:
                            F[45, onboard_index[string.lib[0]]] = 1
                        """ Add capturing opponent stones to escape options
                        escape_option_num = get_escape_options(game,
                                                               escape_options,
                                                               string.lib[0],
                                                               current_color,
                                                               string_id)
                        for j in range(escape_option_num):
                            ladder_moves[0] = 0
                            if is_ladder_escape(game,
                                                string_id,
                                                escape_options[j],
                                                j == escape_option_num - 1, # is atari-pos
                                                feature.search_games,
                                                0,
                                                ladder_moves):
                                F[45, onboard_index[escape_options[j]]] = 1
                        """
                    ladder_checked[string_id] = True


cdef int get_escape_options(board.game_state_t *game,
                            int escape_options[4],
                            int atari_pos,
                            int escape_color,
                            int string_id):
    cdef board.string_t *string
    cdef board.string_t *neighbor
    cdef int neighbor_id
    cdef int escape_options_num = 0 

    # Add capturing atari neighbor to options
    string = &game.string[string_id]
    neighbor_id = string.neighbor[0]
    while neighbor_id != board.NEIGHBOR_END:
        neighbor = &game.string[neighbor_id]
        if neighbor.libs == 1 and board.is_legal(game, neighbor.lib[0], escape_color):
            escape_options[escape_options_num] = neighbor.lib[0]
            escape_options_num += 1
        neighbor_id = string.neighbor[neighbor_id]

    # prior capture to escape
    escape_options[escape_options_num] = atari_pos
    escape_options_num += 1

    return escape_options_num


cdef int is_ladder_capture(board.game_state_t *game,
                            int string_id,
                            int pos,
                            int atari_pos,
                            board.game_state_t search_games[80],
                            int depth,
                            int *ladder_moves):
    cdef board.game_state_t *ladder_game = &search_games[depth]
    cdef board.string_t *string
    cdef char capture_color = game.current_color
    cdef char escape_color = board.FLIP_COLOR(game.current_color)
    cdef int escape_options[4]
    cdef int escape_options_num
    cdef int escape_result
    cdef int i

    if depth >= MAX_LADDER_DEPTH:
        return 0

    if ladder_moves[0] > MAX_LADDER_MOVES:
        return -1

    """
    printer.print_board(game)
    print('IsCapture? Move: ({:d}, {:d}) Color: {:d} IsLegal: {:s} Depth: {:d}'.format(
        board.CORRECT_X(pos, board.board_size, board.OB_SIZE),
        board.CORRECT_Y(pos, board.board_size, board.OB_SIZE),
        capture_color,
        str(board.is_legal(game, pos, capture_color)),
        depth))
    """
    if not board.is_legal(game, pos, capture_color):
        """
        printer.print_board(game)
        print('Unable to capture !! Illegal Move: ({:d}, {:d}) Color: {:d}'.format(
            board.CORRECT_X(pos, board.board_size, board.OB_SIZE), board.CORRECT_Y(pos, board.board_size, board.OB_SIZE), capture_color))
        """
        return -1

    ladder_moves[0] += 1

    board.copy_game(ladder_game, game)
    board.put_stone(ladder_game, pos, ladder_game.current_color)
    ladder_game.current_color = board.FLIP_COLOR(ladder_game.current_color)

    escape_options_num = get_escape_options(ladder_game,
                                            escape_options,
                                            atari_pos,
                                            escape_color,
                                            string_id)
    for i in range(escape_options_num):
        escape_result =  is_ladder_escape(ladder_game,
                            string_id,
                            escape_options[i],
                            i == escape_options_num - 1, # is atari-pos
                            search_games,
                            depth+1,
                            ladder_moves)
        if escape_result == 1:
            return -1
    return 1


cdef int is_ladder_escape(board.game_state_t *game,
                          int string_id,
                          int pos,
                          bint is_atari_pos,
                          board.game_state_t search_games[80],
                          int depth,
                          int *ladder_moves):
    cdef board.game_state_t *ladder_game = &search_games[depth]
    cdef board.string_t *string
    cdef char escape_color = game.current_color
    cdef char capture_color = board.FLIP_COLOR(game.current_color)
    cdef int neighbor_id
    cdef board.string_t *neighbor_string
    cdef int first_ladder_capture, first_ladder_escape
    cdef int second_ladder_capture, second_ladder_escape
    cdef int capture_result
    cdef int j

    if depth >= MAX_LADDER_DEPTH:
       return 0

    if ladder_moves[0] > MAX_LADDER_MOVES:
       return 1

    """
    printer.print_board(game)
    print('IsEscape? Move: ({:d}, {:d}) Color: {:d} IsLegal: {:s} Depth: {:d}'.format(
        board.CORRECT_X(pos, board.board_size, board.OB_SIZE),
        board.CORRECT_Y(pos, board.board_size, board.OB_SIZE),
        capture_color,
        str(board.is_legal(game, pos, capture_color)),
        depth))
    """
    if not board.is_legal(game, pos, escape_color):
        """
        printer.print_board(game)
        print('Unable to escape !! Illegal Move: ({:d}, {:d}) Color: {:d}'.format(
            board.CORRECT_X(pos, board.board_size, board.OB_SIZE), board.CORRECT_Y(pos, board.board_size, board.OB_SIZE), capture_color))
        """
        return -1

    ladder_moves[0] += 1

    board.copy_game(ladder_game, game)
    board.put_stone(ladder_game, pos, ladder_game.current_color)
    ladder_game.current_color = board.FLIP_COLOR(ladder_game.current_color)

    # May change string_id by string merge
    if is_atari_pos:
        string_id = ladder_game.string_id[pos]

    string = &ladder_game.string[string_id]
    if string.libs == 1:
        """
        printer.print_board(ladder_game)
        print 'Captured !!'
        """
        return -1
    elif string.libs >= 3:
        """
        printer.print_board(ladder_game)
        print 'Escaped !!'
        """
        return 1
    else:
        first_ladder_capture = string.lib[0]
        first_ladder_escape = string.lib[string.lib[0]]
        second_ladder_capture = first_ladder_escape
        second_ladder_escape = first_ladder_capture
        capture_result = is_ladder_capture(ladder_game,
                             string_id,
                             first_ladder_capture,
                             first_ladder_escape,
                             search_games,
                             depth+1,
                             ladder_moves)
        if capture_result == 1:
            return -1
        else:
            capture_result = is_ladder_capture(ladder_game,
                                 string_id,
                                 second_ladder_capture,
                                 second_ladder_escape,
                                 search_games,
                                 depth+1,
                                 ladder_moves)
            if capture_result == 1:
                return -1
            else:
                return 1
