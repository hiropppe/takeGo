# -*- coding: utf-8 -*-
#cython: boundscheck=False
#cython: wraparound=False

import numpy as np

cimport numpy as np

from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy
from libc.stdint cimport intptr_t

cimport board 
cimport printer


cdef policy_feature_t *allocate_feature():
    cdef policy_feature_t *feature
    cdef int i

    feature = <policy_feature_t *>malloc(sizeof(policy_feature_t))
    memset(feature, 0, sizeof(policy_feature_t))

    feature.planes = np.zeros((MAX_POLICY_PLANES, board.pure_board_max), dtype=np.int32)

    return feature


cdef void initialize_feature(policy_feature_t *feature):
    feature.planes[...] = 0

    for i in range(80):
        board.initialize_board(&feature.search_games[i], False)


cdef void free_feature(policy_feature_t *feature):
    if feature:
        free(feature)


cdef void update(policy_feature_t *feature, board.game_state_t *game):
    cdef int[:, ::1] F = feature.planes

    cdef char current_color = game.current_color
    cdef char other_color = board.FLIP_COLOR(game.current_color)
    cdef int neighbor4[4]
    cdef int pos, npos
    cdef char color, ncolor
    cdef int string_id, nstring_id
    cdef board.string_t *string
    cdef board.string_t *nstring
    cdef int capture_size, self_atari_size, libs_after_move
    cdef short ladder_checked[288] # MAX_STRING
    cdef int i, j
    cdef int ladder_capture, ladder_escape, ladder_x, ladder_y
    cdef int escape_options[4]
    cdef int escape_options_num
    cdef int ladder_moves[1] # workaround. wanna use int pointer

    F[...] = 0
    # Ones: A constant plane filled with 1
    F[3, :] = 1
    board.fill_n_short(ladder_checked, 288, 0)

    for i in range(board.pure_board_max):
        capture_size = self_atari_size = libs_after_move = 0
        pos = board.onboard_pos[i]
        color = game.board[pos]
        # Stone colour(3): Player stone(0) / opponent stone(1) / empty(2)
        if color == current_color:
            F[0, i] = 1
        elif color == other_color:
            F[1, i] = 1
        else:
            F[2, i] = 1

        if board.is_legal_not_eye(game, pos, current_color):
            board.get_neighbor4(neighbor4, pos)
            for n in range(4):
                npos = neighbor4[n]
                nstring_id = game.string_id[npos]
                if nstring_id:
                    nstring = &game.string[nstring_id]
                    if nstring.libs == 1 and nstring.lib[pos] != 0:
                        if nstring.color == current_color:
                            self_atari_size += nstring.size
                        else:
                            capture_size += nstring.size

                    if nstring.color == current_color:
                        libs_after_move += (nstring.libs - 1)
                    elif nstring.libs == 1 and nstring.lib[pos] == 0:
                        libs_after_move += 1
                elif game.board[npos] != board.S_OB:
                    libs_after_move += 1
            # Capture size(8): How many opponent stones would be captured
            F[20 + board.MIN(capture_size, 7), i] = 1
            # Self-atari size(8): How many of own stones would be captured
            if self_atari_size:
                F[28 + board.MIN(self_atari_size, 8) - 1, i] = 1
            # Liberties after move(8): Number of liberties after this move is played
            F[36 + board.MIN(libs_after_move, 8) - 1, i] = 1
            F[46, i] = 1
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
                        for j in range(2):
                            if j == 0:
                                ladder_capture = string.lib[0]
                                ladder_escape = string.lib[string.lib[0]]
                            else:
                                ladder_capture = string.lib[string.lib[0]]
                                ladder_escape = string.lib[0]
                            ladder_moves[0] = 0
                            if is_ladder_capture(game,
                                                 string_id,
                                                 ladder_capture,
                                                 ladder_escape,
                                                 feature.search_games,
                                                 0,
                                                 ladder_moves):
                                ladder_x = board.CORRECT_X(ladder_capture, board.board_size, board.OB_SIZE)
                                ladder_y = board.CORRECT_Y(ladder_capture, board.board_size, board.OB_SIZE)
                                F[44, board.POS(ladder_x, ladder_y, board.pure_board_size)] = 1
                    # Ladder escape(1): Whether a move at this point is a successful ladder escape
                    elif string.libs == 1 and string.color == current_color:
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
                                ladder_x = board.CORRECT_X(escape_options[j], board.board_size, board.OB_SIZE)
                                ladder_y = board.CORRECT_Y(escape_options[j], board.board_size, board.OB_SIZE)
                                F[45, board.POS(ladder_x, ladder_y, board.pure_board_size)] = 1
                            """
                            # see break ladder if capture one of neighbors. no additional search
                            if escape_options[j] != string.lib[0]:
                                ladder_x = board.CORRECT_X(escape_options[j], board.board_size, board.OB_SIZE)
                                ladder_y = board.CORRECT_Y(escape_options[j], board.board_size, board.OB_SIZE)
                                F[45, board.POS(ladder_x, ladder_y, board.pure_board_size)] = 1
                            elif is_ladder_escape(game, string_id, escape_options[j], j == escape_option_num - 1, feature.search_games, 0):
                                ladder_x = board.CORRECT_X(escape_options[j], board.board_size, board.OB_SIZE)
                                ladder_y = board.CORRECT_Y(escape_options[j], board.board_size, board.OB_SIZE)
                                F[45, board.POS(ladder_x, ladder_y, board.pure_board_size)] = 1
                            """
                    ladder_checked[string_id] = True


cdef bint has_atari_neighbor(board.game_state_t *game, int string_id, char escape_color):
    cdef board.string_t *string
    cdef board.string_t *neighbor
    cdef int neighbor_id

    string = &game.string[string_id]
    neighbor_id = string.neighbor[0]
    while neighbor_id != board.neighbor_end:
        neighbor = &game.string[neighbor_id]
        if neighbor.libs == 1 and board.is_legal(game, neighbor.lib[0], escape_color):
            return True
        else:
            neighbor_id = string.neighbor[neighbor_id]


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
    # prior to escape atari
    string = &game.string[string_id]
    neighbor_id = string.neighbor[0]
    while neighbor_id != board.neighbor_end:
        neighbor = &game.string[neighbor_id]
        if neighbor.libs == 1 and board.is_legal(game, neighbor.lib[0], escape_color):
            escape_options[escape_options_num] = neighbor.lib[0]
            escape_options_num += 1
        neighbor_id = string.neighbor[neighbor_id]

    escape_options[escape_options_num] = atari_pos
    escape_options_num += 1

    return escape_options_num


cdef bint is_ladder_capture(board.game_state_t *game,
                            int string_id,
                            int pos,
                            int atari_pos,
                            board.game_state_t search_games[80],
                            int depth,
                            int ladder_moves[1]):
    cdef board.game_state_t *ladder_game = &search_games[depth]
    cdef board.string_t *string
    cdef char capture_color = game.current_color
    cdef char escape_color = board.FLIP_COLOR(game.current_color)
    cdef int escape_options[4]
    cdef int escape_options_num
    cdef int i

    if depth >= 80 or ladder_moves[0] > 200:
        return False

    # print '(', board.CORRECT_X(pos, board.board_size, board.OB_SIZE), ',', board.CORRECT_Y(pos, board.board_size, board.OB_SIZE), ')', capture_color, board.is_legal(game, pos, capture_color), board.is_suicide(game, pos, capture_color), depth
    # printer.print_board(game)
    if not board.is_legal(game, pos, capture_color):
        # printer.print_board(game)
        # print 'Unable to capture !! not legal at (', board.CORRECT_X(pos, board.board_size, board.OB_SIZE), ',', board.CORRECT_Y(pos, board.board_size, board.OB_SIZE), ')', capture_color
        return False

    ladder_moves[0] += 1

    board.copy_game(ladder_game, game)
    board.do_move(ladder_game, pos)
    """
    # see break ladder if capture one of neighbors. no additional search
    return not (has_atari_neighbor(ladder_game, string_id, escape_color) or
           is_ladder_escape(ladder_game, string_id, atari_pos, True, search_games, depth+1))
    """
    escape_options_num = get_escape_options(ladder_game,
                                            escape_options,
                                            atari_pos,
                                            escape_color,
                                            string_id)
    for i in range(escape_options_num):
        if is_ladder_escape(ladder_game,
                            string_id,
                            escape_options[i],
                            i == escape_options_num - 1, # is atari-pos
                            search_games,
                            depth+1,
                            ladder_moves):
            return False
    return True


cdef bint is_ladder_escape(board.game_state_t *game,
                           int string_id,
                           int pos,
                           bint is_atari_pos,
                           board.game_state_t search_games[80],
                           int depth,
                           int ladder_moves[1]):
    cdef board.game_state_t *ladder_game = &search_games[depth]
    cdef board.string_t *string
    cdef char escape_color = game.current_color
    cdef char capture_color = board.FLIP_COLOR(game.current_color)
    cdef int neighbor_id
    cdef board.string_t *neighbor_string
    cdef int ladder_capture, ladder_escape
    cdef int j

    if depth >= 80 or ladder_moves[0] > 200:
       return False

    # print '(', board.CORRECT_X(pos, board.board_size, board.OB_SIZE), ',', board.CORRECT_Y(pos, board.board_size, board.OB_SIZE), ')', escape_color, board.is_legal(game, pos, escape_color), board.is_suicide(game, pos, escape_color), depth
    # printer.print_board(game)
    if not board.is_legal(game, pos, escape_color):
        # printer.print_board(game)
        # print 'Unable to escape !! not legal at (', board.CORRECT_X(pos, board.board_size, board.OB_SIZE), ',', board.CORRECT_Y(pos, board.board_size, board.OB_SIZE), ')', capture_color
        return False

    ladder_moves[0] += 1

    board.copy_game(ladder_game, game)
    board.do_move(ladder_game, pos)
    if is_atari_pos:
        string_id = ladder_game.string_id[pos]
    string = &ladder_game.string[string_id]
    if string.libs == 1:
        # printer.print_board(ladder_game)
        # print 'Captured !!'
        return False
    elif string.libs >= 3:
        # printer.print_board(ladder_game)
        # print 'Escaped !!'
        return True
    else:
        for j in range(2):
            if j == 0:
                ladder_capture = string.lib[0]
                ladder_escape = string.lib[string.lib[0]]
            else:
                ladder_capture = string.lib[string.lib[0]]
                ladder_escape = string.lib[0]

            if is_ladder_capture(ladder_game,
                                 string_id,
                                 ladder_capture,
                                 ladder_escape,
                                 search_games,
                                 depth+1,
                                 ladder_moves):
                return False

        return True
