# -*- coding: utf-8 -*-
#cython: boundscheck=False
#cython: wraparound=False

from libcpp.string cimport string as cppstring

cimport board 
cimport point


cdef void print_board(board.game_state_t *game):
    cdef char stone[<int>board.S_MAX]
    cdef int i, x, y, pos
    cdef list buf = []

    stone[:] = ['+', 'B', 'W', '#']

    buf.append("Prisoner(Black) : {:d}\n".format(game.prisoner[<int>board.S_BLACK]))
    buf.append("Prisoner(White) : {:d}\n".format(game.prisoner[<int>board.S_WHITE]))
    buf.append("Move : {:d}\n".format(game.moves))

    buf.append("    ")
    i = 1
    for _ in range(board.board_start, board.board_end + 1):
        buf.append(" {:s}".format(cppstring(1, <char>point.gogui_x[i])))
        i += 1
    buf.append("\n")

    buf.append("   +")
    for i in range(board.pure_board_size * 2 + 1):
        buf.append("-")
    buf.append("+\n")

    i = 1
    for y in range(board.board_start, board.board_end + 1):
        buf.append("{:d}:|".format(board.pure_board_size + 1 - i).rjust(4, ' '))
        for x in range(board.board_start, board.board_end + 1):
            pos = board.POS(x, y, board.board_size)
            buf.append(" {:s}".format(cppstring(1, stone[<int>game.board[pos]])))
        buf.append(" |\n")
        i += 1

    buf.append("   +")
    for i in range(1, board.pure_board_size * 2 + 1 + 1):
        buf.append("-")
    buf.append("+\n")

    print(''.join(buf))
