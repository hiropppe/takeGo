from libc.string cimport memset

from nose.tools import ok_, eq_

import numpy as np
cimport numpy as np

from bamboo.board cimport S_BLACK, S_WHITE
from bamboo.board cimport FLIP_COLOR
from bamboo.board cimport game_state_t
from bamboo.board cimport allocate_game, free_game, put_stone
from bamboo.parseboard cimport parse

from bamboo.seki cimport is_self_atari, check_seki


def setup():
    pass


def test_seki_0():
    cdef game_state_t *game = allocate_game()
    cdef bint seki[529]

    memset(seki, 0, sizeof(bint)*529)

    (moves, pure_moves) = parse(game,
                              ". . . . . . . . .|"
                              ". . . . . . . . .|"
                              ". . W W W B B . .|"
                              ". . W B B W B . .|"
                              ". . W B a W B . .|"
                              ". . W B b W B . .|"
                              ". . W B W W B . .|"
                              ". . W W B B B . .|"
                              ". . . . . . . . .|")

    check_seki(game, seki)

    eq_(True, seki[moves['a']])
    eq_(True, seki[moves['b']])

    eq_(2, np.asarray(seki).sum())


def test_seki_1():
    cdef game_state_t *game = allocate_game()
    cdef bint seki[529]

    memset(seki, 0, sizeof(bint)*529)

    (moves, pure_moves) = parse(game,
                              ". . B W a B W . .|"
                              ". . B W b B W . .|"
                              ". . B W B B W . .|"
                              ". . B B W W W . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|")

    check_seki(game, seki)

    eq_(True, seki[moves['a']])
    eq_(True, seki[moves['b']])

    eq_(2, np.asarray(seki).sum())


def test_seki_2():
    cdef game_state_t *game = allocate_game()
    cdef bint seki[529]

    memset(seki, 0, sizeof(bint)*529)

    (moves, pure_moves) = parse(game,
                              ". . . W W B B W .|"
                              ". . . W B . B a W|"
                              ". . . W B B B B B|"
                              ". . . W W W W W W|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|")

    check_seki(game, seki)

    eq_(True, seki[moves['a']])

    eq_(1, np.asarray(seki).sum())


def test_seki_3():
    cdef game_state_t *game = allocate_game()
    cdef bint seki[529]

    memset(seki, 0, sizeof(bint)*529)

    (moves, pure_moves) = parse(game,
                              ". . . . . . . . .|"
                              ". . . . . . . . .|"
                              ". . W W W B B B .|"
                              ". W W B B W W W B|"
                              ". W B B B W . W B|"
                              "W B . B a W W B .|"
                              "W B B B W W B B .|"
                              ". W W W B B B . .|"
                              ". . . . . . . . .|")

    check_seki(game, seki)

    eq_(True, seki[moves['a']])

    eq_(1, np.asarray(seki).sum())


def test_seki_4():
    cdef game_state_t *game = allocate_game()
    cdef bint seki[529]

    memset(seki, 0, sizeof(bint)*529)

    (moves, pure_moves) = parse(game,
                              "W . W B B B W . .|"
                              "W W W B . B W . .|"
                              "a B B B B W W . .|"
                              "B B B W W W . . .|"
                              "W W W W . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|")

    check_seki(game, seki)

    eq_(True, seki[moves['a']])

    eq_(1, np.asarray(seki).sum())


def test_seki_5():
    cdef game_state_t *game = allocate_game()
    cdef bint seki[529]

    memset(seki, 0, sizeof(bint)*529)

    (moves, pure_moves) = parse(game,
                              ". . W B . B W B .|"
                              ". . W B B W W B .|"
                              ". . W B a W W B .|"
                              ". . W B B W W B .|"
                              ". . W B W . W B .|"
                              ". . W B B W W B .|"
                              ". . W W B B B W .|"
                              ". . . . W W W W .|"
                              ". . . . . . . . .|")

    check_seki(game, seki)

    eq_(True, seki[moves['a']])

    eq_(1, np.asarray(seki).sum())

