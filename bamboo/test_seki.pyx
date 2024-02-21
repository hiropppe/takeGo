from libc.string cimport memset

import numpy as np
cimport numpy as np

from bamboo.board cimport S_BLACK, S_WHITE
from bamboo.board cimport FLIP_COLOR
from bamboo.board cimport game_state_t
from bamboo.board cimport allocate_game, free_game, put_stone
from bamboo.parseboard cimport parse

from bamboo.seki cimport check_seki


def setup():
    pass


def test_seki_0():
    cdef game_state_t *game = allocate_game()
    cdef int seki[529]

    memset(seki, 0, sizeof(int)*529)

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

    assert (True == seki[moves['a']])
    assert (True == seki[moves['b']])

    assert (12 == np.asarray(seki).sum())


def test_seki_1():
    cdef game_state_t *game = allocate_game()
    cdef int seki[529]

    memset(seki, 0, sizeof(int)*529)

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

    assert (True == seki[moves['a']])
    assert (True == seki[moves['b']])

    assert (9 == np.asarray(seki).sum())


def test_seki_2():
    cdef game_state_t *game = allocate_game()
    cdef int seki[529]

    memset(seki, 0, sizeof(int)*529)

    (moves, pure_moves) = parse(game,
                              ". . . W W B B W a|"
                              ". . . W B b B c W|"
                              ". . . W B B B B B|"
                              ". . . W W W W W W|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|")

    check_seki(game, seki)

    assert (True == seki[moves['a']])
    assert (True == seki[moves['b']])
    assert (True == seki[moves['c']])

    assert (14 == np.asarray(seki).sum())


def test_seki_3():
    cdef game_state_t *game = allocate_game()
    cdef int seki[529]

    memset(seki, 0, sizeof(int)*529)

    (moves, pure_moves) = parse(game,
                              ". . . . . . . . .|"
                              ". . . . . . . . .|"
                              ". . W W W B B B .|"
                              ". W W B B W W W B|"
                              ". W B B B W a W B|"
                              "W B b B c W W B .|"
                              "W B B B W W B B .|"
                              ". W W W B B B . .|"
                              ". . . . . . . . .|")

    check_seki(game, seki)

    assert (True == seki[moves['a']])
    assert (True == seki[moves['b']])
    assert (True == seki[moves['c']])

    assert (22 == np.asarray(seki).sum())


def test_seki_4():
    cdef game_state_t *game = allocate_game()
    cdef int seki[529]

    memset(seki, 0, sizeof(int)*529)

    (moves, pure_moves) = parse(game,
                              "W a W B B B W . .|"
                              "W W W B b B W . .|"
                              "c B B B B W W . .|"
                              "B B B W W W . . .|"
                              "W W W W . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|")

    check_seki(game, seki)

    assert (True == seki[moves['a']])
    assert (True == seki[moves['b']])
    assert (True == seki[moves['c']])

    assert (20 == np.asarray(seki).sum())


def test_seki_5():
    cdef game_state_t *game = allocate_game()
    cdef int seki[529]

    memset(seki, 0, sizeof(int)*529)

    (moves, pure_moves) = parse(game,
                              ". . W B a B W B .|"
                              ". . W B B W W B .|"
                              ". . W B b W W B .|"
                              ". . W B B W W B .|"
                              ". . W B W c W B .|"
                              ". . W B B W W B .|"
                              ". . W W B B B W .|"
                              ". . . . W W W W .|"
                              ". . . . . . . . .|")

    check_seki(game, seki)

    assert (True == seki[moves['a']])
    assert (True == seki[moves['b']])
    assert (True == seki[moves['c']])

    assert (25 == np.asarray(seki).sum())


def test_seki_6():
    cdef game_state_t *game = allocate_game()
    cdef int seki[529]

    memset(seki, 0, sizeof(int)*529)

    (moves, pure_moves) = parse(game,
                              "W a B b B W . . .|"
                              "c W B B B W . . .|"
                              "B B W W W W . . .|"
                              "d B W . . . . . .|"
                              "B B W . . . . . .|"
                              "W W W . . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|")

    check_seki(game, seki)

    assert (True == seki[moves['a']])
    assert (True == seki[moves['b']])
    assert (True == seki[moves['c']])
    assert (True == seki[moves['d']])

    assert (16 == np.asarray(seki).sum())


def test_seki_7():
    cdef game_state_t *game = allocate_game()
    cdef int seki[529]

    memset(seki, 0, sizeof(int)*529)

    (moves, pure_moves) = parse(game,
                              "a B B b W B . . .|"
                              "B W W W W B . . .|"
                              "c W B B B B . . .|"
                              "W W B . . . . . .|"
                              "B B B . . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|")

    check_seki(game, seki)

    assert (True == seki[moves['a']])
    assert (True == seki[moves['b']])
    assert (True == seki[moves['c']])

    assert (14 == np.asarray(seki).sum())


def test_seki_8():
    cdef game_state_t *game = allocate_game()
    cdef int seki[529]

    memset(seki, 0, sizeof(int)*529)

    (moves, pure_moves) = parse(game,
                              "a B c B W . . . .|"
                              "W W B B W . . . .|"
                              "b B B W W . . . .|"
                              "B B W W . . . . .|"
                              "W W W . . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|")

    check_seki(game, seki)

    assert (True == seki[moves['a']])
    assert (True == seki[moves['b']])
    assert (True == seki[moves['c']])

    assert (13 == np.asarray(seki).sum())


def test_seki_9():
    cdef game_state_t *game = allocate_game()
    cdef int seki[529]

    memset(seki, 0, sizeof(int)*529)

    (moves, pure_moves) = parse(game,
                              "W B a W W B . . .|"
                              "W b B W W B . . .|"
                              "W W W W W B . . .|"
                              "B B B B B B . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|")

    check_seki(game, seki)

    assert (True == seki[moves['a']])
    assert (True == seki[moves['b']])

    assert (15 == np.asarray(seki).sum())


def test_bent4():
    cdef game_state_t *game = allocate_game()
    cdef int seki[529]

    memset(seki, 0, sizeof(int)*529)

    (moves, pure_moves) = parse(game,
                              "W a B W . . . . .|"
                              "W B B W . . . . .|"
                              "W b B W . . . . .|"
                              "B B B W . . . . .|"
                              "W W W W . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|")

    check_seki(game, seki)

    assert (True == seki[moves['a']])
    assert (True == seki[moves['b']])

    assert (12 == np.asarray(seki).sum())
