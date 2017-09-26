import numpy as np

cimport numpy as np

from libc.stdlib cimport malloc, free

from nose.tools import ok_, eq_

from bamboo.board cimport S_EMPTY, S_BLACK, S_WHITE
from bamboo.board cimport game_state_t, allocate_game, free_game
from bamboo.parseboard cimport parse
from bamboo.local_pattern cimport initialize_rands, x33_bits, x33_hash, x33_hash_from_bits

from bamboo.local_pattern import print_x33


def test_x33_bit():
    cdef unsigned long long bits
    cdef game_state_t *game = allocate_game()
    (moves, pure_moves) = parse(game,
                             ". B . B . . .|"
                             ". B W . . . B|"
                             ". B W . . B W|"
                             ". a B W W W W|"
                             "B W B W W B B|"
                             "B W W W . W B|"
                             ". . . . W B .|")

    bits = x33_bits(game, moves['a'], S_BLACK)

    eq_(bits, <unsigned long long>6787556593)
    
    free_game(game)


def test_x33_hash_from_bits_0():
    cdef unsigned long long hash1
    cdef unsigned long long hash2
    cdef game_state_t *game = allocate_game()
    (moves, pure_moves) = parse(game,
                             ". B W . .|"
                             ". B W . .|"
                             ". . a . .|"
                             ". . . . .|"
                             ". . . . .|")

    initialize_rands()

    hash1 = x33_hash(game, moves['a'], <int>S_BLACK)
    hash2 = x33_hash_from_bits(x33_bits(game, moves['a'], <int>S_BLACK))

    eq_(hash1, hash2)

    free_game(game)


def test_x33_hash_from_bits_1():
    cdef unsigned long long hash1
    cdef unsigned long long hash2
    cdef game_state_t *game = allocate_game()
    (moves, pure_moves) = parse(game,
                             ". B . B . . .|"
                             ". B W . . . B|"
                             ". B W . . B W|"
                             ". a B W W W W|"
                             "B W B W W B B|"
                             "B W W W . W B|"
                             ". . . . W B .|")

    initialize_rands()

    hash1 = x33_hash(game, moves['a'], <int>S_BLACK)
    hash2 = x33_hash_from_bits(x33_bits(game, moves['a'], <int>S_BLACK))

    eq_(hash1, hash2)

    free_game(game)
