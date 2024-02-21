import numpy as np

cimport numpy as np

from libc.stdlib cimport malloc, free

from bamboo.board cimport S_EMPTY, S_BLACK, S_WHITE
from bamboo.board cimport game_state_t, allocate_game, free_game, put_stone
from bamboo.parseboard cimport parse
from bamboo.local_pattern cimport initialize_rands, d12_rsp_bits, d12_rsp_hash, d12_rsp_hash_from_bits, d12_pos_mt
from bamboo.local_pattern cimport d12_rspos_bits, d12_rspos_hash, d12_rspos_hash_from_bits

from bamboo.local_pattern import print_d12, print_d12_rspos


def test_d12_rsp_bits_0():
    cdef unsigned long long bits
    cdef game_state_t *game = allocate_game()
    cdef int empty_ix[12]
    cdef int empty_pos[12]
    cdef int n_empty_val = 0
    cdef int *n_empty = &n_empty_val

    (moves, pure_moves) = parse(game,
                             ". . . . . . .|"
                             ". . . B . . .|"
                             ". B W B b . .|"
                             ". B B a B . .|"
                             ". . B W B . .|"
                             ". . . W . . .|"
                             ". . . . . . .|")

    put_stone(game, moves['a'], S_WHITE)
    bits = d12_rsp_bits(game, moves['a'], S_WHITE, empty_ix, empty_pos, n_empty)

    assert (bits == <unsigned long long>0b1001100100010101000110011011111111001111110011011111)
    
    free_game(game)


def test_d12_rsp_hash_0():
    cdef unsigned long long hash1
    cdef unsigned long long hash2
    cdef game_state_t *game = allocate_game()
    cdef int empty_ix[12]
    cdef int empty_pos[12]
    cdef int n_empty_val = 0
    cdef int *n_empty = &n_empty_val

    (moves, pure_moves) = parse(game,
                             ". . . . . . .|"
                             ". . . B . . .|"
                             ". B W B b . .|"
                             ". B B a B . .|"
                             ". . B W B . .|"
                             ". . . W . . .|"
                             ". . . . . . .|")

    initialize_rands()

    put_stone(game, moves['a'], S_WHITE)

    hash1 = d12_rsp_hash(game, moves['a'], <int>S_WHITE, empty_ix, empty_pos, n_empty)
    bits = d12_rsp_bits(game, moves['a'], S_WHITE, empty_ix, empty_pos, n_empty)
    hash2 = d12_rsp_hash_from_bits(bits)

    assert (hash1 == hash2)

    free_game(game)


def test_d12_rspos_bits_0():
    cdef unsigned long long bits
    cdef game_state_t *game = allocate_game()
    cdef int empty_ix[12]
    cdef int empty_pos[12]
    cdef int n_empty_val = 0
    cdef int *n_empty = &n_empty_val

    (moves, pure_moves) = parse(game,
                             ". . . . . . .|"
                             ". . . B . . .|"
                             ". B W B b . .|"
                             ". B B a B . .|"
                             ". . B W B . .|"
                             ". . . W . . .|"
                             ". . . . . . .|")

    put_stone(game, moves['a'], S_WHITE)
    bits = d12_rspos_bits(game, moves['a'], S_WHITE, empty_ix, empty_pos, n_empty)
    bits |= (1 << 3) 

    assert (bits == <unsigned long long>0b1001100100010101000110011011111111001111110011011111000000001000)
    assert(n_empty[0] == 2)
    assert(empty_ix[0] == 3)
    assert(empty_ix[1] == 7)
    assert(empty_pos[0] == 50)
    assert(empty_pos[1] == 62)
    
    free_game(game)


def test_d12_rspos_hash_0():
    cdef unsigned long long hash1
    cdef unsigned long long hash2
    cdef game_state_t *game = allocate_game()
    cdef int empty_ix[12]
    cdef int empty_pos[12]
    cdef int n_empty_val = 0
    cdef int *n_empty = &n_empty_val

    (moves, pure_moves) = parse(game,
                             ". . . . . . .|"
                             ". . . B . . .|"
                             ". B W B b . .|"
                             ". B B a B . .|"
                             ". . B W B . .|"
                             ". . . W . . .|"
                             ". . . . . . .|")

    initialize_rands()

    put_stone(game, moves['a'], S_WHITE)

    hash1 = d12_rspos_hash(game, moves['a'], <int>S_WHITE, empty_ix, empty_pos, n_empty)
    hash1 ^= d12_pos_mt[1 << 3]

    assert(n_empty[0] == 2)
    assert(empty_ix[0] == 3)
    assert (empty_ix[1] == 7)
    assert (empty_pos[0] == 50)
    assert (empty_pos[1] == 62)

    bits = d12_rspos_bits(game, moves['a'], S_WHITE, empty_ix, empty_pos, n_empty)
    bits |= (1 << 3) 

    assert (n_empty[0] == 2)
    assert (empty_ix[0] == 3)
    assert (empty_ix[1] == 7)
    assert (empty_pos[0] == 50)
    assert (empty_pos[1] == 62)

    hash2 = d12_rspos_hash_from_bits(bits)

    assert (hash1, hash2)

    free_game(game)
