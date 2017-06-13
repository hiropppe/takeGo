import numpy as np

cimport numpy as np

from libc.stdlib cimport malloc, free

from nose.tools import ok_, eq_

from bamboo.go.board cimport S_EMPTY, S_BLACK, S_WHITE
from bamboo.go.board cimport game_state_t, allocate_game, free_game
from bamboo.go.parseboard cimport parse
from bamboo.rollout.pattern cimport initialize_hash, x33_bits, x33_hash

from bamboo.rollout.pattern import print_x33


def test_x33_bit():
    cdef unsigned long long pat
    cdef game_state_t *game = allocate_game()
    (moves, pure_moves) = parse(game,
                             ". B . B . . .|"
                             ". B W . . . B|"
                             ". B W . . B W|"
                             ". a B W W W W|"
                             "B W B W W B B|"
                             "B W W W . W B|"
                             ". . . . W B .|")

    pat = x33_bits(game, moves['a'], S_BLACK)

    print pat
    print_x33(pat)

    free_game(game)
