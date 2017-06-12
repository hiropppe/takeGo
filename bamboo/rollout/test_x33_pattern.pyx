from libc.stdlib cimport malloc, free

from nose.tools import ok_, eq_

from bamboo.go.board cimport S_EMPTY, S_BLACK, S_WHITE
from bamboo.go.board cimport game_state_t, allocate_game, free_game
from bamboo.go.parseboard cimport parse
from bamboo.rollout.pattern cimport x33_bit

from bamboo.rollout.pattern import print_x33


def test_x33_bit():
    cdef unsigned long long pat
    cdef game_state_t *game = allocate_game()
    (moves, pure_moves) = parse(game,
                             "B B B . . . .|"
                             "B a W . . . .|"
                             "B B . . . . .|"
                             ". . . . . . .|"
                             ". . . . . . .|"
                             ". . . . . . .|")

    pat = x33_bit(game, moves['a'], S_WHITE)

    print_x33(pat)

    free_game(game)
