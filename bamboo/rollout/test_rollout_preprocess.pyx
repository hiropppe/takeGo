import numpy as np
cimport numpy as np

from operator import itemgetter

from libc.stdlib cimport malloc, free

from nose.tools import ok_, eq_

from bamboo.go.board cimport S_EMPTY, S_BLACK, S_WHITE
from bamboo.go.board cimport FLIP_COLOR
from bamboo.go.board cimport game_state_t
from bamboo.go.board cimport allocate_game, free_game, put_stone
from bamboo.go.printer cimport print_board
from bamboo.go.parseboard cimport parse

from bamboo.rollout.preprocess cimport NON_RESPONSE_PAT
from bamboo.rollout.preprocess cimport rollout_feature_t
from bamboo.rollout.preprocess cimport RolloutFeature
from bamboo.rollout.pattern cimport initialize_hash, put_x33_hash
from bamboo.rollout.pattern import print_x33


cdef int nakade_size = 0
cdef int x33_size = 0
cdef int d12_size = 0

def setup():
    global nakade_size, x33_size, d12_size

    initialize_hash()

    put_x33_hash(0b0100000000000000110000000000000010, 0)
    put_x33_hash(0b0001000000000000001100000000000010, 0)
    put_x33_hash(0b0000010000000000000011000000000010, 0)
    put_x33_hash(0b0000000100000000000000110000000010, 0)
    put_x33_hash(0b0000000001000000000000001100000010, 0)
    put_x33_hash(0b0000000000010000000000000011000010, 0)
    put_x33_hash(0b0000000000000100000000000000110010, 0)
    put_x33_hash(0b0000000000000001000000000000001110, 0)


def teardown():
    pass


def test_update_0():
    cdef game_state_t *game = allocate_game()
    cdef RolloutFeature feature = RolloutFeature(nakade_size, x33_size, d12_size)
    cdef rollout_feature_t *black = &feature.feature_planes[<int>S_BLACK]
    cdef rollout_feature_t *white = &feature.feature_planes[<int>S_WHITE]

    (moves, pure_moves) = parse(game,
                              ". . . . . . . . .|"
                              ". . . . . w o t .|"
                              ". . . . . i n r .|"
                              ". . . . . b e s .|"
                              ". . . . . d c . .|"
                              ". . . . . f g u .|"
                              ". . . v h a j l .|"
                              ". . . . q k m p .|"
                              ". . . . . . . . .|")

    game.current_color = S_BLACK

    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)

    feature.update(game)

    ok_(white.tensor[pure_moves['a']-9][NON_RESPONSE_PAT] == feature.x33_start) 

    print_board(game)

    free_game(game)
