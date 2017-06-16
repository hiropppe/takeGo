import numpy as np
cimport numpy as np

from operator import itemgetter

from libc.stdlib cimport malloc, free

from nose.tools import ok_, eq_

from bamboo.go.board cimport S_EMPTY, S_BLACK, S_WHITE
from bamboo.go.board cimport FLIP_COLOR
from bamboo.go.board cimport game_state_t, board_size, pure_board_size, pure_board_max
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

    """
    Pattern 0:
        +++    000
        +o+    0 0
        ++B    003
    """
    put_x33_hash(0b1000000000000001110, 0)
    put_x33_hash(0b10000000000000001101, 0)
    put_x33_hash(0b10000000000000011000010, 0)
    put_x33_hash(0b100000000000000011000001, 0)
    put_x33_hash(0b10000000000000011000000000010, 0)
    put_x33_hash(0b100000000000000011000000000001, 0)
    put_x33_hash(0b100000000000000110000000000000010, 0)
    put_x33_hash(0b1000000000000000110000000000000001, 0)

    """
    Pattern 1:
        +++    000
        +o+    0 0
        +B+    030
    """
    put_x33_hash(0b100000000000000110010, 1)
    put_x33_hash(0b1000000000000000110001, 1)
    put_x33_hash(0b1000000000000001100000010, 1)
    put_x33_hash(0b10000000000000001100000001, 1)
    put_x33_hash(0b100000000000000110000000010, 1)
    put_x33_hash(0b1000000000000000110000000001, 1)
    put_x33_hash(0b1000000000000001100000000000010, 1)
    put_x33_hash(0b10000000000000001100000000000001, 1)

    """
    Pattern 2:
        +++    000
        +X+    0 0
        ++B    003
    """
    put_x33_hash(0b1000000000000001101, 2)
    put_x33_hash(0b10000000000000001110, 2)
    put_x33_hash(0b10000000000000011000001, 2)
    put_x33_hash(0b100000000000000011000010, 2)
    put_x33_hash(0b10000000000000011000000000001, 2)
    put_x33_hash(0b100000000000000011000000000010, 2)
    put_x33_hash(0b100000000000000110000000000000001, 2)
    put_x33_hash(0b1000000000000000110000000000000010, 2)

    """
    Pattern 3:
        +++    000
        +x+    0 0
        +B+    030
    """
    put_x33_hash(0b100000000000000110001, 3)
    put_x33_hash(0b1000000000000000110010, 3)
    put_x33_hash(0b1000000000000001100000001, 3)
    put_x33_hash(0b10000000000000001100000010, 3)
    put_x33_hash(0b100000000000000110000000001, 3)
    put_x33_hash(0b1000000000000000110000000010, 3)
    put_x33_hash(0b1000000000000001100000000000001, 3)
    put_x33_hash(0b10000000000000001100000000000010, 3)

    """
    Pattern 4:
        +++    000
        Wo+    3 0
        +B+    030    
    """
    put_x33_hash(0b1001000000000001100110001, 4)
    put_x33_hash(0b1001000000000001100110010, 4)
    put_x33_hash(0b10000100000000001100110001, 4)
    put_x33_hash(0b10000100000000001100110010, 4)
    put_x33_hash(0b100001000000000110000110001, 4)
    put_x33_hash(0b100001000000000110000110010, 4)
    put_x33_hash(0b1000000100000000110000110001, 4)
    put_x33_hash(0b1000000100000000110000110010, 4)
    put_x33_hash(0b1000010000000001100001100000001, 4)
    put_x33_hash(0b1000010000000001100001100000010, 4)
    put_x33_hash(0b1001000000000001100110000000001, 4)
    put_x33_hash(0b1001000000000001100110000000010, 4)
    put_x33_hash(0b10000001000000001100001100000001, 4)
    put_x33_hash(0b10000001000000001100001100000010, 4)
    put_x33_hash(0b10000100000000001100110000000001, 4)
    put_x33_hash(0b10000100000000001100110000000010, 4)

    """
    Pattern 5:
        ++B    003
        +o+    0 0
        +B+    030
    """
    put_x33_hash(0b1010000000000001111000010, 5)
    put_x33_hash(0b10100000000000001111000001, 5)
    put_x33_hash(0b100000001000000110000001110, 5)
    put_x33_hash(0b1000000010000000110000001101, 5)
    put_x33_hash(0b10000000100000011000000110010, 5)
    put_x33_hash(0b10100000000000011110000000010, 5)
    put_x33_hash(0b100000001000000011000000110001, 5)
    put_x33_hash(0b101000000000000011110000000001, 5)
    put_x33_hash(0b1000000000001001100000000001110, 5)
    put_x33_hash(0b1000000010000001100000011000010, 5)
    put_x33_hash(0b10000000000010001100000000001101, 5)
    put_x33_hash(0b10000000100000001100000011000001, 5)
    put_x33_hash(0b100000000000100110000000000110010, 5)
    put_x33_hash(0b100000001000000110000001100000010, 5)
    put_x33_hash(0b1000000000001000110000000000110001, 5)
    put_x33_hash(0b1000000010000000110000001100000001, 5)

    """
    Pattern 6:
        +++    000
        Wx+    3 0
        WB+    330
    """
    put_x33_hash(0b1001001000000001100111110, 6)
    put_x33_hash(0b1001010000000001100111101, 6)
    put_x33_hash(0b10000101000000001100111110, 6)
    put_x33_hash(0b10000110000000001100111101, 6)
    put_x33_hash(0b100011000000000110011110010, 6)
    put_x33_hash(0b100101000000000110011110001, 6)
    put_x33_hash(0b1000010100000000110011110010, 6)
    put_x33_hash(0b1000100100000000110011110001, 6)
    put_x33_hash(0b1010010000000001111001100000010, 6)
    put_x33_hash(0b1100010000000001111001100000001, 6)
    put_x33_hash(0b10010001000000001111001100000010, 6)
    put_x33_hash(0b10100001000000001111001100000001, 6)
    put_x33_hash(0b101001000000000111100110000000010, 6)
    put_x33_hash(0b110000100000000111100110000000010, 6)
    put_x33_hash(0b1001001000000000111100110000000001, 6)
    put_x33_hash(0b1010000100000000111100110000000001, 6)

    """
    Pattern 7:
        +WB    033
        +x+    0 0
        +B+    030
    """
    put_x33_hash(0b110000001000000111100001101, 7)
    put_x33_hash(0b110100000000000111111000010, 7)
    put_x33_hash(0b1001000010000000111100001110, 7)
    put_x33_hash(0b1001010000000000111111000001, 7)
    put_x33_hash(0b10110000000000011111100000001, 7)
    put_x33_hash(0b101001000000000011111100000010, 7)
    put_x33_hash(0b1000000001001001100000000111101, 7)
    put_x33_hash(0b1000000011000001100000011110001, 7)
    put_x33_hash(0b1100000001000001111000000110010, 7)
    put_x33_hash(0b10000000000110001100000000111110, 7)
    put_x33_hash(0b10000000100100001100000011110010, 7)
    put_x33_hash(0b10010000000100001111000000110001, 7)
    put_x33_hash(0b100001001000000110000111100000001, 7)
    put_x33_hash(0b110000000000100111100000000110001, 7)
    put_x33_hash(0b1000000110000000110000111100000010, 7)
    put_x33_hash(0b1001000000001000111100000000110010, 7)

    """
    Pattern 8:
        WB+    330
        +x+    0 0
        B++    300
    """
    put_x33_hash(0b1010010000000001111001101, 8)
    put_x33_hash(0b10100001000000001111001110, 8)
    put_x33_hash(0b100100001000000110011001101, 8)
    put_x33_hash(0b1000010010000000110011001110, 8)
    put_x33_hash(0b10000000110000011000000111101, 8)
    put_x33_hash(0b100000001001000011000000111110, 8)
    put_x33_hash(0b1100000000001001111000000001101, 8)
    put_x33_hash(0b10010000000010001111000000001110, 8)
    put_x33_hash(0b100000000100100110000000011110001, 8)
    put_x33_hash(0b100100001000000110011001100000001, 8)
    put_x33_hash(0b100101000000000110011110000000010, 8)
    put_x33_hash(0b110000000100000111100000011000010, 8)
    put_x33_hash(0b1000000000011000110000000011110010, 8)
    put_x33_hash(0b1000010010000000110011001100000010, 8)
    put_x33_hash(0b1000010100000000110011110000000001, 8)
    put_x33_hash(0b1001000000010000111100000011000001, 8)

    """
    Pattern 9:
        +++    000
        +xW    0 3
        ++W    003 
    """
    put_x33_hash(0b101000000000000111110, 9)
    put_x33_hash(0b1010000000000000111101, 9)
    put_x33_hash(0b10100000000000011110010, 9)
    put_x33_hash(0b101000000000000011110001, 9)
    put_x33_hash(0b1000001000000001100001110, 9)
    put_x33_hash(0b10000010000000001100001101, 9)
    put_x33_hash(0b100010000000000110011000010, 9)
    put_x33_hash(0b1000100000000000110011000001, 9)
    put_x33_hash(0b10001000000000011001100000010, 9)
    put_x33_hash(0b100010000000000011001100000001, 9)
    put_x33_hash(0b1010000000000001111000000000010, 9)
    put_x33_hash(0b10100000000000001111000000000001, 9)
    put_x33_hash(0b100000100000000110000110000000010, 9)
    put_x33_hash(0b101000000000000111100000000000010, 9)
    put_x33_hash(0b1000001000000000110000110000000001, 9)
    put_x33_hash(0b1010000000000000111100000000000001, 9)

    """
    Pattern 10:
        ++W    003
        +x+    0 0
        ++W    003 
    """
    put_x33_hash(0b10010000000000011001101, 10)
    put_x33_hash(0b10010000000000011001110, 10)
    put_x33_hash(0b100001000000000011001101, 10)
    put_x33_hash(0b100001000000000011001110, 10)
    put_x33_hash(0b10000000010000011000000001101, 10)
    put_x33_hash(0b10000000010000011000000001110, 10)
    put_x33_hash(0b100000000001000011000000001101, 10)
    put_x33_hash(0b100000000001000011000000001110, 10)
    put_x33_hash(0b100000000100000110000000011000001, 10)
    put_x33_hash(0b100000000100000110000000011000010, 10)
    put_x33_hash(0b100100000000000110011000000000001, 10)
    put_x33_hash(0b100100000000000110011000000000010, 10)
    put_x33_hash(0b1000000000010000110000000011000001, 10)
    put_x33_hash(0b1000000000010000110000000011000010, 10)
    put_x33_hash(0b1000010000000000110011000000000001, 10)
    put_x33_hash(0b1000010000000000110011000000000010, 10)


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

    # put B[a]
    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    feature.update(game)
    # updated around move['a'] 
    eq_(white.tensor[pure_moves['a']-pure_board_size-1][NON_RESPONSE_PAT], feature.x33_start) 
    eq_(white.tensor[pure_moves['a']-pure_board_size][NON_RESPONSE_PAT], feature.x33_start+1) 
    eq_(white.tensor[pure_moves['a']-pure_board_size+1][NON_RESPONSE_PAT], feature.x33_start) 
    eq_(white.tensor[pure_moves['a']-1][NON_RESPONSE_PAT], feature.x33_start+1) 
    eq_(white.tensor[pure_moves['a']+1][NON_RESPONSE_PAT], feature.x33_start+1) 
    eq_(white.tensor[pure_moves['a']+pure_board_size-1][NON_RESPONSE_PAT], feature.x33_start) 
    eq_(white.tensor[pure_moves['a']+pure_board_size][NON_RESPONSE_PAT], feature.x33_start+1) 
    eq_(white.tensor[pure_moves['a']+pure_board_size+1][NON_RESPONSE_PAT], feature.x33_start) 
    eq_(number_of_active_positions(white, NON_RESPONSE_PAT), 8)

    # put W[b]
    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    feature.update(game)
    # updated around move['a'] 
    eq_(black.tensor[pure_moves['a']-pure_board_size-1][NON_RESPONSE_PAT], feature.x33_start + 2) 
    eq_(black.tensor[pure_moves['a']-pure_board_size][NON_RESPONSE_PAT], feature.x33_start + 3) 
    eq_(black.tensor[pure_moves['a']-pure_board_size+1][NON_RESPONSE_PAT], feature.x33_start + 2) 
    eq_(black.tensor[pure_moves['a']-1][NON_RESPONSE_PAT], feature.x33_start + 3) 
    eq_(black.tensor[pure_moves['a']+1][NON_RESPONSE_PAT], feature.x33_start + 3) 
    eq_(black.tensor[pure_moves['a']+pure_board_size-1][NON_RESPONSE_PAT], feature.x33_start + 2) 
    eq_(black.tensor[pure_moves['a']+pure_board_size][NON_RESPONSE_PAT], feature.x33_start + 3) 
    eq_(black.tensor[pure_moves['a']+pure_board_size+1][NON_RESPONSE_PAT], feature.x33_start + 2) 
    # updated around move['b']
    eq_(black.tensor[pure_moves['b']-pure_board_size-1][NON_RESPONSE_PAT], feature.x33_start) 
    eq_(black.tensor[pure_moves['b']-pure_board_size][NON_RESPONSE_PAT], feature.x33_start+1) 
    eq_(black.tensor[pure_moves['b']-pure_board_size+1][NON_RESPONSE_PAT], feature.x33_start) 
    eq_(black.tensor[pure_moves['b']-1][NON_RESPONSE_PAT], feature.x33_start+1) 
    eq_(black.tensor[pure_moves['b']+1][NON_RESPONSE_PAT], feature.x33_start+1) 
    eq_(black.tensor[pure_moves['b']+pure_board_size-1][NON_RESPONSE_PAT], feature.x33_start) 
    eq_(black.tensor[pure_moves['b']+pure_board_size][NON_RESPONSE_PAT], feature.x33_start+1) 
    eq_(black.tensor[pure_moves['b']+pure_board_size+1][NON_RESPONSE_PAT], feature.x33_start) 
    eq_(number_of_active_positions(black, NON_RESPONSE_PAT), 16)

    # put B[c]
    put_stone(game, moves['c'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    feature.update(game)
    # updated around move['b']
    eq_(white.tensor[pure_moves['b']-pure_board_size-1][NON_RESPONSE_PAT], feature.x33_start + 2) 
    eq_(white.tensor[pure_moves['b']-pure_board_size][NON_RESPONSE_PAT], feature.x33_start + 3) 
    eq_(white.tensor[pure_moves['b']-pure_board_size+1][NON_RESPONSE_PAT], feature.x33_start + 2) 
    eq_(white.tensor[pure_moves['b']-1][NON_RESPONSE_PAT], feature.x33_start + 3) 
    eq_(white.tensor[pure_moves['b']+1][NON_RESPONSE_PAT], feature.x33_start + 4) 
    eq_(white.tensor[pure_moves['b']+pure_board_size-1][NON_RESPONSE_PAT], feature.x33_start + 2) 
    eq_(white.tensor[pure_moves['b']+pure_board_size][NON_RESPONSE_PAT], feature.x33_start + 4) 
    eq_(white.tensor[pure_moves['b']+pure_board_size+1][NON_RESPONSE_PAT], -1) 
    # updated around move['c']
    eq_(white.tensor[pure_moves['c']-pure_board_size-1][NON_RESPONSE_PAT], -1) 
    eq_(white.tensor[pure_moves['c']-pure_board_size][NON_RESPONSE_PAT], feature.x33_start + 4) 
    eq_(white.tensor[pure_moves['c']-pure_board_size+1][NON_RESPONSE_PAT], feature.x33_start) 
    eq_(white.tensor[pure_moves['c']-1][NON_RESPONSE_PAT], feature.x33_start + 4) 
    eq_(white.tensor[pure_moves['c']+1][NON_RESPONSE_PAT], feature.x33_start + 1) 
    eq_(white.tensor[pure_moves['c']+pure_board_size-1][NON_RESPONSE_PAT], feature.x33_start + 5) 
    eq_(white.tensor[pure_moves['c']+pure_board_size][NON_RESPONSE_PAT], feature.x33_start + 5) 
    eq_(white.tensor[pure_moves['c']+pure_board_size+1][NON_RESPONSE_PAT], feature.x33_start) 
    eq_(number_of_active_positions(white, NON_RESPONSE_PAT), 18)

    # put W[d]
    put_stone(game, moves['d'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    feature.update(game)
    # updated around move['c']
    eq_(black.tensor[pure_moves['c']-pure_board_size-1][NON_RESPONSE_PAT], -1) 
    eq_(black.tensor[pure_moves['c']-pure_board_size][NON_RESPONSE_PAT], feature.x33_start + 6) 
    eq_(black.tensor[pure_moves['c']-pure_board_size+1][NON_RESPONSE_PAT], feature.x33_start + 2) 
    eq_(black.tensor[pure_moves['c']-1][NON_RESPONSE_PAT], -1) 
    eq_(black.tensor[pure_moves['c']+1][NON_RESPONSE_PAT], feature.x33_start + 3) 
    eq_(black.tensor[pure_moves['c']+pure_board_size-1][NON_RESPONSE_PAT], feature.x33_start + 7) 
    eq_(black.tensor[pure_moves['c']+pure_board_size][NON_RESPONSE_PAT], feature.x33_start + 8) 
    eq_(black.tensor[pure_moves['c']+pure_board_size+1][NON_RESPONSE_PAT], feature.x33_start + 2) 
    # updated around move['d']
    eq_(black.tensor[pure_moves['b']-pure_board_size-1][NON_RESPONSE_PAT], feature.x33_start) 
    eq_(black.tensor[pure_moves['b']-pure_board_size][NON_RESPONSE_PAT], feature.x33_start + 1) 
    eq_(black.tensor[pure_moves['b']-pure_board_size+1][NON_RESPONSE_PAT], feature.x33_start) 
    eq_(black.tensor[pure_moves['b']-1][NON_RESPONSE_PAT], feature.x33_start + 9) 
    eq_(black.tensor[pure_moves['b']+1][NON_RESPONSE_PAT], feature.x33_start + 6) 
    eq_(black.tensor[pure_moves['d']-1][NON_RESPONSE_PAT], feature.x33_start + 9) 
    eq_(black.tensor[pure_moves['d']+pure_board_size-1][NON_RESPONSE_PAT], feature.x33_start + 10) 
    eq_(black.tensor[pure_moves['d']+pure_board_size][NON_RESPONSE_PAT], feature.x33_start + 7) 
    eq_(black.tensor[pure_moves['d']+pure_board_size+1][NON_RESPONSE_PAT], feature.x33_start + 8) 
    eq_(number_of_active_positions(black, NON_RESPONSE_PAT), 17)

    print_board(game)

    free_game(game)


cdef int number_of_active_positions(rollout_feature_t *feature, int feature_id):
    cdef int i
    cdef int n_active = 0
    for i in range(pure_board_max):
        if feature.tensor[i][feature_id] >= 0:
            n_active += 1
    return n_active
