import numpy as np
cimport numpy as np

from operator import itemgetter

from libc.stdlib cimport malloc, free

from nose.tools import ok_, eq_

from bamboo.go.board cimport BOARD_MAX, S_EMPTY, S_BLACK, S_WHITE, PASS
from bamboo.go.board cimport FLIP_COLOR
from bamboo.go.board cimport game_state_t, rollout_feature_t, board_size, pure_board_size, pure_board_max
from bamboo.go.board cimport allocate_game, free_game, put_stone
from bamboo.go.printer cimport print_board
from bamboo.go.parseboard cimport parse

from bamboo.rollout.preprocess cimport F_RESPONSE, F_SAVE_ATARI, F_NEIGHBOR, F_NAKADE, F_RESPONSE_PAT, F_NON_RESPONSE_PAT
from bamboo.rollout.preprocess cimport response_start, save_atari_start, neighbor_start, nakade_start, d12_start, x33_start
from bamboo.rollout.preprocess cimport initialize_const, initialize_planes, initialize_probs, update_planes, update_planes_all, memorize_updated, choice_rollout_move 
from bamboo.rollout.pattern cimport initialize_rands, put_x33_hash, put_d12_hash
from bamboo.rollout.pattern import print_x33


cdef int nakade_size = 0
cdef int x33_size = 100
cdef int d12_size = 100

def setup():
    initialize_const(nakade_size, x33_size, d12_size)
    initialize_rands()

    put_12diamond_test_patterns()
    put_3x3_test_patterns()


def teardown():
    pass


def put_12diamond_test_patterns():
    """
    Pattern 0:
      *       0
     +++     000
    *+B+*   00300
     +++     000
      *       0
    """
    put_d12_hash(0b0100000000000000000000000011000000000001, 0)
    put_d12_hash(0b0100000000000000000000000011000000010000, 0)
    put_d12_hash(0b0100000000000000000000000011000010000000, 0)
    put_d12_hash(0b0100000000000000000000000011100000000000, 0)
    put_d12_hash(0b1000000000000000000000000011000000000001, 0)
    put_d12_hash(0b1000000000000000000000000011000000010000, 0)
    put_d12_hash(0b1000000000000000000000000011000010000000, 0)
    put_d12_hash(0b1000000000000000000000000011100000000000, 0)

    """
    Pattern 1:
      +       0
     *+*     000
    ++B++   00300
     *+*     000
      +       0
    """
    put_d12_hash(0b0100000000000000000000000011000000000010, 1)
    put_d12_hash(0b0100000000000000000000000011000000001000, 1)
    put_d12_hash(0b0100000000000000000000000011000100000000, 1)
    put_d12_hash(0b0100000000000000000000000011010000000000, 1)
    put_d12_hash(0b1000000000000000000000000011000000000010, 1)
    put_d12_hash(0b1000000000000000000000000011000000001000, 1)
    put_d12_hash(0b1000000000000000000000000011000100000000, 1)
    put_d12_hash(0b1000000000000000000000000011010000000000, 1)

    """
    Pattern 2:
      +       0
     +*+     000
    +*B*+   00300
     +*+     000
      +       0
    """
    put_d12_hash(0b0100000000000000000000000011000000000100, 2)
    put_d12_hash(0b0100000000000000000000000011000000100000, 2)
    put_d12_hash(0b0100000000000000000000000011000001000000, 2)
    put_d12_hash(0b0100000000000000000000000011001000000000, 2)
    put_d12_hash(0b1000000000000000000000000011000000000100, 2)
    put_d12_hash(0b1000000000000000000000000011000000100000, 2)
    put_d12_hash(0b1000000000000000000000000011000001000000, 2)
    put_d12_hash(0b1000000000000000000000000011001000000000, 2)

    """
    Pattern 3-13:
      +       0
     W++     300
    ++B++   00300
     +++     000
      *       0
    """
    put_d12_hash(0b10000100000000000000000000110011100000000000, 3)
    put_d12_hash(0b10000100000000000000000000110011010000000000, 4)
    put_d12_hash(0b10000100000000000000000000110011001000000000, 5)
    put_d12_hash(0b10000100000000000000000000110011000100000000, 6)
    put_d12_hash(0b10000100000000000000000000110011000010000000, 7)
    put_d12_hash(0b10000100000000000000000000110011000001000000, 8)
    put_d12_hash(0b10000100000000000000000000110011000000100000, 9)
    put_d12_hash(0b10000100000000000000000000110011000000010000, 10)
    put_d12_hash(0b10000100000000000000000000110011000000001000, 11)
    put_d12_hash(0b10000100000000000000000000110011000000000100, 12)
    put_d12_hash(0b10000100000000000000000000110011000000000001, 13)

    """
    Pattern 14-22:
      +       0
     +W+     030
    ++WB+   00330
     ++*     000
      B       3
    """
    put_d12_hash(0b0100000000010000001000001011000000001100000011000011010000000000, 14)
    put_d12_hash(0b0100000000010000001000001011000000001100000011000011001000000000, 15)
    put_d12_hash(0b0100000000010000001000001011000000001100000011000011000100000000, 16)
    put_d12_hash(0b0100000000010000001000001011000000001100000011000011000010000000, 17)
    put_d12_hash(0b0100000000010000001000001011000000001100000011000011000000100000, 18)
    put_d12_hash(0b0100000000010000001000001011000000001100000011000011000000010000, 19)
    put_d12_hash(0b0100000000010000001000001011000000001100000011000011000000001000, 20)
    put_d12_hash(0b0100000000010000001000001011000000001100000011000011000000000010, 21)
    put_d12_hash(0b0100000000010000001000001011000000001100000011000011000000000001, 22)

    """
    Pattern 23-31:
      +       0
     +++     000
    +WB++   03300
     WB+     330
      +       0
    """
    put_d12_hash(0b011000001000000000000100001111000011000000000011100000000000, 23)
    put_d12_hash(0b011000001000000000000100001111000011000000000011010000000000, 24)
    put_d12_hash(0b011000001000000000000100001111000011000000000011000010000000, 25)
    put_d12_hash(0b011000001000000000000100001111000011000000000011000001000000, 26)
    put_d12_hash(0b011000001000000000000100001111000011000000000011000000010000, 27)
    put_d12_hash(0b011000001000000000000100001111000011000000000011000000001000, 28)
    put_d12_hash(0b011000001000000000000100001111000011000000000011000000000100, 29)
    put_d12_hash(0b011000001000000000000100001111000011000000000011000000000010, 30)
    put_d12_hash(0b011000001000000000000100001111000011000000000011000000000001, 31)


def put_3x3_test_patterns():
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


def test_update_save_atari():
    cdef game_state_t *game = allocate_game()
    cdef rollout_feature_t *black

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . . a . . .|"
                              ". . W B W . .|"
                              ". . . W . . .|"
                              ". . . . . . .|"
                              ". . . . . . .|")

    game.current_color = S_BLACK

    update_planes(game)

    black = &game.rollout_feature_planes[<int>S_BLACK]

    eq_(black.tensor[F_SAVE_ATARI][pure_moves['a']], save_atari_start) 
    eq_(number_of_active_positions(black, F_SAVE_ATARI), 1)

    free_game(game)


def test_update_save_atari_connect_string():
    cdef game_state_t *game = allocate_game()
    cdef rollout_feature_t *white

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". . . W . . .|"
                              ". . B a B . .|"
                              ". . B W B . .|"
                              ". . . B . . .|"
                              ". . . . . . .|"
                              ". . . . . . .|")

    game.current_color = S_WHITE

    update_planes(game)

    white = &game.rollout_feature_planes[<int>S_WHITE]

    eq_(white.tensor[F_SAVE_ATARI][pure_moves['a']], save_atari_start) 
    eq_(number_of_active_positions(white, F_SAVE_ATARI), 1)

    free_game(game)


def test_update_save_atari_not_escape():
    cdef game_state_t *game = allocate_game()
    cdef rollout_feature_t *black

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . W a W . .|"
                              ". . W B W . .|"
                              ". . . W . . .|"
                              ". . . . . . .|"
                              ". . . . . . .|")

    game.current_color = S_BLACK

    update_planes(game)

    black = &game.rollout_feature_planes[<int>S_BLACK]

    eq_(black.tensor[F_SAVE_ATARI][pure_moves['a']], -1) 
    eq_(number_of_active_positions(black, F_SAVE_ATARI), 0)

    free_game(game)


def test_update_save_atari_not_escape_on_edge():
    cdef game_state_t *game = allocate_game()
    cdef rollout_feature_t *white

    (moves, pure_moves) = parse(game,
                              "W B . . . . .|"
                              "a B . . . . .|"
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . . . . . .|")

    game.current_color = S_WHITE

    update_planes(game)

    white = &game.rollout_feature_planes[<int>S_WHITE]

    eq_(white.tensor[F_SAVE_ATARI][pure_moves['a']], -1) 
    eq_(number_of_active_positions(white, F_SAVE_ATARI), 0)

    free_game(game)


def test_update_neighbor_0():
    cdef game_state_t *game = allocate_game()
    cdef rollout_feature_t *black
    cdef rollout_feature_t *white

    (moves, pure_moves) = parse(game,
                              ". . . . .|"
                              ". . . . .|"
                              ". . a b c|"
                              ". . . . .|"
                              ". . . . .|")

    game.current_color = S_BLACK

    black = &game.rollout_feature_planes[<int>S_BLACK]
    white = &game.rollout_feature_planes[<int>S_WHITE]

    # put B[a]
    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)
    # updated around move['a'] 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['a']-pure_board_size-1], neighbor_start) 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['a']-pure_board_size], neighbor_start+1) 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['a']-pure_board_size+1], neighbor_start+2) 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['a']-1], neighbor_start+3) 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['a']+1], neighbor_start+4) 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['a']+pure_board_size-1], neighbor_start+5) 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['a']+pure_board_size], neighbor_start+6) 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['a']+pure_board_size+1], neighbor_start+7) 
    eq_(number_of_active_positions(white, F_NEIGHBOR), 8)

    # put W[b]
    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)
    # updated around move['b'] 
    eq_(black.tensor[F_NEIGHBOR][pure_moves['b']-pure_board_size-1], neighbor_start) 
    eq_(black.tensor[F_NEIGHBOR][pure_moves['b']-pure_board_size], neighbor_start+1) 
    eq_(black.tensor[F_NEIGHBOR][pure_moves['b']-pure_board_size+1], neighbor_start+2) 
    eq_(black.tensor[F_NEIGHBOR][pure_moves['b']-1], -1) 
    eq_(black.tensor[F_NEIGHBOR][pure_moves['b']+1], neighbor_start+4) 
    eq_(black.tensor[F_NEIGHBOR][pure_moves['b']+pure_board_size-1], neighbor_start+5) 
    eq_(black.tensor[F_NEIGHBOR][pure_moves['b']+pure_board_size], neighbor_start+6) 
    eq_(black.tensor[F_NEIGHBOR][pure_moves['b']+pure_board_size+1], neighbor_start+7) 
    eq_(number_of_active_positions(black, F_NEIGHBOR), 7)

    # put B[c]
    put_stone(game, moves['c'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)
    # updated around move['c'] 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['c']-pure_board_size-1], neighbor_start) 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['c']-pure_board_size], neighbor_start+1) 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['c']-pure_board_size+1], -1) 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['c']-1], -1) 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['c']+1], -1) 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['c']+pure_board_size-1], neighbor_start+5) 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['c']+pure_board_size], neighbor_start+6) 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['c']+pure_board_size+1], -1) 
    eq_(number_of_active_positions(white, F_NEIGHBOR), 4)

    # PASS W
    put_stone(game, PASS, game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)
    # cleared around move['b'] 
    eq_(number_of_active_positions(black, F_NEIGHBOR), 0)

    # PASS B
    put_stone(game, PASS, game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)
    # cleared around move['c'] 
    eq_(number_of_active_positions(white, F_NEIGHBOR), 0)


def test_update_12diamond_0():
    cdef game_state_t *game = allocate_game()
    cdef rollout_feature_t *black
    cdef rollout_feature_t *white

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

    black = &game.rollout_feature_planes[<int>S_BLACK]
    white = &game.rollout_feature_planes[<int>S_WHITE]

    # put B[a]
    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)
    # updated around move['a'] 
    eq_(white.tensor[F_RESPONSE][pure_moves['a']-2*pure_board_size], response_start) 
    eq_(white.tensor[F_RESPONSE][pure_moves['a']-pure_board_size-1], response_start) 
    eq_(white.tensor[F_RESPONSE][pure_moves['a']-pure_board_size], response_start) 
    eq_(white.tensor[F_RESPONSE][pure_moves['a']-pure_board_size+1], response_start) 
    eq_(white.tensor[F_RESPONSE][pure_moves['a']-2], response_start) 
    eq_(white.tensor[F_RESPONSE][pure_moves['a']-1], response_start) 
    eq_(white.tensor[F_RESPONSE][pure_moves['a']+1], response_start) 
    eq_(white.tensor[F_RESPONSE][pure_moves['a']+2], response_start) 
    eq_(white.tensor[F_RESPONSE][pure_moves['a']+pure_board_size-1], response_start) 
    eq_(white.tensor[F_RESPONSE][pure_moves['a']+pure_board_size], response_start) 
    eq_(white.tensor[F_RESPONSE][pure_moves['a']+pure_board_size+1], response_start) 
    eq_(white.tensor[F_RESPONSE][pure_moves['a']+2*pure_board_size], response_start) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['a']-2*pure_board_size], d12_start) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['a']-pure_board_size-1], d12_start+1) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['a']-pure_board_size], d12_start+2) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['a']-pure_board_size+1], d12_start+1) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['a']-2], d12_start) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['a']-1], d12_start+2) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['a']+1], d12_start+2) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['a']+2], d12_start) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['a']+pure_board_size-1], d12_start+1) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['a']+pure_board_size], d12_start+2) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['a']+pure_board_size+1], d12_start+1) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['a']+2*pure_board_size], d12_start) 
    eq_(number_of_active_positions(white, F_RESPONSE), 12)
    eq_(number_of_active_positions(white, F_RESPONSE_PAT), 12)

    # put W[b]
    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)
    # updated around move['b']
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['b']-2*pure_board_size], d12_start) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['b']-pure_board_size-1], d12_start+1) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['b']-pure_board_size], d12_start+2) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['b']-pure_board_size+1], d12_start+1) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['b']-2], d12_start) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['b']-1], d12_start+2) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['b']+1], d12_start+2) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['b']+2], d12_start) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['b']+pure_board_size-1], d12_start+1) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['b']+pure_board_size], d12_start+2) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['b']+pure_board_size+1], d12_start+1) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['b']+2*pure_board_size], d12_start) 
    # no others
    eq_(number_of_active_positions(black, F_RESPONSE), 12)
    eq_(number_of_active_positions(black, F_RESPONSE_PAT), 12)

    # put B[b]
    put_stone(game, moves['c'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)
    # updated around move['c']
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['c']-2*pure_board_size], d12_start+13) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['c']-pure_board_size-1], -1) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['c']-pure_board_size], d12_start+12) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['c']-pure_board_size+1], d12_start+11) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['c']-2], d12_start+10) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['c']-1], d12_start+9) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['c']+1], d12_start+8) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['c']+2], d12_start+7) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['c']+pure_board_size-1], d12_start+6) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['c']+pure_board_size], d12_start+5) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['c']+pure_board_size+1], d12_start+4) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['c']+2*pure_board_size], d12_start+3) 
    # no others
    eq_(number_of_active_positions(white, F_RESPONSE), 11)
    eq_(number_of_active_positions(white, F_RESPONSE_PAT), 11)

    # put W[d]
    put_stone(game, moves['d'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)
    # updated around move['d']
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['d']-2*pure_board_size], d12_start+22) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['d']-pure_board_size-1], d12_start+21) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['d']-pure_board_size], -1) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['d']-pure_board_size+1], d12_start+20) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['d']-2], d12_start+19) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['d']-1], d12_start+18) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['d']+1], -1) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['d']+2], d12_start+17) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['d']+pure_board_size-1], d12_start+16) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['d']+pure_board_size], d12_start+15) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['d']+pure_board_size+1], d12_start+14) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['d']+2*pure_board_size], -1) 
    # no others, previous positions are cleared
    eq_(number_of_active_positions(black, F_RESPONSE), 9)
    eq_(number_of_active_positions(black, F_RESPONSE_PAT), 9)

    # put B[e]
    put_stone(game, moves['e'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)
    # updated around move['e']
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['e']-2*pure_board_size], d12_start+31) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['e']-pure_board_size-1], d12_start+30) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['e']-pure_board_size], d12_start+29) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['e']-pure_board_size+1], d12_start+28) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['e']-2], d12_start+27) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['e']-1], -1) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['e']+1], d12_start+26) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['e']+2], d12_start+25) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['e']+pure_board_size-1], -1) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['e']+pure_board_size], -1) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['e']+pure_board_size+1], d12_start+24) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['e']+2*pure_board_size], d12_start+23) 
    # no others, previous positions are cleared
    eq_(number_of_active_positions(white, F_RESPONSE), 9)
    eq_(number_of_active_positions(white, F_RESPONSE_PAT), 9)

    # print_board(game)

    free_game(game)


def test_update_12diamond_after_pass_0():
    cdef game_state_t *game = allocate_game()
    cdef rollout_feature_t *black
    cdef rollout_feature_t *white

    (moves, pure_moves) = parse(game,
                              ". . . . . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|"
                              ". . . . . W e . .|"
                              ". . . . . W B . .|"
                              ". . . . . f . . .|"
                              ". . . . . B . . .|"
                              ". . . . . . . . .|"
                              ". . . . . . . . .|")

    game.current_color = S_BLACK

    black = &game.rollout_feature_planes[<int>S_BLACK]
    white = &game.rollout_feature_planes[<int>S_WHITE]

    # put B[e]
    put_stone(game, moves['e'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    # put W[f]
    put_stone(game, moves['f'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    # B PASS
    put_stone(game, PASS, game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    eq_(number_of_active_positions(white, F_RESPONSE_PAT), 0)

    free_game(game)


def test_update_3x3_0():
    cdef game_state_t *game = allocate_game()
    cdef rollout_feature_t *black
    cdef rollout_feature_t *white

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

    black = &game.rollout_feature_planes[<int>S_BLACK]
    white = &game.rollout_feature_planes[<int>S_WHITE]

    # put B[a]
    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)
    # updated around move['a'] 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['a']-pure_board_size-1], x33_start) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['a']-pure_board_size], x33_start+1) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['a']-pure_board_size+1], x33_start) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['a']-1], x33_start+1) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['a']+1], x33_start+1) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['a']+pure_board_size-1], x33_start) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['a']+pure_board_size], x33_start+1) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['a']+pure_board_size+1], x33_start) 
    eq_(number_of_active_positions(white, F_NON_RESPONSE_PAT), 8)

    # put W[b]
    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)
    # updated around move['a'] 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['a']-pure_board_size-1], x33_start + 2) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['a']-pure_board_size], x33_start + 3) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['a']-pure_board_size+1], x33_start + 2) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['a']-1], x33_start + 3) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['a']+1], x33_start + 3) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['a']+pure_board_size-1], x33_start + 2) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['a']+pure_board_size], x33_start + 3) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['a']+pure_board_size+1], x33_start + 2) 
    # updated around move['b']
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['b']-pure_board_size-1], x33_start) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['b']-pure_board_size], x33_start+1) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['b']-pure_board_size+1], x33_start) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['b']-1], x33_start+1) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['b']+1], x33_start+1) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['b']+pure_board_size-1], x33_start) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['b']+pure_board_size], x33_start+1) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['b']+pure_board_size+1], x33_start) 
    eq_(number_of_active_positions(black, F_NON_RESPONSE_PAT), 16)

    # put B[c]
    put_stone(game, moves['c'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)
    # updated around move['b']
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['b']-pure_board_size-1], x33_start + 2) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['b']-pure_board_size], x33_start + 3) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['b']-pure_board_size+1], x33_start + 2) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['b']-1], x33_start + 3) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['b']+1], x33_start + 4) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['b']+pure_board_size-1], x33_start + 2) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['b']+pure_board_size], x33_start + 4) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['b']+pure_board_size+1], -1) 
    # updated around move['c']
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['c']-pure_board_size-1], -1) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['c']-pure_board_size], x33_start + 4) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['c']-pure_board_size+1], x33_start) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['c']-1], x33_start + 4) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['c']+1], x33_start + 1) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['c']+pure_board_size-1], x33_start + 5) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['c']+pure_board_size], x33_start + 5) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['c']+pure_board_size+1], x33_start) 
    eq_(number_of_active_positions(white, F_NON_RESPONSE_PAT), 18)

    # put W[d]
    put_stone(game, moves['d'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)
    # updated around move['c']
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['c']-pure_board_size-1], -1) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['c']-pure_board_size], x33_start + 6) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['c']-pure_board_size+1], x33_start + 2) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['c']-1], -1) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['c']+1], x33_start + 3) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['c']+pure_board_size-1], x33_start + 7) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['c']+pure_board_size], x33_start + 8) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['c']+pure_board_size+1], x33_start + 2) 
    # updated around move['d']
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['b']-pure_board_size-1], x33_start) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['b']-pure_board_size], x33_start + 1) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['b']-pure_board_size+1], x33_start) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['b']-1], x33_start + 9) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['b']+1], x33_start + 6) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['d']-1], x33_start + 9) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['d']+pure_board_size-1], x33_start + 10) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['d']+pure_board_size], x33_start + 7) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['d']+pure_board_size+1], x33_start + 8) 
    eq_(number_of_active_positions(black, F_NON_RESPONSE_PAT), 17)

    # print_board(game)

    free_game(game)


def test_update_all_save_atari():
    cdef game_state_t *game = allocate_game()
    cdef rollout_feature_t *black
    cdef rollout_feature_t *white

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . . . . . W|"
                              "W B W B W B W|"
                              "B W B W B W B|"
                              ". . . . . . .|"
                              ". . . . . . .|")

    black = &game.rollout_feature_planes[<int>S_BLACK]
    white = &game.rollout_feature_planes[<int>S_WHITE]

    game.current_color = S_BLACK
    update_planes_all(game)
    eq_(number_of_active_positions(black, F_SAVE_ATARI), 7)

    game.current_color = FLIP_COLOR(game.current_color)
    update_planes_all(game)
    eq_(number_of_active_positions(white, F_SAVE_ATARI), 6)


def test_update_all_neighbor():
    cdef game_state_t *game = allocate_game()
    cdef rollout_feature_t *black
    cdef rollout_feature_t *white

    (moves, pure_moves) = parse(game,
                              ". . . . .|"
                              ". . . . .|"
                              ". . a b c|"
                              ". . . . .|"
                              ". . . . .|")

    game.current_color = S_BLACK

    black = &game.rollout_feature_planes[<int>S_BLACK]
    white = &game.rollout_feature_planes[<int>S_WHITE]

    # put B[a]
    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes_all(game)
    # updated around move['a'] 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['a']-pure_board_size-1], neighbor_start) 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['a']-pure_board_size], neighbor_start+1) 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['a']-pure_board_size+1], neighbor_start+2) 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['a']-1], neighbor_start+3) 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['a']+1], neighbor_start+4) 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['a']+pure_board_size-1], neighbor_start+5) 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['a']+pure_board_size], neighbor_start+6) 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['a']+pure_board_size+1], neighbor_start+7) 
    eq_(number_of_active_positions(white, F_NEIGHBOR), 8)

    # put W[b]
    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes_all(game)
    # updated around move['b'] 
    eq_(black.tensor[F_NEIGHBOR][pure_moves['b']-pure_board_size-1], neighbor_start) 
    eq_(black.tensor[F_NEIGHBOR][pure_moves['b']-pure_board_size], neighbor_start+1) 
    eq_(black.tensor[F_NEIGHBOR][pure_moves['b']-pure_board_size+1], neighbor_start+2) 
    eq_(black.tensor[F_NEIGHBOR][pure_moves['b']-1], -1) 
    eq_(black.tensor[F_NEIGHBOR][pure_moves['b']+1], neighbor_start+4) 
    eq_(black.tensor[F_NEIGHBOR][pure_moves['b']+pure_board_size-1], neighbor_start+5) 
    eq_(black.tensor[F_NEIGHBOR][pure_moves['b']+pure_board_size], neighbor_start+6) 
    eq_(black.tensor[F_NEIGHBOR][pure_moves['b']+pure_board_size+1], neighbor_start+7) 
    eq_(number_of_active_positions(black, F_NEIGHBOR), 7)

    # put B[c]
    put_stone(game, moves['c'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes_all(game)
    # updated around move['c'] 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['c']-pure_board_size-1], neighbor_start) 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['c']-pure_board_size], neighbor_start+1) 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['c']-pure_board_size+1], -1) 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['c']-1], -1) 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['c']+1], -1) 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['c']+pure_board_size-1], neighbor_start+5) 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['c']+pure_board_size], neighbor_start+6) 
    eq_(white.tensor[F_NEIGHBOR][pure_moves['c']+pure_board_size+1], -1) 
    eq_(number_of_active_positions(white, F_NEIGHBOR), 4)

    # PASS W
    put_stone(game, PASS, game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes_all(game)
    # cleared around move['b'] 
    eq_(number_of_active_positions(black, F_NEIGHBOR), 0)

    # PASS B
    put_stone(game, PASS, game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes_all(game)
    # cleared around move['c'] 
    eq_(number_of_active_positions(white, F_NEIGHBOR), 0)


def test_update_all_12diamond():
    cdef game_state_t *game = allocate_game()
    cdef rollout_feature_t *black
    cdef rollout_feature_t *white

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

    black = &game.rollout_feature_planes[<int>S_BLACK]
    white = &game.rollout_feature_planes[<int>S_WHITE]

    # put B[a]
    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes_all(game)
    # updated around move['a'] 
    eq_(white.tensor[F_RESPONSE][pure_moves['a']-2*pure_board_size], response_start) 
    eq_(white.tensor[F_RESPONSE][pure_moves['a']-pure_board_size-1], response_start) 
    eq_(white.tensor[F_RESPONSE][pure_moves['a']-pure_board_size], response_start) 
    eq_(white.tensor[F_RESPONSE][pure_moves['a']-pure_board_size+1], response_start) 
    eq_(white.tensor[F_RESPONSE][pure_moves['a']-2], response_start) 
    eq_(white.tensor[F_RESPONSE][pure_moves['a']-1], response_start) 
    eq_(white.tensor[F_RESPONSE][pure_moves['a']+1], response_start) 
    eq_(white.tensor[F_RESPONSE][pure_moves['a']+2], response_start) 
    eq_(white.tensor[F_RESPONSE][pure_moves['a']+pure_board_size-1], response_start) 
    eq_(white.tensor[F_RESPONSE][pure_moves['a']+pure_board_size], response_start) 
    eq_(white.tensor[F_RESPONSE][pure_moves['a']+pure_board_size+1], response_start) 
    eq_(white.tensor[F_RESPONSE][pure_moves['a']+2*pure_board_size], response_start) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['a']-2*pure_board_size], d12_start) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['a']-pure_board_size-1], d12_start+1) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['a']-pure_board_size], d12_start+2) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['a']-pure_board_size+1], d12_start+1) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['a']-2], d12_start) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['a']-1], d12_start+2) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['a']+1], d12_start+2) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['a']+2], d12_start) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['a']+pure_board_size-1], d12_start+1) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['a']+pure_board_size], d12_start+2) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['a']+pure_board_size+1], d12_start+1) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['a']+2*pure_board_size], d12_start) 
    eq_(number_of_active_positions(white, F_RESPONSE), 12)
    eq_(number_of_active_positions(white, F_RESPONSE_PAT), 12)

    # put W[b]
    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes_all(game)
    # updated around move['b']
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['b']-2*pure_board_size], d12_start) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['b']-pure_board_size-1], d12_start+1) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['b']-pure_board_size], d12_start+2) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['b']-pure_board_size+1], d12_start+1) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['b']-2], d12_start) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['b']-1], d12_start+2) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['b']+1], d12_start+2) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['b']+2], d12_start) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['b']+pure_board_size-1], d12_start+1) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['b']+pure_board_size], d12_start+2) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['b']+pure_board_size+1], d12_start+1) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['b']+2*pure_board_size], d12_start) 
    # no others
    eq_(number_of_active_positions(black, F_RESPONSE), 12)
    eq_(number_of_active_positions(black, F_RESPONSE_PAT), 12)

    # put B[b]
    put_stone(game, moves['c'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes_all(game)
    # updated around move['c']
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['c']-2*pure_board_size], d12_start+13) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['c']-pure_board_size-1], -1) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['c']-pure_board_size], d12_start+12) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['c']-pure_board_size+1], d12_start+11) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['c']-2], d12_start+10) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['c']-1], d12_start+9) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['c']+1], d12_start+8) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['c']+2], d12_start+7) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['c']+pure_board_size-1], d12_start+6) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['c']+pure_board_size], d12_start+5) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['c']+pure_board_size+1], d12_start+4) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['c']+2*pure_board_size], d12_start+3) 
    # no others
    eq_(number_of_active_positions(white, F_RESPONSE), 11)
    eq_(number_of_active_positions(white, F_RESPONSE_PAT), 11)

    # put W[d]
    put_stone(game, moves['d'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes_all(game)
    # updated around move['d']
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['d']-2*pure_board_size], d12_start+22) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['d']-pure_board_size-1], d12_start+21) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['d']-pure_board_size], -1) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['d']-pure_board_size+1], d12_start+20) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['d']-2], d12_start+19) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['d']-1], d12_start+18) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['d']+1], -1) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['d']+2], d12_start+17) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['d']+pure_board_size-1], d12_start+16) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['d']+pure_board_size], d12_start+15) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['d']+pure_board_size+1], d12_start+14) 
    eq_(black.tensor[F_RESPONSE_PAT][pure_moves['d']+2*pure_board_size], -1) 
    # no others, previous positions are cleared
    eq_(number_of_active_positions(black, F_RESPONSE), 9)
    eq_(number_of_active_positions(black, F_RESPONSE_PAT), 9)

    # put B[e]
    put_stone(game, moves['e'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes_all(game)
    # updated around move['e']
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['e']-2*pure_board_size], d12_start+31) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['e']-pure_board_size-1], d12_start+30) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['e']-pure_board_size], d12_start+29) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['e']-pure_board_size+1], d12_start+28) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['e']-2], d12_start+27) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['e']-1], -1) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['e']+1], d12_start+26) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['e']+2], d12_start+25) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['e']+pure_board_size-1], -1) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['e']+pure_board_size], -1) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['e']+pure_board_size+1], d12_start+24) 
    eq_(white.tensor[F_RESPONSE_PAT][pure_moves['e']+2*pure_board_size], d12_start+23) 
    # no others, previous positions are cleared
    eq_(number_of_active_positions(white, F_RESPONSE), 9)
    eq_(number_of_active_positions(white, F_RESPONSE_PAT), 9)

    # print_board(game)

    free_game(game)


def test_update_all_3x3():
    cdef game_state_t *game = allocate_game()
    cdef rollout_feature_t *black
    cdef rollout_feature_t *white

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

    black = &game.rollout_feature_planes[<int>S_BLACK]
    white = &game.rollout_feature_planes[<int>S_WHITE]

    # put B[a]
    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes_all(game)
    # updated around move['a'] 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['a']-pure_board_size-1], x33_start) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['a']-pure_board_size], x33_start+1) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['a']-pure_board_size+1], x33_start) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['a']-1], x33_start+1) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['a']+1], x33_start+1) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['a']+pure_board_size-1], x33_start) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['a']+pure_board_size], x33_start+1) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['a']+pure_board_size+1], x33_start) 
    eq_(number_of_active_positions(white, F_NON_RESPONSE_PAT), 8)

    # put W[b]
    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes_all(game)
    # updated around move['a'] 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['a']-pure_board_size-1], x33_start + 2) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['a']-pure_board_size], x33_start + 3) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['a']-pure_board_size+1], x33_start + 2) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['a']-1], x33_start + 3) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['a']+1], x33_start + 3) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['a']+pure_board_size-1], x33_start + 2) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['a']+pure_board_size], x33_start + 3) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['a']+pure_board_size+1], x33_start + 2) 
    # updated around move['b']
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['b']-pure_board_size-1], x33_start) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['b']-pure_board_size], x33_start+1) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['b']-pure_board_size+1], x33_start) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['b']-1], x33_start+1) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['b']+1], x33_start+1) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['b']+pure_board_size-1], x33_start) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['b']+pure_board_size], x33_start+1) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['b']+pure_board_size+1], x33_start) 
    eq_(number_of_active_positions(black, F_NON_RESPONSE_PAT), 16)

    # put B[c]
    put_stone(game, moves['c'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes_all(game)
    # updated around move['b']
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['b']-pure_board_size-1], x33_start + 2) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['b']-pure_board_size], x33_start + 3) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['b']-pure_board_size+1], x33_start + 2) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['b']-1], x33_start + 3) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['b']+1], x33_start + 4) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['b']+pure_board_size-1], x33_start + 2) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['b']+pure_board_size], x33_start + 4) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['b']+pure_board_size+1], -1) 
    # updated around move['c']
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['c']-pure_board_size-1], -1) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['c']-pure_board_size], x33_start + 4) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['c']-pure_board_size+1], x33_start) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['c']-1], x33_start + 4) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['c']+1], x33_start + 1) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['c']+pure_board_size-1], x33_start + 5) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['c']+pure_board_size], x33_start + 5) 
    eq_(white.tensor[F_NON_RESPONSE_PAT][pure_moves['c']+pure_board_size+1], x33_start) 
    eq_(number_of_active_positions(white, F_NON_RESPONSE_PAT), 18)

    # put W[d]
    put_stone(game, moves['d'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes_all(game)
    # updated around move['c']
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['c']-pure_board_size-1], -1) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['c']-pure_board_size], x33_start + 6) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['c']-pure_board_size+1], x33_start + 2) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['c']-1], -1) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['c']+1], x33_start + 3) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['c']+pure_board_size-1], x33_start + 7) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['c']+pure_board_size], x33_start + 8) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['c']+pure_board_size+1], x33_start + 2) 
    # updated around move['d']
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['b']-pure_board_size-1], x33_start) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['b']-pure_board_size], x33_start + 1) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['b']-pure_board_size+1], x33_start) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['b']-1], x33_start + 9) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['b']+1], x33_start + 6) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['d']-1], x33_start + 9) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['d']+pure_board_size-1], x33_start + 10) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['d']+pure_board_size], x33_start + 7) 
    eq_(black.tensor[F_NON_RESPONSE_PAT][pure_moves['d']+pure_board_size+1], x33_start + 8) 
    eq_(number_of_active_positions(black, F_NON_RESPONSE_PAT), 17)

    # print_board(game)

    free_game(game)


def test_memorize_updated():
    cdef game_state_t *game = allocate_game()
    cdef rollout_feature_t *black = &game.rollout_feature_planes[<int>S_BLACK]

    initialize_planes(game)

    eq_(black.updated[0], BOARD_MAX)

    # add 264
    eq_(memorize_updated(black, 264), True)
    eq_(black.updated[0], 264)
    eq_(black.updated[264], BOARD_MAX)

    # 264 already exists
    eq_(memorize_updated(black, 264), False)

    # add 48, 480
    eq_(memorize_updated(black, 48), True)
    eq_(black.updated[0], 48)
    eq_(black.updated[48], 264)
    eq_(black.updated[264], BOARD_MAX)
    eq_(memorize_updated(black, 480), True)
    eq_(black.updated[0], 480)
    eq_(black.updated[480], 48)
    eq_(black.updated[48], 264)
    eq_(black.updated[264], BOARD_MAX)


def test_choice_rollout_move():
    cdef game_state_t *game = allocate_game()
    cdef int pos, color 
    cdef int i

    initialize_probs(game)

    game.current_color = S_BLACK
    color = <int>S_BLACK

    game.rollout_probs[color][60] = 0.3
    game.rollout_probs[color][70] = 0.4
    game.rollout_probs[color][288] = 0.1
    game.rollout_probs[color][300] = 0.2

    game.rollout_row_probs[color][3] = 0.7
    game.rollout_row_probs[color][15] = 0.3

    for i in range(100):
        pos = choice_rollout_move(game)
        ok_(pos in (60, 70, 288, 300))


cdef int number_of_active_positions(rollout_feature_t *feature, int feature_id):
    cdef int i
    cdef int n_active = 0
    for i in range(pure_board_max):
        if feature.tensor[feature_id][i] >= 0:
            n_active += 1
    return n_active
