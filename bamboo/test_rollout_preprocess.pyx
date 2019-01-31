import numpy as np
cimport numpy as np

from operator import itemgetter

from libc.stdlib cimport malloc, free

from nose.tools import ok_, eq_

from bamboo.board cimport BOARD_MAX, PURE_BOARD_SIZE, PURE_BOARD_MAX, S_EMPTY, S_BLACK, S_WHITE, PASS
from bamboo.board cimport FLIP_COLOR, Y
from bamboo.board cimport game_state_t, rollout_feature_t, board_size, board_max, pure_board_size, pure_board_max
from bamboo.board cimport onboard_pos 
from bamboo.board cimport set_board_size, allocate_game, free_game, put_stone, copy_game
from bamboo.zobrist_hash cimport initialize_hash 
from bamboo.printer cimport print_board
from bamboo.parseboard cimport parse

from bamboo.rollout_preprocess cimport F_RESPONSE, F_SAVE_ATARI, F_NEIGHBOR, F_NAKADE, F_D12_RSP_PAT, F_X33_PAT
from bamboo.rollout_preprocess cimport response_start, save_atari_start, neighbor_start, nakade_start, d12_rsp_start, x33_start
from bamboo.rollout_preprocess cimport F_SELF_ATARI, F_LAST_MOVE_DISTANCE, F_D12_PAT
from bamboo.rollout_preprocess cimport self_atari_start, last_move_distance_start, d12_start
from bamboo.rollout_preprocess cimport initialize_rollout_const, initialize_planes, initialize_probs, update_planes, update_tree_planes_all, memorize_updated, choice_rollout_move, set_illegal, norm_probs 
from bamboo.local_pattern cimport initialize_rands, put_x33_hash, put_d12_rspos_hash, put_d12_hash
from bamboo.local_pattern import print_x33
from bamboo.nakade cimport initialize_nakade_hash


cdef int nakade_size = 8
cdef int x33_size = 100
cdef int d12_rsp_size = 100
cdef int d12_size = 100


def setup():
    initialize_rollout_const(nakade_size,
            x33_size,
            d12_rsp_size,
            d12_size,
            pos_aware_d12=True)
    initialize_rands()
    initialize_hash()

    put_12diamond_rspos_test_patterns()
    put_3x3_test_patterns()
    put_12diamond_test_patterns()


def teardown():
    pass


def put_12diamond_rspos_test_patterns():
    """
    Pattern 0:
      *       0
     +++     000
    *+B+*   00300
     +++     000
      *       0
    """
    put_d12_rspos_hash(0b0100000000000000000000000011000000000001, 0)
    put_d12_rspos_hash(0b0100000000000000000000000011000000010000, 0)
    put_d12_rspos_hash(0b0100000000000000000000000011000010000000, 0)
    put_d12_rspos_hash(0b0100000000000000000000000011100000000000, 0)
    put_d12_rspos_hash(0b1000000000000000000000000011000000000001, 0)
    put_d12_rspos_hash(0b1000000000000000000000000011000000010000, 0)
    put_d12_rspos_hash(0b1000000000000000000000000011000010000000, 0)
    put_d12_rspos_hash(0b1000000000000000000000000011100000000000, 0)

    """
    Pattern 1:
      +       0
     *+*     000
    ++B++   00300
     *+*     000
      +       0
    """
    put_d12_rspos_hash(0b0100000000000000000000000011000000000010, 1)
    put_d12_rspos_hash(0b0100000000000000000000000011000000001000, 1)
    put_d12_rspos_hash(0b0100000000000000000000000011000100000000, 1)
    put_d12_rspos_hash(0b0100000000000000000000000011010000000000, 1)
    put_d12_rspos_hash(0b1000000000000000000000000011000000000010, 1)
    put_d12_rspos_hash(0b1000000000000000000000000011000000001000, 1)
    put_d12_rspos_hash(0b1000000000000000000000000011000100000000, 1)
    put_d12_rspos_hash(0b1000000000000000000000000011010000000000, 1)

    """
    Pattern 2:
      +       0
     +*+     000
    +*B*+   00300
     +*+     000
      +       0
    """
    put_d12_rspos_hash(0b0100000000000000000000000011000000000100, 2)
    put_d12_rspos_hash(0b0100000000000000000000000011000000100000, 2)
    put_d12_rspos_hash(0b0100000000000000000000000011000001000000, 2)
    put_d12_rspos_hash(0b0100000000000000000000000011001000000000, 2)
    put_d12_rspos_hash(0b1000000000000000000000000011000000000100, 2)
    put_d12_rspos_hash(0b1000000000000000000000000011000000100000, 2)
    put_d12_rspos_hash(0b1000000000000000000000000011000001000000, 2)
    put_d12_rspos_hash(0b1000000000000000000000000011001000000000, 2)

    """
    Pattern 3-13:
      +       0
     W++     300
    ++B++   00300
     +++     000
      *       0
    """
    put_d12_rspos_hash(0b10000100000000000000000000110011100000000000, 3)
    put_d12_rspos_hash(0b10000100000000000000000000110011010000000000, 4)
    put_d12_rspos_hash(0b10000100000000000000000000110011001000000000, 5)
    put_d12_rspos_hash(0b10000100000000000000000000110011000100000000, 6)
    put_d12_rspos_hash(0b10000100000000000000000000110011000010000000, 7)
    put_d12_rspos_hash(0b10000100000000000000000000110011000001000000, 8)
    put_d12_rspos_hash(0b10000100000000000000000000110011000000100000, 9)
    put_d12_rspos_hash(0b10000100000000000000000000110011000000010000, 10)
    put_d12_rspos_hash(0b10000100000000000000000000110011000000001000, 11)
    put_d12_rspos_hash(0b10000100000000000000000000110011000000000100, 12)
    put_d12_rspos_hash(0b10000100000000000000000000110011000000000001, 13)

    """
    Pattern 14-22:
      +       0
     +W+     030
    ++WB+   00330
     ++*     000
      B       3
    """
    put_d12_rspos_hash(0b0100000000010000001000001011000000001100000011000011010000000000, 14)
    put_d12_rspos_hash(0b0100000000010000001000001011000000001100000011000011001000000000, 15)
    put_d12_rspos_hash(0b0100000000010000001000001011000000001100000011000011000100000000, 16)
    put_d12_rspos_hash(0b0100000000010000001000001011000000001100000011000011000010000000, 17)
    put_d12_rspos_hash(0b0100000000010000001000001011000000001100000011000011000000100000, 18)
    put_d12_rspos_hash(0b0100000000010000001000001011000000001100000011000011000000010000, 19)
    put_d12_rspos_hash(0b0100000000010000001000001011000000001100000011000011000000001000, 20)
    put_d12_rspos_hash(0b0100000000010000001000001011000000001100000011000011000000000010, 21)
    put_d12_rspos_hash(0b0100000000010000001000001011000000001100000011000011000000000001, 22)

    """
    Pattern 23-31:
      +       0
     +++     000
    +WB++   03300
     WB+     330
      +       0
    """
    put_d12_rspos_hash(0b011000001000000000000100001111000011000000000011100000000000, 23)
    put_d12_rspos_hash(0b011000001000000000000100001111000011000000000011010000000000, 24)
    put_d12_rspos_hash(0b011000001000000000000100001111000011000000000011000010000000, 25)
    put_d12_rspos_hash(0b011000001000000000000100001111000011000000000011000001000000, 26)
    put_d12_rspos_hash(0b011000001000000000000100001111000011000000000011000000010000, 27)
    put_d12_rspos_hash(0b011000001000000000000100001111000011000000000011000000001000, 28)
    put_d12_rspos_hash(0b011000001000000000000100001111000011000000000011000000000100, 29)
    put_d12_rspos_hash(0b011000001000000000000100001111000011000000000011000000000010, 30)
    put_d12_rspos_hash(0b011000001000000000000100001111000011000000000011000000000001, 31)


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
        +x+    0 0
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


def put_12diamond_test_patterns():
    """
    Pattern 0:
      +       0
     +B+     030
    ++o++   00 00
     +++     000
      +       0
    """
    put_d12_hash(0b00000000000000000001000000000000000000000011000010, 0)
    put_d12_hash(0b00000000000001000000000000000000000011000000000010, 0)
    put_d12_hash(0b00000000000100000000000000000000001100000000000010, 0)
    put_d12_hash(0b00000100000000000000000000001100000000000000000010, 0)
    put_d12_hash(0b00000000000000000010000000000000000000000011000001, 0)
    put_d12_hash(0b00000000000010000000000000000000000011000000000001, 0)
    put_d12_hash(0b00000000001000000000000000000000001100000000000001, 0)
    put_d12_hash(0b00001000000000000000000000001100000000000000000001, 0)

    """
    Pattern 1:
      +       0
     B++     300
    ++o++   00 00
     +++     000
      +       0
    """
    put_d12_hash(0b00000000000000000000010000000000000000000000110010, 1)
    put_d12_hash(0b00000000000000000100000000000000000000001100000010, 1)
    put_d12_hash(0b00000001000000000000000000000011000000000000000010, 1)
    put_d12_hash(0b00010000000000000000000000110000000000000000000010, 1)
    put_d12_hash(0b00000000000000000000100000000000000000000000110001, 1)
    put_d12_hash(0b00000000000000001000000000000000000000001100000001, 1)
    put_d12_hash(0b00000010000000000000000000000011000000000000000001, 1)
    put_d12_hash(0b00100000000000000000000000110000000000000000000001, 1)

    """
    Pattern 2:
      +       0
     +B+     030
    ++x++   00 00
     +++     000
      +       0
    """
    put_d12_hash(0b00000000000000000001000000000000000000000011000001, 2)
    put_d12_hash(0b00000000000001000000000000000000000011000000000001, 2)
    put_d12_hash(0b00000000000100000000000000000000001100000000000001, 2)
    put_d12_hash(0b00000100000000000000000000001100000000000000000001, 2)
    put_d12_hash(0b00000000000000000010000000000000000000000011000010, 2)
    put_d12_hash(0b00000000000010000000000000000000000011000000000010, 2)
    put_d12_hash(0b00000000001000000000000000000000001100000000000010, 2)
    put_d12_hash(0b00001000000000000000000000001100000000000000000010, 2)

    """
    Pattern 3:
      +       0
     B++     300
    ++x++   00 00
     +++     000
      +       0
    """
    put_d12_hash(0b00000000000000000000010000000000000000000000110001, 3)
    put_d12_hash(0b00000000000000000100000000000000000000001100000001, 3)
    put_d12_hash(0b00000001000000000000000000000011000000000000000001, 3)
    put_d12_hash(0b00010000000000000000000000110000000000000000000001, 3)
    put_d12_hash(0b00000000000000000000100000000000000000000000110010, 3)
    put_d12_hash(0b00000000000000001000000000000000000000001100000010, 3)
    put_d12_hash(0b00000010000000000000000000000011000000000000000010, 3)
    put_d12_hash(0b00100000000000000000000000110000000000000000000010, 3)

    """
    Pattern 4:
      B       3
     +++     000
    ++o++   00 00
     +++     000
      +       0
    """
    put_d12_hash(0b00000000000000000000000100000000000000000000001110, 4)
    put_d12_hash(0b00000000000000010000000000000000000000110000000010, 4)
    put_d12_hash(0b00000000010000000000000000000000110000000000000010, 4)
    put_d12_hash(0b01000000000000000000000011000000000000000000000010, 4)
    put_d12_hash(0b00000000000000000000001000000000000000000000001101, 4)
    put_d12_hash(0b00000000000000100000000000000000000000110000000001, 4)
    put_d12_hash(0b00000000100000000000000000000000110000000000000001, 4)
    put_d12_hash(0b10000000000000000000000011000000000000000000000001, 4)

    """
    Pattern 5:
      B       3
     +++     000
    ++x++   00 00
     +++     000
      +       0
    """
    put_d12_hash(0b00000000000000000000000100000000000000000000001101, 5)
    put_d12_hash(0b00000000000000010000000000000000000000110000000001, 5)
    put_d12_hash(0b00000000010000000000000000000000110000000000000001, 5)
    put_d12_hash(0b01000000000000000000000011000000000000000000000001, 5)
    put_d12_hash(0b00000000000000000000001000000000000000000000001110, 5)
    put_d12_hash(0b00000000000000100000000000000000000000110000000010, 5)
    put_d12_hash(0b00000000100000000000000000000000110000000000000010, 5)
    put_d12_hash(0b10000000000000000000000011000000000000000000000010, 5)

    """
    Pattern 6:
      B       3
     +++     000
    ++o++   00 00
     +++     000
      #       0
    """
    put_d12_hash(0b11000000000000000000000100000000000000000000001110, 6)
    put_d12_hash(0b00000000110000010000000000000000000000110000000010, 6)
    put_d12_hash(0b00000000010000110000000000000000110000000000000010, 6)
    put_d12_hash(0b01000000000000000000001111000000000000000000000010, 6)
    put_d12_hash(0b11000000000000000000001000000000000000000000001101, 6)
    put_d12_hash(0b00000000110000100000000000000000000000110000000001, 6)
    put_d12_hash(0b00000000100000110000000000000000110000000000000001, 6)
    put_d12_hash(0b10000000000000000000001111000000000000000000000001, 6)

    """
    Pattern 7:
      B       3
     +++     000
    ++x++   00 00
     +++     000
      #       0
    """
    put_d12_hash(0b11000000000000000000000100000000000000000000001101, 7)
    put_d12_hash(0b00000000110000010000000000000000000000110000000001, 7)
    put_d12_hash(0b00000000010000110000000000000000110000000000000001, 7)
    put_d12_hash(0b01000000000000000000001111000000000000000000000001, 7)
    put_d12_hash(0b11000000000000000000001000000000000000000000001110, 7)
    put_d12_hash(0b00000000110000100000000000000000000000110000000010, 7)
    put_d12_hash(0b00000000100000110000000000000000110000000000000010, 7)
    put_d12_hash(0b10000000000000000000001111000000000000000000000010, 7)

    """
    Pattern 8:
      +       0
     ++B     003
    ++o++   00 00
     +++     000
      #       0
    """
    put_d12_hash(0b11000000000000000100000000000000000000001100000010, 8)
    put_d12_hash(0b11000000000000000000010000000000000000000000110010, 8)
    put_d12_hash(0b00000000110000000000010000000000000000000000110010, 8)

    """
    Pattern 9:
      +       0
     +B+     030
    ++o++   00 00
     +++     000
      #       0
    """
    put_d12_hash(0b11000000000000000001000000000000000000000011000010, 9)
    put_d12_hash(0b00000000110001000000000000000000000011000000000010, 9)

    """
    Pattern 10:
      B       3
     +++     000
    ++o++   00 00
     ###     000
      #       0
    """
    put_d12_hash(0b11111111000000000000000100000000000000000000001110, 10)
    put_d12_hash(0b00110000111100011100000000000000000000110000000010, 10)

    """
    Pattern 11:
      +       +
     +W+     030
    ++x++   00 00
     +++     000
      B       3
    """
    put_d12_hash(0b01000000000000000010000011000000000000000011000001, 11)

    """
    Pattern 12:
      W       3
     +++     000
    ++x++   00 00
     +B+     030
      +       0
    """
    put_d12_hash(0b00000100000000000000001000001100000000000000001101, 12)

    """
    Pattern 13:
      +       0
     +B+     030
    ++x++   00 00
     +++     000
      #       0
    """
    put_d12_hash(0b11000000000000000001000000000000000000000011000001, 13)

    """
    Pattern 14:
      +       0
     B++     300
    ++x++   00 00
     +++     000
      #       0
    """
    put_d12_hash(0b11000000000000000000010000000000000000000000110001, 14)
    put_d12_hash(0b11000000000000000100000000000000000000001100000001, 14)

    """
    Pattern 15:
      B       3
     +++     000
    ++x++   00 00
     ###     000
      #       0
    """
    put_d12_hash(0b11111111000000000000000100000000000000000000001101, 15)

    """
    Pattern 16:
      +       0
     +++     000
    ++o++   00 00
     W++     300
      B       3
    """
    put_d12_hash(0b01000010000000000000000011000011000000000000000010, 16)

    """
    Pattern 17:
      +       0
     +++     000
    +Wo++   03 00
     +B+     030
      +       0
    """
    put_d12_hash(0b00000100000010000000000000001100000011000000000010, 17)

    """
    Pattern 18:
      +       0
     +++     000
    W+o+#   30 00
     B++     300
      +       0
    """
    put_d12_hash(0b00000001110000100000000000000011000000110000000010, 18)

    """
    Pattern 19:
      +       0
     ++W     003
    ++o+B   00 03
     +++     000
      +       0
    """
    put_d12_hash(0b00000000010000001000000000000000110000001100000010, 19)

    """
    Pattern 20:
      +       0
     +W+     030
    ++oB+   00 30
     +++     000
      B       0
    """
    put_d12_hash(0b01000000000100000010000011000000001100000011000010, 20)

    """
    Pattern 21:
      W       3
     ++B     003
    ++o++   00 00
     +B+     030
      +       0
    """
    put_d12_hash(0b00000100000000000100001000001100000000001100001110, 21)

    """
    Pattern 22:
      +       0
     +B+     030
    ++o++   00 00
     B++     300
      +       0
    """
    put_d12_hash(0b00000001000000000001000000000011000000000011000010, 22)

    """
    Pattern 23:
      B       3
     +++     000
    +Bo++   03 00
     +++     000
      +       0
    """
    put_d12_hash(0b00000000000001000000000100000000000011000000001110, 23)


def test_update_nakade_3_0():
    cdef game_state_t *game = allocate_game()
    cdef rollout_feature_t *feature

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . B b B . .|"
                              ". B W a W B .|"
                              ". . B B B . .|"
                              ". . . . . . .|"
                              ". . . . . . .|")

    initialize_nakade_hash()

    game.current_color = S_WHITE

    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    feature = &game.rollout_feature_planes[<int>S_WHITE]

    eq_(feature.tensor[F_NAKADE][moves['a']], nakade_start) 
    eq_(number_of_active_positions(feature, F_NAKADE), 1)

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". . . W . . .|"
                              ". . W B W . .|"
                              ". . W a b . .|"
                              ". . W B W . .|"
                              ". . . W . . .|"
                              ". . . . . . .|")

    game.current_color = S_BLACK

    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    feature = &game.rollout_feature_planes[<int>S_BLACK]

    eq_(feature.tensor[F_NAKADE][moves['a']], nakade_start) 
    eq_(number_of_active_positions(feature, F_NAKADE), 1)

    free_game(game)


def test_update_nakade_3_1():
    cdef game_state_t *game = allocate_game()
    cdef rollout_feature_t *feature

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". B B B B . .|"
                              ". B W a b . .|"
                              ". B B W B . .|"
                              ". . B B B . .|"
                              ". . . . . . .|")

    initialize_nakade_hash()

    game.current_color = S_WHITE

    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    feature = &game.rollout_feature_planes[<int>S_WHITE]

    eq_(feature.tensor[F_NAKADE][moves['a']], nakade_start + 1) 
    eq_(number_of_active_positions(feature, F_NAKADE), 1)

    game.current_color = S_WHITE

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . B B B . .|"
                              ". B B W B . .|"
                              ". B W a B . .|"
                              ". B B b B . .|"
                              ". . . . . . .|")

    game.current_color = S_WHITE

    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    feature = &game.rollout_feature_planes[<int>S_WHITE]

    eq_(feature.tensor[F_NAKADE][moves['a']], nakade_start + 1) 
    eq_(number_of_active_positions(feature, F_NAKADE), 1)

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . B B B . .|"
                              ". . B W B B .|"
                              ". . B a W B .|"
                              ". . B b B B .|"
                              ". . . . . . .|")

    game.current_color = S_WHITE

    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    feature = &game.rollout_feature_planes[<int>S_WHITE]

    eq_(feature.tensor[F_NAKADE][moves['a']], nakade_start + 1) 
    eq_(number_of_active_positions(feature, F_NAKADE), 1)

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . B B B B .|"
                              ". . b a W B .|"
                              ". . B W B B .|"
                              ". . B B B . .|"
                              ". . . . . . .|")

    game.current_color = S_WHITE

    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    feature = &game.rollout_feature_planes[<int>S_WHITE]

    eq_(feature.tensor[F_NAKADE][moves['a']], nakade_start + 1) 
    eq_(number_of_active_positions(feature, F_NAKADE), 1)

    free_game(game)


def test_update_nakade_4():
    cdef game_state_t *game = allocate_game()
    cdef rollout_feature_t *feature

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". . W W W . .|"
                              ". W W B W W .|"
                              ". W B a B W .|"
                              ". W W b W W .|"
                              ". . . . . . .|"
                              ". . . . . . .|")

    initialize_nakade_hash()

    game.current_color = S_BLACK

    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    feature = &game.rollout_feature_planes[<int>S_BLACK]

    eq_(feature.tensor[F_NAKADE][moves['a']], nakade_start + 2) 
    eq_(number_of_active_positions(feature, F_NAKADE), 1)

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". . W W W . .|"
                              ". W W B W . .|"
                              ". W B a b . .|"
                              ". W W B W . .|"
                              ". . W W W . .|"
                              ". . . . . . .|")

    game.current_color = S_BLACK

    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    feature = &game.rollout_feature_planes[<int>S_BLACK]

    eq_(feature.tensor[F_NAKADE][moves['a']], nakade_start + 2) 
    eq_(number_of_active_positions(feature, F_NAKADE), 1)

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". . W W W . .|"
                              ". . W B W W .|"
                              ". . b a B W .|"
                              ". . W B W W .|"
                              ". . W W W . .|"
                              ". . . . . . .|")

    game.current_color = S_BLACK

    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    feature = &game.rollout_feature_planes[<int>S_BLACK]

    eq_(feature.tensor[F_NAKADE][moves['a']], nakade_start + 2) 
    eq_(number_of_active_positions(feature, F_NAKADE), 1)

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". W W b W W .|"
                              ". W B a B W .|"
                              ". W W B W W .|"
                              ". . W W W . .|"
                              ". . . . . . .|")

    game.current_color = S_BLACK

    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    feature = &game.rollout_feature_planes[<int>S_BLACK]

    eq_(feature.tensor[F_NAKADE][moves['a']], nakade_start + 2) 
    eq_(number_of_active_positions(feature, F_NAKADE), 1)

    free_game(game)


def test_update_nakade_5_0():
    cdef game_state_t *game = allocate_game()
    cdef rollout_feature_t *feature

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". . B B B . .|"
                              ". B B W B B .|"
                              ". B W a W B .|"
                              ". B B W B B .|"
                              ". . B b B . .|"
                              ". . . . . . .|")

    initialize_nakade_hash()

    game.current_color = S_WHITE

    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    feature = &game.rollout_feature_planes[<int>S_WHITE]

    eq_(feature.tensor[F_NAKADE][moves['a']], nakade_start + 3) 
    eq_(number_of_active_positions(feature, F_NAKADE), 1)

    free_game(game)


def test_update_nakade_5_1():
    cdef game_state_t *game = allocate_game()
    cdef rollout_feature_t *feature

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". . B B B . .|"
                              ". B B W B . .|"
                              ". B W a b . .|"
                              ". B W W B . .|"
                              ". B B B B . .|"
                              ". . . . . . .|")

    initialize_nakade_hash()

    game.current_color = S_WHITE

    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    feature = &game.rollout_feature_planes[<int>S_WHITE]

    eq_(feature.tensor[F_NAKADE][moves['a']], nakade_start + 4) 
    eq_(number_of_active_positions(feature, F_NAKADE), 1)

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". B B B B . .|"
                              ". B W W B B .|"
                              ". B W a W B .|"
                              ". B B b B B .|"
                              ". . . . . . .|"
                              ". . . . . . .|")

    game.current_color = S_WHITE

    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    feature = &game.rollout_feature_planes[<int>S_WHITE]

    eq_(feature.tensor[F_NAKADE][moves['a']], nakade_start + 4) 
    eq_(number_of_active_positions(feature, F_NAKADE), 1)

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". . B B B B .|"
                              ". . B W W B .|"
                              ". . b a W B .|"
                              ". . B W B B .|"
                              ". . B B B . .|"
                              ". . . . . . .|")

    game.current_color = S_WHITE

    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    feature = &game.rollout_feature_planes[<int>S_WHITE]

    eq_(feature.tensor[F_NAKADE][moves['a']], nakade_start + 4) 
    eq_(number_of_active_positions(feature, F_NAKADE), 1)

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". B B b B B .|"
                              ". B W a W B .|"
                              ". B B W W B .|"
                              ". . B B B B .|"
                              ". . . . . . .|")

    game.current_color = S_WHITE

    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    eq_(feature.tensor[F_NAKADE][moves['a']], nakade_start + 4) 
    eq_(number_of_active_positions(feature, F_NAKADE), 1)

    feature = &game.rollout_feature_planes[<int>S_WHITE]

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". . B B B . .|"
                              ". . B W B B .|"
                              ". . b a W B .|"
                              ". . B W W B .|"
                              ". . B B B B .|"
                              ". . . . . . .|")

    game.current_color = S_WHITE

    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    eq_(feature.tensor[F_NAKADE][moves['a']], nakade_start + 4) 
    eq_(number_of_active_positions(feature, F_NAKADE), 1)

    feature = &game.rollout_feature_planes[<int>S_WHITE]

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". B B b B B .|"
                              ". B W a W B .|"
                              ". B W W B B .|"
                              ". B B B B . .|"
                              ". . . . . . .|")

    game.current_color = S_WHITE

    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    eq_(feature.tensor[F_NAKADE][moves['a']], nakade_start + 4) 
    eq_(number_of_active_positions(feature, F_NAKADE), 1)

    feature = &game.rollout_feature_planes[<int>S_WHITE]

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". B B B B . .|"
                              ". B W W B . .|"
                              ". B W a b . .|"
                              ". B B W B . .|"
                              ". . B B B . .|"
                              ". . . . . . .|")

    game.current_color = S_WHITE

    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    eq_(feature.tensor[F_NAKADE][moves['a']], nakade_start + 4) 
    eq_(number_of_active_positions(feature, F_NAKADE), 1)

    feature = &game.rollout_feature_planes[<int>S_WHITE]

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". . B B B B .|"
                              ". B B W W B .|"
                              ". B W a W B .|"
                              ". B B b B B .|"
                              ". . . . . . .|"
                              ". . . . . . .|")

    game.current_color = S_WHITE

    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    eq_(feature.tensor[F_NAKADE][moves['a']], nakade_start + 4) 
    eq_(number_of_active_positions(feature, F_NAKADE), 1)

    feature = &game.rollout_feature_planes[<int>S_WHITE]

    free_game(game)


def test_update_nakade_6():
    cdef game_state_t *game = allocate_game()
    cdef rollout_feature_t *feature

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". . W W W . .|"
                              ". W W B W W .|"
                              ". W B a B W .|"
                              ". W B B W W .|"
                              ". W W b W . .|"
                              ". . . . . . .|")

    initialize_nakade_hash()

    game.current_color = S_BLACK

    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    feature = &game.rollout_feature_planes[<int>S_BLACK]

    eq_(feature.tensor[F_NAKADE][moves['a']], nakade_start + 5) 
    eq_(number_of_active_positions(feature, F_NAKADE), 1)

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". W W W W . .|"
                              ". W B B W W .|"
                              ". b B a B W .|"
                              ". W W B W W .|"
                              ". . W W W . .|"
                              ". . . . . . .|")

    game.current_color = S_BLACK

    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    feature = &game.rollout_feature_planes[<int>S_BLACK]

    eq_(feature.tensor[F_NAKADE][moves['a']], nakade_start + 5) 
    eq_(number_of_active_positions(feature, F_NAKADE), 1)

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". . W b W W .|"
                              ". W W B B W .|"
                              ". W B a B W .|"
                              ". W W B W W .|"
                              ". . W W W . .|"
                              ". . . . . . .|")

    game.current_color = S_BLACK

    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    feature = &game.rollout_feature_planes[<int>S_BLACK]

    eq_(feature.tensor[F_NAKADE][moves['a']], nakade_start + 5) 
    eq_(number_of_active_positions(feature, F_NAKADE), 1)

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". . W W W . .|"
                              ". W W B W W .|"
                              ". W B a B b .|"
                              ". W W B B W .|"
                              ". . W W W W .|"
                              ". . . . . . .|")

    game.current_color = S_BLACK

    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)

    feature = &game.rollout_feature_planes[<int>S_BLACK]

    eq_(feature.tensor[F_NAKADE][moves['a']], nakade_start + 5) 
    eq_(number_of_active_positions(feature, F_NAKADE), 1)

    free_game(game)


def test_update_self_atari():
    cdef game_state_t *game = allocate_game()
    cdef rollout_feature_t *feature

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . W a W . .|"
                              ". . W B W . .|"
                              ". . . W . . .|"
                              ". . . . . . .|"
                              ". . . . . . .|")

    game.current_color = S_BLACK

    update_tree_planes_all(game)

    feature = &game.rollout_feature_planes[<int>S_BLACK]

    eq_(feature.tensor[F_SELF_ATARI][moves['a']], self_atari_start) 
    eq_(number_of_active_positions(feature, F_SELF_ATARI), 1)

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . B W B . .|"
                              ". . B a B . .|"
                              ". . . B . . .|"
                              ". . . . . . .|"
                              ". . . . . . .|")

    game.current_color = S_WHITE

    update_tree_planes_all(game)

    feature = &game.rollout_feature_planes[<int>S_WHITE]

    eq_(feature.tensor[F_SELF_ATARI][moves['a']], self_atari_start) 
    eq_(number_of_active_positions(feature, F_SELF_ATARI), 1)

    (moves, pure_moves) = parse(game,
                              "B a . . . . .|"
                              "W W . . . . .|"
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . . . . . .|")

    game.current_color = S_BLACK

    update_tree_planes_all(game)

    feature = &game.rollout_feature_planes[<int>S_BLACK]

    eq_(feature.tensor[F_SELF_ATARI][moves['a']], self_atari_start) 
    eq_(number_of_active_positions(feature, F_SELF_ATARI), 1)

    (moves, pure_moves) = parse(game,
                              "a B . . . . .|"
                              "W W . . . . .|"
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . . . . . .|")

    game.current_color = S_BLACK

    update_tree_planes_all(game)

    feature = &game.rollout_feature_planes[<int>S_BLACK]

    eq_(feature.tensor[F_SELF_ATARI][moves['a']], self_atari_start) 
    eq_(number_of_active_positions(feature, F_SELF_ATARI), 1)

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . . . B B B|"
                              ". . . . . a W|")

    game.current_color = S_WHITE

    update_tree_planes_all(game)

    feature = &game.rollout_feature_planes[<int>S_WHITE]

    eq_(feature.tensor[F_SELF_ATARI][moves['a']], self_atari_start) 
    eq_(number_of_active_positions(feature, F_SELF_ATARI), 1)

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . . . B B B|"
                              ". . . . W W a|")

    game.current_color = S_WHITE

    update_tree_planes_all(game)

    feature = &game.rollout_feature_planes[<int>S_WHITE]

    eq_(feature.tensor[F_SELF_ATARI][moves['a']], self_atari_start) 
    eq_(number_of_active_positions(feature, F_SELF_ATARI), 1)

    free_game(game)


def test_update_last_move_distance():
    cdef game_state_t *game = allocate_game()
    cdef rollout_feature_t *feature

    (moves, pure_moves) = parse(game,
        ". . . . . . . . . . . . . . . . . . .|"
        ". . . . . . . . . . . . . . . . . . .|"
        ". . . . . . . . . . . . . . . a . . .|"
        ". . . . . . f e . . . . . . b B c . .|"
        ". . . . . . . . . . . . . . . d . . .|"
        ". . . . . . . . . . . . . . . . . . .|"
        ". . . . . . . . . . . . . . . . . . .|"
        ". . . . . . . . . . . . . . . . . . .|"
        ". . . . . . . . . . i . . . . . . . .|"
        ". . . . . . . . . j . . . . . . . . .|"
        ". . . . . . . . . . . . . . . . . . .|"
        ". . . . . . . . . . . . . . . g . . .|"
        ". . . . . . . . . . . . . . . h . . .|"
        ". . . . . . . . . . . . . . . . . . .|"
        ". . . . . . . . . . . . . . . . . . .|"
        ". . . . . . . . . . . . . . . . . . .|"
        ". . . . . . . . . . . . . . . . . . .|"
        ". . . . . . . . . . . . . . . . . . .|"
        ". . . . . . . . . . . . . . . . . . .|")

    game.current_color = S_WHITE

    update_tree_planes_all(game)

    feature = &game.rollout_feature_planes[<int>S_WHITE]

    eq_(feature.tensor[F_LAST_MOVE_DISTANCE][moves['a']], last_move_distance_start + 2) 
    eq_(feature.tensor[F_LAST_MOVE_DISTANCE][moves['b']], last_move_distance_start + 2) 
    eq_(feature.tensor[F_LAST_MOVE_DISTANCE][moves['c']], last_move_distance_start + 2) 
    eq_(feature.tensor[F_LAST_MOVE_DISTANCE][moves['d']], last_move_distance_start + 2) 
    eq_(feature.tensor[F_LAST_MOVE_DISTANCE][moves['e']], last_move_distance_start + 16) 
    eq_(feature.tensor[F_LAST_MOVE_DISTANCE][moves['f']], last_move_distance_start + 17) 
    eq_(feature.tensor[F_LAST_MOVE_DISTANCE][moves['g']], last_move_distance_start + 16) 
    eq_(feature.tensor[F_LAST_MOVE_DISTANCE][moves['h']], last_move_distance_start + 17) 
    eq_(feature.tensor[F_LAST_MOVE_DISTANCE][moves['i']], last_move_distance_start + 15) 
    eq_(feature.tensor[F_LAST_MOVE_DISTANCE][moves['j']], last_move_distance_start + 17) 

    put_stone(game, moves['a'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_tree_planes_all(game)

    feature = &game.rollout_feature_planes[<int>S_BLACK]

    eq_(feature.tensor[F_LAST_MOVE_DISTANCE][moves['b']], last_move_distance_start + 3 + 2) 
    eq_(feature.tensor[F_LAST_MOVE_DISTANCE][moves['c']], last_move_distance_start + 3 + 2) 
    eq_(feature.tensor[F_LAST_MOVE_DISTANCE][moves['d']], last_move_distance_start + 4 + 2) 
    eq_(feature.tensor[F_LAST_MOVE_DISTANCE][moves['e']], last_move_distance_start + 17 + 16) 
    eq_(feature.tensor[F_LAST_MOVE_DISTANCE][moves['f']], last_move_distance_start + 17 + 17) 
    eq_(feature.tensor[F_LAST_MOVE_DISTANCE][moves['g']], last_move_distance_start + 17 + 16) 
    eq_(feature.tensor[F_LAST_MOVE_DISTANCE][moves['h']], last_move_distance_start + 17 + 17) 
    eq_(feature.tensor[F_LAST_MOVE_DISTANCE][moves['i']], last_move_distance_start + 17 + 15) 
    eq_(feature.tensor[F_LAST_MOVE_DISTANCE][moves['j']], last_move_distance_start + 17 + 17) 

    free_game(game)

def test_update_12diamond():
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
    update_tree_planes_all(game)

    eq_(white.tensor[F_D12_PAT][moves['a']-board_size*2], d12_start + 4) 
    eq_(white.tensor[F_D12_PAT][moves['a']-board_size-1], d12_start + 1) 
    eq_(white.tensor[F_D12_PAT][moves['a']-board_size],   d12_start) 
    eq_(white.tensor[F_D12_PAT][moves['a']-board_size+1], d12_start + 1) 
    eq_(white.tensor[F_D12_PAT][moves['a']-2], d12_start + 4) 
    eq_(white.tensor[F_D12_PAT][moves['a']-1], d12_start) 
    eq_(white.tensor[F_D12_PAT][moves['a']+1], d12_start) 
    eq_(white.tensor[F_D12_PAT][moves['a']+2], d12_start + 6) 
    eq_(white.tensor[F_D12_PAT][moves['a']+board_size-1], d12_start + 8) 
    eq_(white.tensor[F_D12_PAT][moves['a']+board_size],   d12_start + 9) 
    eq_(white.tensor[F_D12_PAT][moves['a']+board_size+1], d12_start + 8) 
    eq_(white.tensor[F_D12_PAT][moves['a']+board_size*2], d12_start + 10) 
    eq_(number_of_active_positions(white, F_D12_PAT), 12)

    # put W[b]
    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_tree_planes_all(game)

    eq_(black.tensor[F_D12_PAT][moves['b']-board_size*2], d12_start + 6) 
    eq_(black.tensor[F_D12_PAT][moves['b']-board_size-1], d12_start + 1) 
    eq_(black.tensor[F_D12_PAT][moves['b']-board_size], d12_start) 
    eq_(black.tensor[F_D12_PAT][moves['b']-board_size+1], d12_start + 1) 
    eq_(black.tensor[F_D12_PAT][moves['b']-2], d12_start + 4) 
    eq_(black.tensor[F_D12_PAT][moves['b']-1], d12_start) 
    eq_(black.tensor[F_D12_PAT][moves['b']+1], d12_start) 
    eq_(black.tensor[F_D12_PAT][moves['b']+2], d12_start + 6) 
    eq_(black.tensor[F_D12_PAT][moves['b']+board_size-1], d12_start + 1) 
    eq_(black.tensor[F_D12_PAT][moves['b']+board_size], d12_start + 11) 
    eq_(black.tensor[F_D12_PAT][moves['b']+board_size+1], d12_start + 1) 

    eq_(black.tensor[F_D12_PAT][moves['a']-board_size-1], d12_start + 3) 
    eq_(black.tensor[F_D12_PAT][moves['a']-board_size], d12_start + 12) 
    eq_(black.tensor[F_D12_PAT][moves['a']-board_size+1], d12_start + 3) 
    eq_(black.tensor[F_D12_PAT][moves['a']-2], d12_start + 5) 
    eq_(black.tensor[F_D12_PAT][moves['a']-1], d12_start + 2) 
    eq_(black.tensor[F_D12_PAT][moves['a']+1], d12_start + 2) 
    eq_(black.tensor[F_D12_PAT][moves['a']+2], d12_start + 7) 
    eq_(black.tensor[F_D12_PAT][moves['a']+board_size-1], d12_start + 14) 
    eq_(black.tensor[F_D12_PAT][moves['a']+board_size], d12_start + 13) 
    eq_(black.tensor[F_D12_PAT][moves['a']+board_size+1], d12_start + 14) 
    eq_(black.tensor[F_D12_PAT][moves['a']+board_size*2], d12_start + 15) 
    eq_(number_of_active_positions(black, F_D12_PAT), 22)

    # put B[c]
    put_stone(game, moves['c'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_tree_planes_all(game)

    eq_(white.tensor[F_D12_PAT][moves['c']-board_size*2], d12_start + 16) 
    eq_(white.tensor[F_D12_PAT][moves['c']-board_size], d12_start + 17) 
    eq_(white.tensor[F_D12_PAT][moves['c']-board_size+1], d12_start + 18) 
    eq_(white.tensor[F_D12_PAT][moves['c']-2], d12_start + 19) 
    eq_(white.tensor[F_D12_PAT][moves['c']-1], d12_start + 20) 
    eq_(white.tensor[F_D12_PAT][moves['c']+1], d12_start + 9) 
    eq_(white.tensor[F_D12_PAT][moves['c']+2], d12_start + 10) 
    eq_(white.tensor[F_D12_PAT][moves['c']+board_size-1], d12_start + 21) 
    eq_(white.tensor[F_D12_PAT][moves['c']+board_size], d12_start + 22) 
    eq_(white.tensor[F_D12_PAT][moves['c']+board_size+1], d12_start + 8) 
    eq_(white.tensor[F_D12_PAT][moves['c']+board_size*2], d12_start + 23) 
    
    eq_(white.tensor[F_D12_PAT][moves['b']-board_size*2], d12_start + 7) 
    eq_(white.tensor[F_D12_PAT][moves['b']-board_size-1], d12_start + 3) 
    eq_(white.tensor[F_D12_PAT][moves['b']-board_size], d12_start + 2) 
    eq_(white.tensor[F_D12_PAT][moves['b']-2], d12_start + 5) 
    eq_(white.tensor[F_D12_PAT][moves['b']-1], d12_start + 2) 

    eq_(white.tensor[F_D12_PAT][moves['a']-board_size-1], d12_start + 1) 
    eq_(white.tensor[F_D12_PAT][moves['a']-2], d12_start + 4) 
    eq_(white.tensor[F_D12_PAT][moves['a']-1], d12_start) 
    eq_(white.tensor[F_D12_PAT][moves['a']+2], d12_start + 6) 
    eq_(white.tensor[F_D12_PAT][moves['a']+board_size-1], d12_start + 8) 
    eq_(white.tensor[F_D12_PAT][moves['a']+board_size],   d12_start + 9) 
    eq_(white.tensor[F_D12_PAT][moves['a']+board_size+1], d12_start + 8) 
    eq_(white.tensor[F_D12_PAT][moves['a']+board_size*2], d12_start + 10) 
    eq_(number_of_active_positions(white, F_D12_PAT), 24)

    """
    # put W[d]
    put_stone(game, moves['d'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)
    # updated around move['c']
    eq_(black.tensor[F_D12_PAT][moves['c']-board_size-1], -1) 
    eq_(black.tensor[F_D12_PAT][moves['c']-board_size], d12_start + 6) 
    eq_(black.tensor[F_D12_PAT][moves['c']-board_size+1], d12_start + 2) 
    eq_(black.tensor[F_D12_PAT][moves['c']-1], -1) 
    eq_(black.tensor[F_D12_PAT][moves['c']+1], d12_start + 3) 
    eq_(black.tensor[F_D12_PAT][moves['c']+board_size-1], d12_start + 7) 
    eq_(black.tensor[F_D12_PAT][moves['c']+board_size], d12_start + 8) 
    eq_(black.tensor[F_D12_PAT][moves['c']+board_size+1], d12_start + 2) 
    # updated around move['d']
    eq_(black.tensor[F_D12_PAT][moves['b']-board_size-1], d12_start) 
    eq_(black.tensor[F_D12_PAT][moves['b']-board_size], d12_start + 1) 
    eq_(black.tensor[F_D12_PAT][moves['b']-board_size+1], d12_start) 
    eq_(black.tensor[F_D12_PAT][moves['b']-1], d12_start + 9) 
    eq_(black.tensor[F_D12_PAT][moves['b']+1], d12_start + 6) 
    eq_(black.tensor[F_D12_PAT][moves['d']-1], d12_start + 9) 
    eq_(black.tensor[F_D12_PAT][moves['d']+board_size-1], d12_start + 10) 
    eq_(black.tensor[F_D12_PAT][moves['d']+board_size], d12_start + 7) 
    eq_(black.tensor[F_D12_PAT][moves['d']+board_size+1], d12_start + 8) 
    eq_(number_of_active_positions(black, F_D12_PAT), 17)
    """
    print_board(game)

    free_game(game)


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

    eq_(black.tensor[F_SAVE_ATARI][moves['a']], save_atari_start) 
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

    eq_(white.tensor[F_SAVE_ATARI][moves['a']], save_atari_start) 
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

    eq_(black.tensor[F_SAVE_ATARI][moves['a']], -1) 
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

    eq_(white.tensor[F_SAVE_ATARI][moves['a']], -1) 
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
    eq_(white.tensor[F_NEIGHBOR][moves['a']-board_size-1], neighbor_start) 
    eq_(white.tensor[F_NEIGHBOR][moves['a']-board_size], neighbor_start+1) 
    eq_(white.tensor[F_NEIGHBOR][moves['a']-board_size+1], neighbor_start+2) 
    eq_(white.tensor[F_NEIGHBOR][moves['a']-1], neighbor_start+3) 
    eq_(white.tensor[F_NEIGHBOR][moves['a']+1], neighbor_start+4) 
    eq_(white.tensor[F_NEIGHBOR][moves['a']+board_size-1], neighbor_start+5) 
    eq_(white.tensor[F_NEIGHBOR][moves['a']+board_size], neighbor_start+6) 
    eq_(white.tensor[F_NEIGHBOR][moves['a']+board_size+1], neighbor_start+7) 
    eq_(number_of_active_positions(white, F_NEIGHBOR), 8)

    # put W[b]
    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)
    # updated around move['b'] 
    eq_(black.tensor[F_NEIGHBOR][moves['b']-board_size-1], neighbor_start) 
    eq_(black.tensor[F_NEIGHBOR][moves['b']-board_size], neighbor_start+1) 
    eq_(black.tensor[F_NEIGHBOR][moves['b']-board_size+1], neighbor_start+2) 
    eq_(black.tensor[F_NEIGHBOR][moves['b']-1], -1) 
    eq_(black.tensor[F_NEIGHBOR][moves['b']+1], neighbor_start+4) 
    eq_(black.tensor[F_NEIGHBOR][moves['b']+board_size-1], neighbor_start+5) 
    eq_(black.tensor[F_NEIGHBOR][moves['b']+board_size], neighbor_start+6) 
    eq_(black.tensor[F_NEIGHBOR][moves['b']+board_size+1], neighbor_start+7) 
    eq_(number_of_active_positions(black, F_NEIGHBOR), 7)

    # put B[c]
    put_stone(game, moves['c'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)
    # updated around move['c'] 
    eq_(white.tensor[F_NEIGHBOR][moves['c']-board_size-1], neighbor_start) 
    eq_(white.tensor[F_NEIGHBOR][moves['c']-board_size], neighbor_start+1) 
    eq_(white.tensor[F_NEIGHBOR][moves['c']-board_size+1], -1) 
    eq_(white.tensor[F_NEIGHBOR][moves['c']-1], -1) 
    eq_(white.tensor[F_NEIGHBOR][moves['c']+1], -1) 
    eq_(white.tensor[F_NEIGHBOR][moves['c']+board_size-1], neighbor_start+5) 
    eq_(white.tensor[F_NEIGHBOR][moves['c']+board_size], neighbor_start+6) 
    eq_(white.tensor[F_NEIGHBOR][moves['c']+board_size+1], -1) 
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


def test_update_12diamond_rspos_0():
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
    eq_(white.tensor[F_RESPONSE][moves['a']-2*board_size], response_start) 
    eq_(white.tensor[F_RESPONSE][moves['a']-board_size-1], response_start) 
    eq_(white.tensor[F_RESPONSE][moves['a']-board_size], response_start) 
    eq_(white.tensor[F_RESPONSE][moves['a']-board_size+1], response_start) 
    eq_(white.tensor[F_RESPONSE][moves['a']-2], response_start) 
    eq_(white.tensor[F_RESPONSE][moves['a']-1], response_start) 
    eq_(white.tensor[F_RESPONSE][moves['a']+1], response_start) 
    eq_(white.tensor[F_RESPONSE][moves['a']+2], response_start) 
    eq_(white.tensor[F_RESPONSE][moves['a']+board_size-1], response_start) 
    eq_(white.tensor[F_RESPONSE][moves['a']+board_size], response_start) 
    eq_(white.tensor[F_RESPONSE][moves['a']+board_size+1], response_start) 
    eq_(white.tensor[F_RESPONSE][moves['a']+2*board_size], response_start) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['a']-2*board_size], d12_rsp_start) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['a']-board_size-1], d12_rsp_start+1) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['a']-board_size], d12_rsp_start+2) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['a']-board_size+1], d12_rsp_start+1) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['a']-2], d12_rsp_start) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['a']-1], d12_rsp_start+2) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['a']+1], d12_rsp_start+2) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['a']+2], d12_rsp_start) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['a']+board_size-1], d12_rsp_start+1) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['a']+board_size], d12_rsp_start+2) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['a']+board_size+1], d12_rsp_start+1) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['a']+2*board_size], d12_rsp_start) 
    eq_(number_of_active_positions(white, F_RESPONSE), 12)
    eq_(number_of_active_positions(white, F_D12_RSP_PAT), 12)

    # put W[b]
    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)
    # updated around move['b']
    eq_(black.tensor[F_D12_RSP_PAT][moves['b']-2*board_size], d12_rsp_start) 
    eq_(black.tensor[F_D12_RSP_PAT][moves['b']-board_size-1], d12_rsp_start+1) 
    eq_(black.tensor[F_D12_RSP_PAT][moves['b']-board_size], d12_rsp_start+2) 
    eq_(black.tensor[F_D12_RSP_PAT][moves['b']-board_size+1], d12_rsp_start+1) 
    eq_(black.tensor[F_D12_RSP_PAT][moves['b']-2], d12_rsp_start) 
    eq_(black.tensor[F_D12_RSP_PAT][moves['b']-1], d12_rsp_start+2) 
    eq_(black.tensor[F_D12_RSP_PAT][moves['b']+1], d12_rsp_start+2) 
    eq_(black.tensor[F_D12_RSP_PAT][moves['b']+2], d12_rsp_start) 
    eq_(black.tensor[F_D12_RSP_PAT][moves['b']+board_size-1], d12_rsp_start+1) 
    eq_(black.tensor[F_D12_RSP_PAT][moves['b']+board_size], d12_rsp_start+2) 
    eq_(black.tensor[F_D12_RSP_PAT][moves['b']+board_size+1], d12_rsp_start+1) 
    eq_(black.tensor[F_D12_RSP_PAT][moves['b']+2*board_size], d12_rsp_start) 
    # no others
    eq_(number_of_active_positions(black, F_RESPONSE), 12)
    eq_(number_of_active_positions(black, F_D12_RSP_PAT), 12)

    # put B[b]
    put_stone(game, moves['c'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)
    # updated around move['c']
    eq_(white.tensor[F_D12_RSP_PAT][moves['c']-2*board_size], d12_rsp_start+13) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['c']-board_size-1], -1) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['c']-board_size], d12_rsp_start+12) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['c']-board_size+1], d12_rsp_start+11) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['c']-2], d12_rsp_start+10) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['c']-1], d12_rsp_start+9) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['c']+1], d12_rsp_start+8) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['c']+2], d12_rsp_start+7) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['c']+board_size-1], d12_rsp_start+6) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['c']+board_size], d12_rsp_start+5) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['c']+board_size+1], d12_rsp_start+4) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['c']+2*board_size], d12_rsp_start+3) 
    # no others
    eq_(number_of_active_positions(white, F_RESPONSE), 11)
    eq_(number_of_active_positions(white, F_D12_RSP_PAT), 11)

    # put W[d]
    put_stone(game, moves['d'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)
    # updated around move['d']
    eq_(black.tensor[F_D12_RSP_PAT][moves['d']-2*board_size], d12_rsp_start+22) 
    eq_(black.tensor[F_D12_RSP_PAT][moves['d']-board_size-1], d12_rsp_start+21) 
    eq_(black.tensor[F_D12_RSP_PAT][moves['d']-board_size], -1) 
    eq_(black.tensor[F_D12_RSP_PAT][moves['d']-board_size+1], d12_rsp_start+20) 
    eq_(black.tensor[F_D12_RSP_PAT][moves['d']-2], d12_rsp_start+19) 
    eq_(black.tensor[F_D12_RSP_PAT][moves['d']-1], d12_rsp_start+18) 
    eq_(black.tensor[F_D12_RSP_PAT][moves['d']+1], -1) 
    eq_(black.tensor[F_D12_RSP_PAT][moves['d']+2], d12_rsp_start+17) 
    eq_(black.tensor[F_D12_RSP_PAT][moves['d']+board_size-1], d12_rsp_start+16) 
    eq_(black.tensor[F_D12_RSP_PAT][moves['d']+board_size], d12_rsp_start+15) 
    eq_(black.tensor[F_D12_RSP_PAT][moves['d']+board_size+1], d12_rsp_start+14) 
    eq_(black.tensor[F_D12_RSP_PAT][moves['d']+2*board_size], -1) 
    # no others, previous positions are cleared
    eq_(number_of_active_positions(black, F_RESPONSE), 9)
    eq_(number_of_active_positions(black, F_D12_RSP_PAT), 9)

    # put B[e]
    put_stone(game, moves['e'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)
    # updated around move['e']
    eq_(white.tensor[F_D12_RSP_PAT][moves['e']-2*board_size], d12_rsp_start+31) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['e']-board_size-1], d12_rsp_start+30) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['e']-board_size], d12_rsp_start+29) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['e']-board_size+1], d12_rsp_start+28) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['e']-2], d12_rsp_start+27) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['e']-1], -1) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['e']+1], d12_rsp_start+26) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['e']+2], d12_rsp_start+25) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['e']+board_size-1], -1) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['e']+board_size], -1) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['e']+board_size+1], d12_rsp_start+24) 
    eq_(white.tensor[F_D12_RSP_PAT][moves['e']+2*board_size], d12_rsp_start+23) 
    # no others, previous positions are cleared
    eq_(number_of_active_positions(white, F_RESPONSE), 9)
    eq_(number_of_active_positions(white, F_D12_RSP_PAT), 9)

    # print_board(game)

    free_game(game)


def test_update_12diamond_rspos_after_pass_0():
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

    eq_(number_of_active_positions(white, F_D12_RSP_PAT), 0)

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
    eq_(white.tensor[F_X33_PAT][moves['a']-board_size-1], x33_start) 
    eq_(white.tensor[F_X33_PAT][moves['a']-board_size], x33_start+1) 
    eq_(white.tensor[F_X33_PAT][moves['a']-board_size+1], x33_start) 
    eq_(white.tensor[F_X33_PAT][moves['a']-1], x33_start+1) 
    eq_(white.tensor[F_X33_PAT][moves['a']+1], x33_start+1) 
    eq_(white.tensor[F_X33_PAT][moves['a']+board_size-1], x33_start) 
    eq_(white.tensor[F_X33_PAT][moves['a']+board_size], x33_start+1) 
    eq_(white.tensor[F_X33_PAT][moves['a']+board_size+1], x33_start) 
    eq_(number_of_active_positions(white, F_X33_PAT), 8)

    # put W[b]
    put_stone(game, moves['b'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)
    # updated around move['a'] 
    eq_(black.tensor[F_X33_PAT][moves['a']-board_size-1], x33_start + 2) 
    eq_(black.tensor[F_X33_PAT][moves['a']-board_size], x33_start + 3) 
    eq_(black.tensor[F_X33_PAT][moves['a']-board_size+1], x33_start + 2) 
    eq_(black.tensor[F_X33_PAT][moves['a']-1], x33_start + 3) 
    eq_(black.tensor[F_X33_PAT][moves['a']+1], x33_start + 3) 
    eq_(black.tensor[F_X33_PAT][moves['a']+board_size-1], x33_start + 2) 
    eq_(black.tensor[F_X33_PAT][moves['a']+board_size], x33_start + 3) 
    eq_(black.tensor[F_X33_PAT][moves['a']+board_size+1], x33_start + 2) 
    # updated around move['b']
    eq_(black.tensor[F_X33_PAT][moves['b']-board_size-1], x33_start) 
    eq_(black.tensor[F_X33_PAT][moves['b']-board_size], x33_start+1) 
    eq_(black.tensor[F_X33_PAT][moves['b']-board_size+1], x33_start) 
    eq_(black.tensor[F_X33_PAT][moves['b']-1], x33_start+1) 
    eq_(black.tensor[F_X33_PAT][moves['b']+1], x33_start+1) 
    eq_(black.tensor[F_X33_PAT][moves['b']+board_size-1], x33_start) 
    eq_(black.tensor[F_X33_PAT][moves['b']+board_size], x33_start+1) 
    eq_(black.tensor[F_X33_PAT][moves['b']+board_size+1], x33_start) 
    eq_(number_of_active_positions(black, F_X33_PAT), 16)

    # put B[c]
    put_stone(game, moves['c'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)
    # updated around move['b']
    eq_(white.tensor[F_X33_PAT][moves['b']-board_size-1], x33_start + 2) 
    eq_(white.tensor[F_X33_PAT][moves['b']-board_size], x33_start + 3) 
    eq_(white.tensor[F_X33_PAT][moves['b']-board_size+1], x33_start + 2) 
    eq_(white.tensor[F_X33_PAT][moves['b']-1], x33_start + 3) 
    eq_(white.tensor[F_X33_PAT][moves['b']+1], x33_start + 4) 
    eq_(white.tensor[F_X33_PAT][moves['b']+board_size-1], x33_start + 2) 
    eq_(white.tensor[F_X33_PAT][moves['b']+board_size], x33_start + 4) 
    eq_(white.tensor[F_X33_PAT][moves['b']+board_size+1], -1) 
    # updated around move['c']
    eq_(white.tensor[F_X33_PAT][moves['c']-board_size-1], -1) 
    eq_(white.tensor[F_X33_PAT][moves['c']-board_size], x33_start + 4) 
    eq_(white.tensor[F_X33_PAT][moves['c']-board_size+1], x33_start) 
    eq_(white.tensor[F_X33_PAT][moves['c']-1], x33_start + 4) 
    eq_(white.tensor[F_X33_PAT][moves['c']+1], x33_start + 1) 
    eq_(white.tensor[F_X33_PAT][moves['c']+board_size-1], x33_start + 5) 
    eq_(white.tensor[F_X33_PAT][moves['c']+board_size], x33_start + 5) 
    eq_(white.tensor[F_X33_PAT][moves['c']+board_size+1], x33_start) 
    eq_(number_of_active_positions(white, F_X33_PAT), 18)

    # put W[d]
    put_stone(game, moves['d'], game.current_color)
    game.current_color = FLIP_COLOR(game.current_color)
    update_planes(game)
    # updated around move['c']
    eq_(black.tensor[F_X33_PAT][moves['c']-board_size-1], -1) 
    eq_(black.tensor[F_X33_PAT][moves['c']-board_size], x33_start + 6) 
    eq_(black.tensor[F_X33_PAT][moves['c']-board_size+1], x33_start + 2) 
    eq_(black.tensor[F_X33_PAT][moves['c']-1], -1) 
    eq_(black.tensor[F_X33_PAT][moves['c']+1], x33_start + 3) 
    eq_(black.tensor[F_X33_PAT][moves['c']+board_size-1], x33_start + 7) 
    eq_(black.tensor[F_X33_PAT][moves['c']+board_size], x33_start + 8) 
    eq_(black.tensor[F_X33_PAT][moves['c']+board_size+1], x33_start + 2) 
    # updated around move['d']
    eq_(black.tensor[F_X33_PAT][moves['b']-board_size-1], x33_start) 
    eq_(black.tensor[F_X33_PAT][moves['b']-board_size], x33_start + 1) 
    eq_(black.tensor[F_X33_PAT][moves['b']-board_size+1], x33_start) 
    eq_(black.tensor[F_X33_PAT][moves['b']-1], x33_start + 9) 
    eq_(black.tensor[F_X33_PAT][moves['b']+1], x33_start + 6) 
    eq_(black.tensor[F_X33_PAT][moves['d']-1], x33_start + 9) 
    eq_(black.tensor[F_X33_PAT][moves['d']+board_size-1], x33_start + 10) 
    eq_(black.tensor[F_X33_PAT][moves['d']+board_size], x33_start + 7) 
    eq_(black.tensor[F_X33_PAT][moves['d']+board_size+1], x33_start + 8) 
    eq_(number_of_active_positions(black, F_X33_PAT), 17)

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

    set_board_size(19)

    initialize_probs(game)

    game.current_color = S_BLACK
    color = <int>S_BLACK

    game.rollout_logits[color][60] = 40
    game.rollout_logits[color][72] = 80
    game.rollout_logits[color][288] = 20
    game.rollout_logits[color][300] = 30
    game.rollout_logits_sum[color] = 170

    norm_probs(game.rollout_probs[color],
               game.rollout_row_probs[color],
               game.rollout_logits[color],
               game.rollout_logits_sum[color])

    for i in range(100):
        pos = choice_rollout_move(game)
        ok_(pos in (120, 132, 396, 408))

    set_illegal(game, 120)
    set_illegal(game, 132)
    set_illegal(game, 396)
    set_illegal(game, 408)

    eq_(choice_rollout_move(game), PASS)


def test_set_illegal():
    cdef game_state_t *game = allocate_game()
    cdef int pos, color 
    cdef int i

    set_board_size(19)

    initialize_probs(game)

    game.current_color = S_BLACK
    color = <int>S_BLACK

    game.rollout_logits[color][60] = 40
    game.rollout_logits[color][72] = 50
    game.rollout_logits[color][288] = 20
    game.rollout_logits[color][300] = 30
    game.rollout_logits_sum[color] = 140

    norm_probs(game.rollout_probs[color],
               game.rollout_row_probs[color],
               game.rollout_logits[color],
               game.rollout_logits_sum[color])

    set_illegal(game, 120)
    eq_(game.rollout_logits_sum[color], 100)
    eq_(round(game.rollout_probs[color][72] + game.rollout_probs[color][288] + game.rollout_probs[color][300]), 1)

    set_illegal(game, 132)
    eq_(game.rollout_logits_sum[color], 50)
    eq_(round(game.rollout_probs[color][288] + game.rollout_probs[color][300]), 1)

    set_illegal(game, 396)
    eq_(game.rollout_logits_sum[color], 30)
    eq_(round(game.rollout_probs[color][288] + game.rollout_probs[color][300]), 1)

    set_illegal(game, 408)
    eq_(game.rollout_logits_sum[color], 0)


def test_copy_game():
    cdef game_state_t *game = allocate_game()
    cdef game_state_t *copy = allocate_game()
    cdef int black, white 
    cdef int i

    set_board_size(19)

    initialize_probs(game)

    black = <int>S_BLACK
    white = <int>S_WHITE

    game.rollout_feature_planes[black].color = black
    game.rollout_feature_planes[black].tensor[0][0] = 1
    game.rollout_feature_planes[black].tensor[5][360] = 1
    game.rollout_feature_planes[black].prev_neighbor8[0] = 96
    game.rollout_feature_planes[black].prev_neighbor8[7] = 144
    game.rollout_feature_planes[black].prev_neighbor8_num = 8
    game.rollout_feature_planes[black].prev_d12[0] = 74
    game.rollout_feature_planes[black].prev_d12[11] = 143
    game.rollout_feature_planes[black].prev_d12_num = 12
    game.rollout_feature_planes[black].updated[0] = 120
    game.rollout_feature_planes[black].updated[528] = 132
    game.rollout_feature_planes[black].updated_num = 2

    game.rollout_feature_planes[white].color = white
    game.rollout_feature_planes[white].tensor[0][0] = 1
    game.rollout_feature_planes[white].tensor[5][360] = 1
    game.rollout_feature_planes[white].prev_neighbor8[0] = 108
    game.rollout_feature_planes[white].prev_neighbor8[7] = 156
    game.rollout_feature_planes[white].prev_neighbor8_num = 8
    game.rollout_feature_planes[white].prev_d12[0] = 88
    game.rollout_feature_planes[white].prev_d12[11] = 178
    game.rollout_feature_planes[white].prev_d12_num = 12
    game.rollout_feature_planes[white].updated[0] = 120
    game.rollout_feature_planes[white].updated[528] = 132
    game.rollout_feature_planes[white].updated_num = 2

    game.rollout_logits[black][0] = 1
    game.rollout_logits[white][360] = 2
    game.rollout_logits_sum[black] = 1
    game.rollout_logits_sum[white] = 2

    game.rollout_probs[black][0] = 1
    game.rollout_probs[white][360] = 2
    game.rollout_row_probs[black][0] = 1
    game.rollout_row_probs[white][18] = 2

    copy_game(copy, game)

    eq_(copy.rollout_feature_planes[black].color, black)
    eq_(copy.rollout_feature_planes[black].tensor[0][0], 1)
    eq_(copy.rollout_feature_planes[black].tensor[5][360], 1)
    eq_(copy.rollout_feature_planes[black].prev_neighbor8[0], 96)
    eq_(copy.rollout_feature_planes[black].prev_neighbor8[7], 144)
    eq_(copy.rollout_feature_planes[black].prev_neighbor8_num, 8)
    eq_(copy.rollout_feature_planes[black].prev_d12[0], 74)
    eq_(copy.rollout_feature_planes[black].prev_d12[11], 143)
    eq_(copy.rollout_feature_planes[black].prev_d12_num, 12)
    eq_(copy.rollout_feature_planes[black].updated[0], 120)
    eq_(copy.rollout_feature_planes[black].updated[528], 132)
    eq_(copy.rollout_feature_planes[black].updated_num, 2)

    eq_(copy.rollout_feature_planes[white].color, white)
    eq_(copy.rollout_feature_planes[white].tensor[0][0], 1)
    eq_(copy.rollout_feature_planes[white].tensor[5][360], 1)
    eq_(copy.rollout_feature_planes[white].prev_neighbor8[0], 108)
    eq_(copy.rollout_feature_planes[white].prev_neighbor8[7], 156)
    eq_(copy.rollout_feature_planes[white].prev_neighbor8_num, 8)
    eq_(copy.rollout_feature_planes[white].prev_d12[0], 88)
    eq_(copy.rollout_feature_planes[white].prev_d12[11], 178)
    eq_(copy.rollout_feature_planes[white].prev_d12_num, 12)
    eq_(copy.rollout_feature_planes[white].updated[0], 120)
    eq_(copy.rollout_feature_planes[white].updated[528], 132)
    eq_(copy.rollout_feature_planes[white].updated_num, 2)

    eq_(copy.rollout_logits[black][0], 1)
    eq_(copy.rollout_logits[white][360], 2)
    eq_(copy.rollout_logits_sum[black], 1)
    eq_(copy.rollout_logits_sum[white], 2)

    eq_(copy.rollout_probs[black][0], 1)
    eq_(copy.rollout_probs[white][360], 2)
    eq_(copy.rollout_row_probs[black][0], 1)
    eq_(copy.rollout_row_probs[white][18], 2)


cdef int number_of_active_positions(rollout_feature_t *feature, int feature_id):
    cdef int i
    cdef int n_active = 0
    for i in range(board_max):
        if feature.tensor[feature_id][i] >= 0:
            n_active += 1
    return n_active
