# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

from libc.stdio cimport printf

from libcpp.unordered_map cimport unordered_map

from bamboo.board cimport BOARD_MAX 
from bamboo.board cimport FLIP_COLOR 
from bamboo.board cimport board_size
from bamboo.board cimport game_state_t
from bamboo.zobrist_hash cimport shape_bit 


NAKADE_PATTERNS[0] = NAKADE_3
NAKADE_PATTERNS[1] = NAKADE_4
NAKADE_PATTERNS[2] = NAKADE_5
NAKADE_PATTERNS[3] = NAKADE_6

NAKADE_MAX_SIZE = NAKADE_5

start = BOARD_MAX / 2

cpdef int initialize_nakade_hash():
    cdef int[6][3] nakade3
    cdef int[4][4] nakade4
    cdef int[9][5] nakade5
    cdef int[4][6] nakade6
    cdef int i, j, k

    nakade3[0][0] = 0; nakade3[0][1] = 1;              nakade3[0][2] = 2;
    nakade3[1][0] = 0; nakade3[1][1] = board_size;     nakade3[1][2] = 2 * board_size;
    nakade3[2][0] = 0; nakade3[2][1] = 1;              nakade3[2][2] = board_size + 1;
    nakade3[3][0] = 0; nakade3[3][1] = board_size - 1; nakade3[3][2] = board_size;
    nakade3[4][0] = 0; nakade3[4][1] = board_size;     nakade3[4][2] = board_size + 1;
    nakade3[5][0] = 0; nakade3[5][1] = 1;              nakade3[5][2] = board_size;

    nakade4[0][0] = 0;              nakade4[0][1] = board_size - 1;
    nakade4[0][2] = board_size;     nakade4[0][3] = board_size + 1;
    nakade4[1][0] = 0;              nakade4[1][1] = board_size - 1;
    nakade4[1][2] = board_size;     nakade4[1][3] = 2 * board_size;
    nakade4[2][0] = 0;              nakade4[2][1] = board_size;
    nakade4[2][2] = board_size + 1; nakade4[2][3] = 2 * board_size;
    nakade4[3][0] = 0;              nakade4[3][1] = 1;
    nakade4[3][2] = 2;              nakade4[3][3] = board_size + 1;
    #nakade4[4][0] = 0;              nakade4[4][1] = 1;
    #nakade4[4][2] = board_size;     nakade4[4][3] = board_size + 1;

    nakade5[0][0] = 0;                  nakade5[0][1] = board_size - 1; nakade5[0][2] = board_size;
    nakade5[0][3] = board_size + 1;     nakade5[0][4] = 2 * board_size;
    nakade5[1][0] = 0;                  nakade5[1][1] = board_size - 1; nakade5[1][2] = board_size;
    nakade5[1][3] = 2 * board_size - 1; nakade5[1][4] = 2 * board_size;
    nakade5[2][0] = 0;                  nakade5[2][1] = 1;              nakade5[2][2] = board_size;
    nakade5[2][3] = board_size + 1;     nakade5[2][4] = board_size + 2;
    nakade5[3][0] = 0;                  nakade5[3][1] = 1;              nakade5[3][2] = board_size;
    nakade5[3][3] = board_size + 1;     nakade5[3][4] = 2 * board_size;
    nakade5[4][0] = 0;                  nakade5[4][1] = 1;              nakade5[4][2] = 2;
    nakade5[4][3] = board_size + 1;     nakade5[4][4] = board_size + 2;
    nakade5[5][0] = 0;                  nakade5[5][1] = board_size;     nakade5[5][2] = board_size + 1;
    nakade5[5][3] = 2 * board_size;     nakade5[5][4] = 2 * board_size + 1;
    nakade5[6][0] = 0;                  nakade5[6][1] = 1;              nakade5[6][2] = 2;
    nakade5[6][3] = board_size;         nakade5[6][4] = board_size + 1;
    nakade5[7][0] = 0;                  nakade5[7][1] = 1;              nakade5[7][2] = board_size;
    nakade5[7][3] = board_size + 1;     nakade5[7][4] = 2 * board_size + 1;
    nakade5[8][0] = 0;                  nakade5[8][1] = 1;              nakade5[8][2] = board_size - 1;
    nakade5[8][3] = board_size;         nakade5[8][4] = board_size + 1;

    nakade6[0][0] = 0;                  nakade6[0][1] = board_size - 1;
    nakade6[0][2] = board_size;         nakade6[0][3] = board_size + 1;
    nakade6[0][4] = 2 * board_size - 1; nakade6[0][5] = 2 * board_size;
    nakade6[1][0] = 0;                  nakade6[1][1] = 1;
    nakade6[1][2] = board_size;         nakade6[1][3] = board_size + 1;
    nakade6[1][4] = board_size + 2;     nakade6[1][5] = 2 * board_size + 1;
    nakade6[2][0] = 0;                  nakade6[2][1] = 1;
    nakade6[2][2] = board_size - 1;     nakade6[2][3] = board_size;
    nakade6[2][4] = board_size + 1;     nakade6[2][5] = 2 * board_size;
    nakade6[3][0] = 0;                  nakade6[3][1] = board_size - 1;
    nakade6[3][2] = board_size;         nakade6[3][3] = board_size + 1;
    nakade6[3][4] = 2 * board_size;     nakade6[3][5] = 2 * board_size + 1;

    nakade_pos[0][0] = 1;
    nakade_pos[0][1] = board_size;
    nakade_pos[0][2] = 1;
    nakade_pos[0][3] = board_size;
    nakade_pos[0][4] = board_size;
    nakade_pos[0][5] = 0;

    nakade_pos[1][0] = board_size;
    nakade_pos[1][1] = board_size;
    nakade_pos[1][2] = board_size;
    nakade_pos[1][3] = 1;
    #nakade_pos[1][4] = 0;

    nakade_pos[2][0] = board_size;
    nakade_pos[2][1] = board_size;
    nakade_pos[2][2] = board_size + 1;
    nakade_pos[2][3] = board_size;
    nakade_pos[2][4] = 1;
    nakade_pos[2][5] = board_size;
    nakade_pos[2][6] = 1;
    nakade_pos[2][7] = board_size + 1;
    nakade_pos[2][8] = board_size;

    nakade_pos[3][0] = board_size;
    nakade_pos[3][1] = board_size + 1;
    nakade_pos[3][2] = board_size;
    nakade_pos[3][3] = board_size;

    nakade_id[0][0] = 0
    nakade_id[0][1] = 0
    nakade_id[0][2] = 1
    nakade_id[0][3] = 1
    nakade_id[0][4] = 1
    nakade_id[0][5] = 1

    nakade_id[1][0] = 2
    nakade_id[1][1] = 2
    nakade_id[1][2] = 2
    nakade_id[1][3] = 2
    #nakade_id[1][4] = 3

    nakade_id[2][0] = 3
    nakade_id[2][1] = 4
    nakade_id[2][2] = 4
    nakade_id[2][3] = 4
    nakade_id[2][4] = 4
    nakade_id[2][5] = 4
    nakade_id[2][6] = 4
    nakade_id[2][7] = 4
    nakade_id[2][8] = 4

    nakade_id[3][0] = 5
    nakade_id[3][1] = 5
    nakade_id[3][2] = 5
    nakade_id[3][3] = 5

    # initialize nakade shape hash
    for i in range(NAKADE_3):
        nakade_hash[0][i] = 0
        for j in range(3):
            nakade_hash[0][i] ^= shape_bit[start + nakade3[i][j]]

    for i in range(NAKADE_4):
        nakade_hash[1][i] = 0
        for j in xrange(4):
            nakade_hash[1][i] ^= shape_bit[start + nakade4[i][j]]

    for i in range(NAKADE_5):
        nakade_hash[2][i] = 0
        for j in xrange(5):
            nakade_hash[2][i] ^= shape_bit[start + nakade5[i][j]]

    for i in range(NAKADE_6):
        nakade_hash[3][i] = 0
        for j in xrange(6):
            nakade_hash[3][i] ^= shape_bit[start + nakade6[i][j]]

    return 6


cdef int get_nakade_index(int capture_num, int *capture_pos) nogil:
    cdef unsigned long long hash = 0
    cdef int reviser
    cdef int i

    reviser = start - capture_pos[0]

    for i in range(capture_num):
        hash ^= shape_bit[capture_pos[i] + reviser]

    for i in range(NAKADE_PATTERNS[capture_num - 3]):
        if nakade_hash[capture_num - 3][i] == hash:
            return i

    return NOT_NAKADE


cdef int get_nakade_id(int capture_num, int nakade_index) nogil:
    return nakade_id[capture_num - 3][nakade_index]


cdef int get_nakade_pos(int capture_num, int *capture_pos, int nakade_index) nogil:
    return capture_pos[0] + nakade_pos[capture_num - 3][nakade_index]


cdef int nakade_at_captured_stone(game_state_t *game, int color) nogil:
    cdef int capture_num = game.capture_num[FLIP_COLOR(color)]
    cdef int *capture_pos = game.capture_pos[FLIP_COLOR(color)]
    cdef unsigned long long hash = 0
    cdef int reviser

    if capture_num < 3 or 6 < capture_num:
        return NOT_NAKADE

    reviser = start - capture_pos[0]

    for i in range(capture_num):
        hash ^= shape_bit[capture_pos[i] + reviser]

    for i in range(NAKADE_PATTERNS[capture_num - 3]):
        if nakade_hash[capture_num - 3][i] == hash:
            return capture_pos[0] + nakade_pos[capture_num - 3][i]

    return NOT_NAKADE
