from libcpp.unordered_map cimport unordered_map

from bamboo.go.board cimport game_state_t


cdef enum:
    NAKADE_3 = 6
    NAKADE_4 = 4
    NAKADE_5 = 9
    NAKADE_6 = 4
    NOT_NAKADE = -1

cdef int[4] NAKADE_PATTERNS

cdef int NAKADE_MAX_SIZE

cdef unsigned long long[4][9] nakade_hash

cdef int[4][9] nakade_pos
cdef int[4][9] nakade_id

cdef int start

cdef int[13] string_pat
cdef unordered_map[long long, int] string_hashmap


cpdef int initialize_nakade_hash()

cdef int get_nakade_index(int capture_num, int *capture_pos) nogil

cdef int get_nakade_id(int capture_num, int nakade_index) nogil

cdef int get_nakade_pos(int capture_num, int *capture_pos, int nakade_index) nogil

cdef int nakade_at_captured_stone(game_state_t *game, int color) nogil
