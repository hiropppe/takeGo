from bamboo.go.board cimport game_state_t

cdef enum:
    x33_MAX = 69338
    d12_MAX = 32207

cdef enum:
    EMPTY
    BLACK1
    BLACK2
    BLACK3
    WHITE3
    WHITE2
    WHITE1
    OB

cdef unsigned long long hash_bit[8]

cdef unsigned int stone_bit_mask[255]

cdef void initialize_hash()

cdef unsigned int calculate_x33_bit(game_state_t *game, int pos, int color) except? -1 

cdef void print_x33(unsigned int pat)

cdef int init_nakade(object nakade_file)
cdef int init_x33(object x33_file)
cdef int init_d12(object d12_file)
