from bamboo.go.board cimport game_state_t

cdef enum:
    x33_MAX = 69338
    d12_MAX = 32207

cdef enum:
    EMPTY
    BLACK
    WHITE
    OB

cdef unsigned long long hash_bit[8][7]

cdef void initialize_hash()

cdef unsigned long long x33_bit(game_state_t *game, int pos, int color) except? -1 

cdef void x33_trans8(unsigned long long pat, unsigned long long *trans)
cdef void x33_trans16(unsigned long long pat, unsigned long long *trans)
cpdef unsigned long long x33_rev(unsigned long long pat)
cpdef unsigned long long x33_rot90(unsigned long long pat)
cpdef unsigned long long x33_fliplr(unsigned long long pat)
cpdef unsigned long long x33_flipud(unsigned long long pat)
cpdef unsigned long long x33_transp(unsigned long long pat)

cpdef void print_x33(unsigned long long pat)

cdef int init_nakade(object nakade_file)
cdef int init_x33(object x33_file)
cdef int init_d12(object d12_file)
