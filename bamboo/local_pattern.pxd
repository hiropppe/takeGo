from libcpp.unordered_map cimport unordered_map

from bamboo.board cimport game_state_t

cdef enum:
    x33_MAX = 69338
    d12_MAX = 32207

cdef unsigned long long color_mt[13][4]
cdef unsigned long long liberty_mt[13][4]
cdef unsigned long long player_mt[3]
cdef unsigned long long d12_pos_mt[2049]

cdef unordered_map[unsigned long long, int] x33_hashmap
cdef unordered_map[unsigned long long, int] d12_rsp_hashmap
cdef unordered_map[unsigned long long, int] d12_hashmap

cpdef void initialize_rands()
cpdef void read_rands(object mt_file)
cpdef void write_rands(object mt_file, int n=?)

cpdef int init_d12_rsp_hash(object d12_csv)
cpdef int init_d12_rspos_hash(object d12_csv)
cpdef int init_x33_hash(object x33_csv)
cpdef int init_d12_hash(object d12_csv)

cpdef void put_d12_rsp_hash(unsigned long long bits, int id)
cpdef void put_d12_rspos_hash(unsigned long long bits, int id)
cpdef void put_x33_hash(unsigned long long bits, int id)
cpdef void put_d12_hash(unsigned long long bits, int id)

# 12 diamond pattern
cdef unsigned long long d12_rsp_hash(game_state_t *game, int pos, int color,
                                 int empty_ix[12], int empty_pos[12], int *n_empty) nogil except? -1
cdef unsigned long long d12_rsp_bits(game_state_t *game, int pos, int color, 
                                 int empty_ix[12], int empty_pos[12], int *n_empty) nogil except? -1
cpdef unsigned long long d12_rsp_hash_from_bits(unsigned long long bits) except? -1

cpdef unsigned long long d12_rsp_trans8_min(unsigned long long pat)
cpdef unsigned long long d12_rsp_trans16_min(unsigned long long pat)

cdef void d12_rsp_trans8(unsigned long long pat, unsigned long long *trans)
cdef void d12_rsp_trans16(unsigned long long pat, unsigned long long *trans)

cpdef unsigned long long d12_rsp_rev(unsigned long long pat)
cpdef unsigned long long d12_rsp_rot90(unsigned long long pat)
cpdef unsigned long long d12_rsp_fliplr(unsigned long long pat)
cpdef unsigned long long d12_rsp_flipud(unsigned long long pat)
cpdef unsigned long long d12_rsp_transp(unsigned long long pat)

cpdef void print_d12_rsp(unsigned long long pat, bint show_bits=?, bint show_board=?)
cpdef void print_d12_rsp_trans8(unsigned long long pat, bint show_bits=?, bint show_board=?)
cpdef void print_d12_rsp_trans16(unsigned long long pat, bint show_bits=?, bint show_board=?)

# 12 diamond with response position pattern
cdef unsigned long long d12_rspos_hash(game_state_t *game, int pos, int color,
                                 int empty_ix[12], int empty_pos[12], int *n_empty) nogil except? -1
cdef unsigned long long d12_rspos_bits(game_state_t *game, int pos, int color, 
                                 int empty_ix[12], int empty_pos[12], int *n_empty) nogil except? -1
cpdef unsigned long long d12_rspos_hash_from_bits(unsigned long long bits) except? -1

cpdef unsigned long long d12_rspos_trans8_min(unsigned long long pat)
cpdef unsigned long long d12_rspos_trans16_min(unsigned long long pat)

cdef void d12_rspos_trans8(unsigned long long pat, unsigned long long *trans)
cdef void d12_rspos_trans16(unsigned long long pat, unsigned long long *trans)

cpdef unsigned long long d12_rspos_rev(unsigned long long pat)
cpdef unsigned long long d12_rspos_rot90(unsigned long long pat)
cpdef unsigned long long d12_rspos_fliplr(unsigned long long pat)
cpdef unsigned long long d12_rspos_flipud(unsigned long long pat)
cpdef unsigned long long d12_rspos_transp(unsigned long long pat)

cpdef void print_d12_rspos(unsigned long long pat, bint show_bits=?, bint show_board=?)
cpdef void print_d12_rspos_trans8(unsigned long long pat, bint show_bits=?, bint show_board=?)
cpdef void print_d12_rspos_trans16(unsigned long long pat, bint show_bits=?, bint show_board=?)

# non response 3x3 pattern
cdef unsigned long long x33_hash(game_state_t *game, int pos, int color) nogil except? -1
cdef unsigned long long x33_bits(game_state_t *game, int pos, int color) nogil except? -1 
cpdef unsigned long long x33_hash_from_bits(unsigned long long bits) except? -1

cpdef unsigned long long x33_trans8_min(unsigned long long pat)
cpdef unsigned long long x33_trans16_min(unsigned long long pat)

cdef void x33_trans8(unsigned long long pat, unsigned long long *trans)
cdef void x33_trans16(unsigned long long pat, unsigned long long *trans)

cpdef unsigned long long x33_rev(unsigned long long pat)
cpdef unsigned long long x33_rot90(unsigned long long pat)
cpdef unsigned long long x33_fliplr(unsigned long long pat)
cpdef unsigned long long x33_flipud(unsigned long long pat)
cpdef unsigned long long x33_transp(unsigned long long pat)

cpdef void print_x33(unsigned long long pat, bint show_bits=?, bint show_board=?)
cpdef void print_x33_trans8(unsigned long long pat, bint show_bits=?, bint show_board=?)
cpdef void print_x33_trans16(unsigned long long pat, bint show_bits=?, bint show_board=?)

# non response 12 diamond pattern
cdef unsigned long long d12_hash(game_state_t *game, int pos, int color) nogil except? -1
cdef unsigned long long d12_bits(game_state_t *game, int pos, int color) nogil except? -1
cpdef unsigned long long d12_hash_from_bits(unsigned long long bits) except? -1

cpdef unsigned long long d12_trans8_min(unsigned long long pat)
cpdef unsigned long long d12_trans16_min(unsigned long long pat)

cdef void d12_trans8(unsigned long long pat, unsigned long long *trans)
cdef void d12_trans16(unsigned long long pat, unsigned long long *trans)

cpdef unsigned long long d12_rev(unsigned long long pat)
cpdef unsigned long long d12_rot90(unsigned long long pat)
cpdef unsigned long long d12_fliplr(unsigned long long pat)
cpdef unsigned long long d12_flipud(unsigned long long pat)
cpdef unsigned long long d12_transp(unsigned long long pat)

cpdef void print_d12(unsigned long long pat, bint show_bits=?, bint show_board=?)
cpdef void print_d12_trans8(unsigned long long pat, bint show_bits=?, bint show_board=?)
cpdef void print_d12_trans16(unsigned long long pat, bint show_bits=?, bint show_board=?)
