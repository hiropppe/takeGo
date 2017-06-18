import pandas as pd

from libc.stdio cimport printf

from libcpp.string cimport string as cppstring

from bamboo.go.board cimport MIN
from bamboo.go.board cimport S_EMPTY, S_BLACK, S_WHITE, S_OB
from bamboo.go.board cimport game_state_t, string_t, get_neighbor8_in_order
from bamboo.go.zobrist_hash cimport mt
from bamboo.go.printer cimport print_board


class IllegalState(Exception):
    pass


cpdef void initialize_hash():
    cdef int i, j
    for i in range(8):
        for j in range(4):
            color_mt[i][j] = mt()
            liberty_mt[i][j] = mt()

    player_mt[<int>S_BLACK] = mt()
    player_mt[<int>S_WHITE] = mt()

    """
    for i in range(8):
        for j in range(4):
            printf('color_mt[%d][%d] = %llu\n', i, j, color_mt[i][j])
    for i in range(8):
        for j in range(4):
            printf('liberty_mt[%d][%d] = %llu\n', i, j, liberty_mt[i][j])

    printf('BLACK_mt = %llu\n', player_mt[<int>S_BLACK])
    printf('WHITE_mt = %llu\n', player_mt[<int>S_WHITE])
    """


cpdef int init_nakade_hash(object nakade_csv):
    return 0 


cpdef int init_d12_hash(object d12_csv):
    return 0 


cpdef int init_x33_hash(object x33_csv):
    cdef unordered_map[unsigned long long, int] x33_id_map
    cdef int x33_id_max = 0
    cdef unsigned long long x33_hash, x33_min_hash
    x33_df = pd.read_csv(x33_csv, index_col=0)
    for bits, row in x33_df.iterrows():
        x33_hash = x33_hash_from_bits(bits)
        x33_min_hash = x33_hash_from_bits(row['min16'])
        if x33_id_map.find(x33_min_hash) == x33_id_map.end():
            x33_id_map[x33_min_hash] = x33_id_max
            x33_id_max += 1
        x33_hashmap[x33_hash] = x33_id_map[x33_min_hash]
    printf('3x3 pattern loaded. #%d\n', x33_id_max+1)
    return x33_id_max + 1


cpdef void put_nakade_hash(unsigned long long bits, int id):
    pass


cpdef void put_d12_hash(unsigned long long bits, int id):
    pass


cpdef void put_x33_hash(unsigned long long bits, int id):
    cdef unsigned long long hash
    hash = x33_hash_from_bits(bits)
    #printf('put hash:%llu of bits:%llu as %d\n', hash, bits, id)
    x33_hashmap[hash] = id


cdef unsigned long long x33_hash(game_state_t *game, int pos, int color) nogil except? -1:
    cdef int neighbor8[8]
    cdef int neighbor_pos
    cdef int string_id
    cdef string_t *string
    cdef unsigned long long x33_hash = 0
    cdef int i

    get_neighbor8_in_order(neighbor8, pos)

    for i in range(8):
        neighbor_pos = neighbor8[i]
        x33_hash ^= color_mt[i][game.board[neighbor_pos]]
        string_id = game.string_id[neighbor_pos]
        if string_id:
            string = &game.string[string_id]
            x33_hash ^= liberty_mt[i][MIN(string.libs, 3)]
        else:
            x33_hash ^= liberty_mt[i][0]

    return x33_hash ^ player_mt[color]


cpdef unsigned long long x33_hash_from_bits(unsigned long long bits) except? -1:
    cdef int i, j
    cdef unsigned long long x33_hash = 0
    for i in range(8):
        x33_hash ^= color_mt[i][bits >> (18+2*i) & 0x3]
        x33_hash ^= liberty_mt[i][bits >> (2+2*i) & 0x3]
    return x33_hash ^ player_mt[bits & 0x3]


cdef unsigned long long x33_bits(game_state_t *game, int pos, int color) except? -1:
    cdef int neighbor8[8]
    cdef int neighbor_pos
    cdef int string_id
    cdef string_t *string
    cdef unsigned long long color_pat = 0 
    cdef int lib_pat = 0
    cdef int i

    get_neighbor8_in_order(neighbor8, pos)

    for i in range(8):
        neighbor_pos = neighbor8[i]
        color_pat |= (game.board[neighbor_pos] << i*2)
        string_id = game.string_id[neighbor_pos]
        if string_id:
            string = &game.string[string_id]
            lib_pat |= (MIN(string.libs, 3) << i*2)

    return (((color_pat << 16) | lib_pat) << 2) | color


cpdef unsigned long long x33_trans8_min(unsigned long long pat):
    cdef unsigned long long trans[8]
    cdef unsigned long long min_pat
    cdef int i

    x33_trans8(pat, trans)

    min_pat = trans[0]
    for i in range(1, 8):
        if trans[i] < min_pat:
            min_pat = trans[i]

    return min_pat


cpdef unsigned long long x33_trans16_min(unsigned long long pat):
    cdef unsigned long long trans[16]
    cdef unsigned long long min_pat
    cdef int i

    x33_trans16(pat, trans)

    min_pat = trans[0]
    for i in range(1, 16):
        if trans[i] < min_pat:
            min_pat = trans[i]

    return min_pat


cdef void x33_trans8(unsigned long long pat, unsigned long long *trans):
    trans[0] = pat
    trans[1] = x33_rot90(pat)
    trans[2] = x33_rot90(trans[1])
    trans[3] = x33_rot90(trans[2])
    trans[4] = x33_fliplr(pat)
    trans[5] = x33_flipud(pat)
    trans[6] = x33_transp(pat)
    trans[7] = x33_fliplr(trans[1])


cdef void x33_trans16(unsigned long long pat, unsigned long long *trans):
    trans[0] = pat
    trans[1] = x33_rot90(pat)
    trans[2] = x33_rot90(trans[1])
    trans[3] = x33_rot90(trans[2])
    trans[4] = x33_fliplr(pat)
    trans[5] = x33_flipud(pat)
    trans[6] = x33_transp(pat)
    trans[7] = x33_fliplr(trans[1])
    trans[8] = x33_rev(trans[0])
    trans[9] = x33_rev(trans[1])
    trans[10] = x33_rev(trans[2])
    trans[11] = x33_rev(trans[3])
    trans[12] = x33_rev(trans[4])
    trans[13] = x33_rev(trans[5])
    trans[14] = x33_rev(trans[6])
    trans[15] = x33_rev(trans[7])


cpdef unsigned long long x33_rev(unsigned long long pat):
    return ((((pat >> 19) & 0x5555) | (((pat >> 18) & 0x5555) << 1)) << 18) | ((pat >> 2 & 0xffff) << 2) | (~pat & 0x3)


cpdef unsigned long long x33_rot90(unsigned long long pat):
    return ((pat & 0xC000C) << 10) | ((pat & 0x30303030) << 4) | ((pat & <unsigned long long>0xC0C0C0C0) >> 4) | ((pat & 0x3000300) << 6) | ((pat & 0xC000C00) >> 6) | ((pat & <unsigned long long>0x300030000) >> 10) | (pat & 0x3)


cpdef unsigned long long x33_fliplr(unsigned long long pat):
    return ((pat & 0x300C300C) << 4) | ((pat & <unsigned long long>0x300C300C0) >> 4) | ((pat & 0x3000300) << 2) | ((pat & 0xC000C00) >> 2) | (pat & <unsigned long long>0xC030C033)


cpdef unsigned long long x33_flipud(unsigned long long pat):
    return ((pat & 0xFC00FC) << 10) | ((pat & <unsigned long long>0x3F003F000) >> 10) | (pat & 0xF000F03)


cpdef unsigned long long x33_transp(unsigned long long pat):
    return ((pat & 0xC300C30) << 4) | ((pat & 0xC000C0) << 6) | ((pat & <unsigned long long>0xC300C300) >> 4) | ((pat & 0x30003000) >> 6) | (pat & <unsigned long long>0x3000F000F)


cpdef void print_x33(unsigned long long pat3, bint show_bits=True, bint show_board=True):
    buf = []
    stone = ['+', 'B', 'W', '#']
    color = ['?', 'x', 'o']
    liberty = [0, 1, 2, 3]
    if show_bits:
        buf.append("0b{:s}".format(bin(pat3)[2:].rjust(18, '0')))
    if show_board:
        if show_bits:
            buf.append("\n")
        buf.append("{:s}{:s}{:s}    {:d}{:d}{:d}\n".format(
            stone[(pat3 >> 18) & 0x3],
            stone[(pat3 >> 20) & 0x3],
            stone[(pat3 >> 22) & 0x3],
            liberty[(pat3 >> 2) & 0x3],
            liberty[(pat3 >> 4) & 0x3],
            liberty[(pat3 >> 6) & 0x3]
            ))
        buf.append("{:s}{:s}{:s}    {:d} {:d}\n".format(
            stone[(pat3 >> 24) & 0x3],
            color[pat3 & 0x3],
            stone[(pat3 >> 26) & 0x3],
            liberty[(pat3 >> 8) & 0x3],
            liberty[(pat3 >> 10) & 0x3]
            ))
        buf.append("{:s}{:s}{:s}    {:d}{:d}{:d}\n".format(
            stone[(pat3 >> 28) & 0x3],
            stone[(pat3 >> 30) & 0x3],
            stone[(pat3 >> 32) & 0x3],
            liberty[(pat3 >> 12) & 0x3],
            liberty[(pat3 >> 14) & 0x3],
            liberty[(pat3 >> 16) & 0x3]
            ))
    print(''.join(buf))


cpdef void print_x33_trans8(unsigned long long pat, bint show_bits=True, bint show_board=True):
    cdef unsigned long long trans[8]
    cdef unsigned long long tmp_pat
    cdef int i, j

    x33_trans16(pat, trans)

    for i in range(8):
        for j in range(i+1, 8):
            if trans[j] < trans[i]:
                tmp = trans[j]
                trans[j] = trans[i]
                trans[i] = tmp

    for i in range(8):
        print_x33(trans[i], show_bits, show_board)


cpdef void print_x33_trans16(unsigned long long pat, bint show_bits=True, bint show_board=True):
    cdef unsigned long long trans[16]
    cdef unsigned long long tmp_pat
    cdef int i, j

    x33_trans16(pat, trans)

    for i in range(16):
        for j in range(i+1, 16):
            if trans[j] < trans[i]:
                tmp = trans[j]
                trans[j] = trans[i]
                trans[i] = tmp

    for i in range(16):
        print_x33(trans[i], show_bits, show_board)
