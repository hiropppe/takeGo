from libc.stdio cimport printf

from libcpp.string cimport string as cppstring

from bamboo.go.board cimport MIN
from bamboo.go.board cimport S_EMPTY, S_BLACK, S_WHITE, S_OB
from bamboo.go.board cimport game_state_t, string_t, get_neighbor8
from bamboo.go.zobrist_hash cimport mt
from bamboo.go.printer cimport print_board


class IllegalState(Exception):
    pass


cdef void initialize_hash():
    cdef int i, j
    for i in range(8):
        for j in range(4):
            color_mt[i][j] = mt()
            liberty_mt[i][j] = mt()


cdef unsigned long long x33_hash(game_state_t *game, int pos, int color) except? -1:
    cdef int neighbor8[8]
    cdef int neighbor_pos
    cdef int string_id
    cdef string_t *string
    cdef unsigned long long hash_value = 0
    cdef int i

    get_neighbor8(neighbor8, pos)

    for i in range(8):
        neighbor_pos = neighbor8[i]
        hash_value |= color_mt[i][game.board[neighbor_pos]]
        string_id = game.string_id[neighbor_pos]
        if string_id:
            string = &game.string[string_id]
            hash_value |= liberty_mt[i][MIN(string.libs, 3)]

    return hash_value


cdef unsigned long long x33_hash_from_bits(unsigned long long bits) except? -1:
    pass


cdef unsigned long long x33_bits(game_state_t *game, int pos, int color) except? -1:
    cdef int neighbor8[8]
    cdef int neighbor_pos
    cdef int string_id
    cdef string_t *string
    cdef unsigned long long color_pat = 0 
    cdef int lib_pat = 0
    cdef int i

    get_neighbor8(neighbor8, pos)

    for i in range(8):
        neighbor_pos = neighbor8[i]
        color_pat |= (game.board[neighbor_pos] << i*2)
        string_id = game.string_id[neighbor_pos]
        if string_id:
            string = &game.string[string_id]
            lib_pat |= (MIN(string.libs, 3) << i*2)

    return (((color_pat << 16) | lib_pat) << 2) | color


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


cpdef void print_x33(unsigned long long pat3):
    buf = []
    stone = ['+', 'B', 'W', '#']
    color = ['?', 'x', 'o']
    liberty = [0, 1, 2, 3]
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


cdef int init_nakade(object nakade_file):
    cdef int nakade_size = 8192
    return nakade_size

cdef int init_x33(object x33_file):
    cdef int x33_size = 69338
    return x33_size

cdef int init_d12(object d12_file):
    cdef int d12_size = 32207
    return d12_size
