import numpy as np
import pandas as pd

cimport numpy as np

from libc.stdio cimport printf

from libcpp.string cimport string as cppstring

from bamboo.board cimport MIN
from bamboo.board cimport S_EMPTY, S_BLACK, S_WHITE, S_OB
from bamboo.board cimport game_state_t, string_t, neighbor8_seq_pos, md2_pos 
from bamboo.zobrist_hash cimport mt
from bamboo.printer cimport print_board


cpdef void initialize_rands():
    cdef int i, j
    for i in range(13):
        for j in range(4):
            color_mt[i][j] = mt()
            liberty_mt[i][j] = mt()

    player_mt[<int>S_BLACK] = mt()
    player_mt[<int>S_WHITE] = mt()

    for i in range(12):
        d12_pos_mt[1 << i] = mt()


cpdef void read_rands(object mt_file):
    cdef int i, j
    with open(mt_file, 'r') as mt_in:
        for i in range(13):
            for j in range(4):
                color_mt[i][j] = long(mt_in.next())
                liberty_mt[i][j] = long(mt_in.next())

        player_mt[<int>S_BLACK] = long(mt_in.next())
        player_mt[<int>S_WHITE] = long(mt_in.next())

        for i in range(12):
            d12_pos_mt[1 << i] = long(mt_in.next())


cpdef void write_rands(object mt_file, int n=118):
    cdef int i
    cdef unsigned long long v
    with open(mt_file, 'w') as mt_out:
        for i in range(n):
            mt_out.write(str(mt()))
            mt_out.write('\n')


cpdef int init_d12_hash(object d12_csv):
    cdef unordered_map[unsigned long long, int] id_map
    cdef int id_max = 0
    cdef unsigned long long hash, min_hash

    if not d12_csv:
        return 0

    df = pd.read_csv(d12_csv, dtype={'pat': np.uint64, 'min8': np.uint64, 'min16': np.uint64})
    for _, row in df.iterrows():
        hash = d12_hash_from_bits(row['pat'])
        min_hash = d12_hash_from_bits(row['min16'])
        if id_map.find(min_hash) == id_map.end():
            id_map[min_hash] = id_max
            id_max += 1
        d12_hashmap[hash] = id_map[min_hash]
    printf('Response 12 diamond pattern loaded. #%d\n', id_max+1)
    return id_max + 1


cpdef int init_d12_move_hash(object d12_csv):
    cdef unordered_map[unsigned long long, int] id_map
    cdef int id_max = 0
    cdef unsigned long long hash, min_hash

    if not d12_csv:
        return 0

    df = pd.read_csv(d12_csv, dtype={'pat': np.uint64, 'min8': np.uint64, 'min16': np.uint64})
    for _, row in df.iterrows():
        hash = d12_move_hash_from_bits(row['pat'])
        min_hash = d12_move_hash_from_bits(row['min16'])
        if id_map.find(min_hash) == id_map.end():
            id_map[min_hash] = id_max
            id_max += 1
        d12_hashmap[hash] = id_map[min_hash]
    printf('Response 12 diamond move pattern loaded. #%d\n', id_max+1)
    return id_max + 1


cpdef int init_x33_hash(object x33_csv):
    cdef unordered_map[unsigned long long, int] id_map
    cdef int id_max = 0
    cdef unsigned long long hash, min_hash

    if not x33_csv:
        return 0

    df = pd.read_csv(x33_csv, dtype={'pat': np.uint64, 'min8': np.uint64, 'min16': np.uint64})
    for _, row in df.iterrows():
        hash = x33_hash_from_bits(row['pat'])
        min_hash = x33_hash_from_bits(row['min16'])
        if id_map.find(min_hash) == id_map.end():
            id_map[min_hash] = id_max
            id_max += 1
        x33_hashmap[hash] = id_map[min_hash]
    printf('Non-response 3x3 pattern loaded. #%d\n', id_max+1)
    return id_max + 1


cpdef int init_nonres_d12_hash(object nonres_d12_csv):
    cdef unordered_map[unsigned long long, int] id_map
    cdef int id_max = 0
    cdef unsigned long long hash, min_hash

    if not nonres_d12_csv:
        return 0

    df = pd.read_csv(nonres_d12_csv, dtype={'pat': np.uint64, 'min8': np.uint64, 'min16': np.uint64})
    for _, row in df.iterrows():
        hash = d12_hash_from_bits(row['pat'])
        min_hash = d12_hash_from_bits(row['min16'])
        if id_map.find(min_hash) == id_map.end():
            id_map[min_hash] = id_max
            id_max += 1
        nonres_d12_hashmap[hash] = id_map[min_hash]
    printf('Non-response 12 diamond pattern loaded. #%d\n', id_max+1)
    return id_max + 1


cpdef void put_d12_hash(unsigned long long bits, int id):
    cdef unsigned long long hash
    hash = d12_hash_from_bits(bits)
    d12_hashmap[hash] = id


cpdef void put_d12_move_hash(unsigned long long bits, int id):
    cdef unsigned long long hash
    hash = d12_move_hash_from_bits(bits)
    d12_hashmap[hash] = id


cpdef void put_x33_hash(unsigned long long bits, int id):
    cdef unsigned long long hash
    hash = x33_hash_from_bits(bits)
    x33_hashmap[hash] = id


cpdef void put_nonres_d12_hash(unsigned long long bits, int id):
    cdef unsigned long long hash
    hash = nonres_d12_hash_from_bits(bits)
    nonres_d12_hashmap[hash] = id


""" 12 diamond(MD2) Pattern functions
"""
cdef unsigned long long d12_hash(game_state_t *game, int pos, int color,
        int empty_ix[12], int empty_pos[12], int *n_empty) nogil except? -1:
    """ 12 diamond color and liberty hash without candidate move position
        Add candidate move by hash ^= d12_pos_mt[1 << i]
        i is candidate(empty) position index in 12 diamond 
    """
    cdef int md_pos, md_color
    cdef string_t *string
    cdef unsigned long long hash = 0
    cdef int i

    n_empty[0] = 0

    string = &game.string[game.string_id[pos]]
    hash ^= color_mt[0][color]
    hash ^= liberty_mt[0][MIN(string.libs, 3)]

    for i in range(1, 13):
        md_pos = md2_pos[pos][i-1]
        md_color = game.board[md_pos]
        hash ^= color_mt[i][md_color]
        string = &game.string[game.string_id[md_pos]]
        if string.flag:
            hash ^= liberty_mt[i][MIN(string.libs, 3)]
        else:
            hash ^= liberty_mt[i][0]

        # memorize empty position for update positional bits or hash
        if md_color == S_EMPTY:
            empty_ix[n_empty[0]] = i-1
            empty_pos[n_empty[0]] = md_pos
            n_empty[0] += 1

    return hash


cpdef unsigned long long d12_hash_from_bits(unsigned long long bits) except? -1:
    """ 12 diamond color and liberty hash with candidate move position
    """
    cdef int i
    cdef unsigned long long hash = 0
    for i in range(13):
        hash ^= color_mt[i][bits >> (26+2*i) & 0x3]
        hash ^= liberty_mt[i][bits >> (2*i) & 0x3]
    return hash


cdef unsigned long long d12_bits(game_state_t *game, int pos, int color,
        int empty_ix[12], int empty_pos[12], int *n_empty) nogil except? -1:
    """ 12 diamond color and liberty bits without candidate move position.
        Add candidate move by bits | (1 << i).
        i is candidate(empty) position index in 12 diamond 
    """
    cdef int md_pos, md_color
    cdef string_t *string
    cdef unsigned long long color_pat = 0
    cdef int lib_pat = 0
    cdef int i

    n_empty[0] = 0

    color_pat |= color
    string = &game.string[game.string_id[pos]]
    lib_pat |= MIN(string.libs, 3)

    for i in range(12):
        md_pos = md2_pos[pos][i]
        md_color = game.board[md_pos]
        color_pat |= (md_color << (i+1)*2)
        string = &game.string[game.string_id[md_pos]]
        if string.flag:
            lib_pat |= (MIN(string.libs, 3) << (i+1)*2)

        # memorize empty position for update positional bits or hash
        if md_color == S_EMPTY:
            empty_ix[n_empty[0]] = i
            empty_pos[n_empty[0]] = md_pos
            n_empty[0] += 1

    return ((color_pat << 26) | lib_pat)


cpdef unsigned long long d12_trans8_min(unsigned long long pat):
    cdef unsigned long long trans[8]
    cdef unsigned long long min_pat
    cdef int i

    d12_trans8(pat, trans)

    min_pat = trans[0]
    for i in range(1, 8):
        if trans[i] < min_pat:
            min_pat = trans[i]

    return min_pat


cpdef unsigned long long d12_trans16_min(unsigned long long pat):
    cdef unsigned long long trans[16]
    cdef unsigned long long min_pat
    cdef int i

    d12_trans16(pat, trans)

    min_pat = trans[0]
    for i in range(1, 16):
        if trans[i] < min_pat:
            min_pat = trans[i]

    return min_pat


cdef void d12_trans8(unsigned long long pat, unsigned long long *trans):
    trans[0] = pat
    trans[1] = d12_rot90(pat)
    trans[2] = d12_rot90(trans[1])
    trans[3] = d12_rot90(trans[2])
    trans[4] = d12_fliplr(pat)
    trans[5] = d12_flipud(pat)
    trans[6] = d12_transp(pat)
    trans[7] = d12_fliplr(trans[1])


cdef void d12_trans16(unsigned long long pat, unsigned long long *trans):
    trans[0] = pat
    trans[1] = d12_rot90(pat)
    trans[2] = d12_rot90(trans[1])
    trans[3] = d12_rot90(trans[2])
    trans[4] = d12_fliplr(pat)
    trans[5] = d12_flipud(pat)
    trans[6] = d12_transp(pat)
    trans[7] = d12_fliplr(trans[1])
    trans[8] = d12_rev(trans[0])
    trans[9] = d12_rev(trans[1])
    trans[10] = d12_rev(trans[2])
    trans[11] = d12_rev(trans[3])
    trans[12] = d12_rev(trans[4])
    trans[13] = d12_rev(trans[5])
    trans[14] = d12_rev(trans[6])
    trans[15] = d12_rev(trans[7])


cpdef unsigned long long d12_rev(unsigned long long pat):
    return ((((pat >> 27) & 0x1555555) | (((pat >> 26) & 0x1555555) << 1)) << 26) | (pat & 0x3ffffff)


cpdef unsigned long long d12_rot90(unsigned long long pat):
    return (((pat & <unsigned long long>0xc03000300c) << 8) |
            ((pat & <unsigned long long>0x30c0000c30) << 14) |
            ((pat & <unsigned long long>0x3000000c0) << 6) |
            ((pat & <unsigned long long>0xc00000300) >> 4) |
            ((pat & <unsigned long long>0xc03000300c000) >> 8) |
            ((pat & <unsigned long long>0x30c0000c30000) >> 14) |
            ((pat & <unsigned long long>0x3000000c0000) << 4) |
            ((pat & <unsigned long long>0xc00000300000) >> 6) |
            (pat & <unsigned long long>0xc000003))


cpdef unsigned long long d12_fliplr(unsigned long long pat):
    return (((pat & <unsigned long long>0x3000c00c0030) << 4) |
            ((pat & <unsigned long long>0x3000c00c00300) >> 4) |
            ((pat & <unsigned long long>0x3000000c00) << 6) |
            ((pat & <unsigned long long>0xc000003000) << 2) |
            ((pat & <unsigned long long>0x3000000c000) >> 2) |
            ((pat & <unsigned long long>0xc0000030000) >> 6) |
            (pat & <unsigned long long>0xcc0033f3000cf))


cpdef unsigned long long d12_flipud(unsigned long long pat):
    return (((pat & <unsigned long long>0x3000000c) << 22) |
            ((pat & <unsigned long long>0xfc00003f0) << 14) |
            ((pat & <unsigned long long>0x3f00000fc0000) >> 14) |
            ((pat & <unsigned long long>0xc000003000000) >> 22) |
            (pat & <unsigned long long>0xff00c03fc03))


cpdef unsigned long long d12_transp(unsigned long long pat):
    return (((pat & <unsigned long long>0xc003003000c) << 8) |
            ((pat & <unsigned long long>0x3030000c0c0) << 6) |
            ((pat & <unsigned long long>0xc00000300) << 10) |
            ((pat & <unsigned long long>0xc003003000c00) >> 8) |
            ((pat & <unsigned long long>0xc0c000303000) >> 6) |
            ((pat & <unsigned long long>0x3000000c0000) >> 10) |
            (pat & <unsigned long long>0x30000ccc00033))


cpdef void print_d12(unsigned long long pat3, bint show_bits=True, bint show_board=True):
    buf = []
    stone = ['+', 'B', 'W', '#']
    liberty = [0, 1, 2, 3]
    if show_bits:
        buf.append("0b{:s}".format(bin(pat3)[2:].rjust(52, '0')))
    if show_board:
        if show_bits:
            buf.append("\n")
        buf.append("  {:s}       {:d} \n".format(
            stone[(pat3 >> 28) & 0x3],
            liberty[(pat3 >> 2) & 0x3],
            ))
        buf.append(" {:s}{:s}{:s}     {:d}{:d}{:d}\n".format(
            stone[(pat3 >> 30) & 0x3],
            stone[(pat3 >> 32) & 0x3],
            stone[(pat3 >> 34) & 0x3],
            liberty[(pat3 >> 4) & 0x3],
            liberty[(pat3 >> 6) & 0x3],
            liberty[(pat3 >> 8) & 0x3]
            ))
        buf.append("{:s}{:s}{:s}{:s}{:s}   {:d}{:d}{:d}{:d}{:d}\n".format(
            stone[(pat3 >> 36) & 0x3],
            stone[(pat3 >> 38) & 0x3],
            stone[(pat3 >> 26) & 0x3],
            stone[(pat3 >> 40) & 0x3],
            stone[(pat3 >> 42) & 0x3],
            liberty[(pat3 >> 10) & 0x3],
            liberty[(pat3 >> 12) & 0x3],
            liberty[pat3 & 0x3],
            liberty[(pat3 >> 14) & 0x3],
            liberty[(pat3 >> 16) & 0x3]
            ))
        buf.append(" {:s}{:s}{:s}     {:d}{:d}{:d}\n".format(
            stone[(pat3 >> 44) & 0x3],
            stone[(pat3 >> 46) & 0x3],
            stone[(pat3 >> 48) & 0x3],
            liberty[(pat3 >> 18) & 0x3],
            liberty[(pat3 >> 20) & 0x3],
            liberty[(pat3 >> 22) & 0x3]
            ))
        buf.append("  {:s}       {:d} \n".format(
            stone[(pat3 >> 50) & 0x3],
            liberty[(pat3 >> 24) & 0x3]
            ))
    print(''.join(buf))


cpdef void print_d12_trans8(unsigned long long pat, bint show_bits=True, bint show_board=True):
    cdef unsigned long long trans[8]
    cdef unsigned long long tmp_pat
    cdef int i, j

    d12_trans16(pat, trans)

    for i in range(8):
        for j in range(i+1, 8):
            if trans[j] < trans[i]:
                tmp = trans[j]
                trans[j] = trans[i]
                trans[i] = tmp

    for i in range(8):
        print_d12(trans[i], show_bits, show_board)


cpdef void print_d12_trans16(unsigned long long pat, bint show_bits=True, bint show_board=True):
    cdef unsigned long long trans[16]
    cdef unsigned long long tmp_pat
    cdef int i, j

    d12_trans16(pat, trans)

    for i in range(16):
        for j in range(i+1, 16):
            if trans[j] < trans[i]:
                tmp = trans[j]
                trans[j] = trans[i]
                trans[i] = tmp

    for i in range(16):
        print_d12(trans[i], show_bits, show_board)


""" 12 diamond(MD2) move Pattern functions
"""
cdef unsigned long long d12_move_hash(game_state_t *game, int pos, int color,
                                      int empty_ix[12], int empty_pos[12], int *n_empty) nogil except? -1:
    """ 12 diamond color and liberty hash without candidate move position
        Add candidate move by hash ^= d12_pos_mt[1 << i]
        i is candidate(empty) position index in 12 diamond 
    """
    cdef int md_pos, md_color
    cdef string_t *string
    cdef unsigned long long hash = 0
    cdef int i

    n_empty[0] = 0

    string = &game.string[game.string_id[pos]]
    hash ^= color_mt[0][color]
    hash ^= liberty_mt[0][MIN(string.libs, 3)]

    for i in range(1, 13):
        md_pos = md2_pos[pos][i-1]
        md_color = game.board[md_pos]
        hash ^= color_mt[i][md_color]
        string = &game.string[game.string_id[md_pos]]
        if string.flag:
            hash ^= liberty_mt[i][MIN(string.libs, 3)]
        else:
            hash ^= liberty_mt[i][0]

        # memorize empty position for update positional bits or hash
        if md_color == S_EMPTY:
            empty_ix[n_empty[0]] = i-1
            empty_pos[n_empty[0]] = md_pos
            n_empty[0] += 1

    return hash


cpdef unsigned long long d12_move_hash_from_bits(unsigned long long bits) except? -1:
    """ 12 diamond color and liberty hash with candidate move position
    """
    cdef int i
    cdef unsigned long long hash = 0
    for i in range(13):
        hash ^= color_mt[i][bits >> (38+2*i) & 0x3]
        hash ^= liberty_mt[i][bits >> (12+2*i) & 0x3]
    return hash ^ d12_pos_mt[bits & 0xfff]


cdef unsigned long long d12_move_bits(game_state_t *game, int pos, int color,
                                 int empty_ix[12], int empty_pos[12], int *n_empty) nogil except? -1:
    """ 12 diamond color and liberty bits without candidate move position.
        Add candidate move by bits | (1 << i).
        i is candidate(empty) position index in 12 diamond 
    """
    cdef int md_pos, md_color
    cdef string_t *string
    cdef unsigned long long color_pat = 0
    cdef int lib_pat = 0
    cdef int i

    n_empty[0] = 0

    color_pat |= color
    string = &game.string[game.string_id[pos]]
    lib_pat |= MIN(string.libs, 3)

    for i in range(12):
        md_pos = md2_pos[pos][i]
        md_color = game.board[md_pos]
        color_pat |= (md_color << (i+1)*2)
        string = &game.string[game.string_id[md_pos]]
        if string.flag:
            lib_pat |= (MIN(string.libs, 3) << (i+1)*2)

        # memorize empty position for update positional bits or hash
        if md_color == S_EMPTY:
            empty_ix[n_empty[0]] = i
            empty_pos[n_empty[0]] = md_pos
            n_empty[0] += 1

    return ((color_pat << 26) | lib_pat) << 12


cpdef unsigned long long d12_move_trans8_min(unsigned long long pat):
    cdef unsigned long long trans[8]
    cdef unsigned long long min_pat
    cdef int i

    d12_trans8(pat, trans)

    min_pat = trans[0]
    for i in range(1, 8):
        if trans[i] < min_pat:
            min_pat = trans[i]

    return min_pat


cpdef unsigned long long d12_move_trans16_min(unsigned long long pat):
    cdef unsigned long long trans[16]
    cdef unsigned long long min_pat
    cdef int i

    d12_trans16(pat, trans)

    min_pat = trans[0]
    for i in range(1, 16):
        if trans[i] < min_pat:
            min_pat = trans[i]

    return min_pat


cdef void d12_move_trans8(unsigned long long pat, unsigned long long *trans):
    trans[0] = pat
    trans[1] = d12_move_rot90(pat)
    trans[2] = d12_move_rot90(trans[1])
    trans[3] = d12_move_rot90(trans[2])
    trans[4] = d12_move_fliplr(pat)
    trans[5] = d12_move_flipud(pat)
    trans[6] = d12_move_transp(pat)
    trans[7] = d12_move_fliplr(trans[1])


cdef void d12_move_trans16(unsigned long long pat, unsigned long long *trans):
    trans[0] = pat
    trans[1] = d12_move_rot90(pat)
    trans[2] = d12_move_rot90(trans[1])
    trans[3] = d12_move_rot90(trans[2])
    trans[4] = d12_move_fliplr(pat)
    trans[5] = d12_move_flipud(pat)
    trans[6] = d12_move_transp(pat)
    trans[7] = d12_move_fliplr(trans[1])
    trans[8] = d12_move_rev(trans[0])
    trans[9] = d12_move_rev(trans[1])
    trans[10] = d12_move_rev(trans[2])
    trans[11] = d12_move_rev(trans[3])
    trans[12] = d12_move_rev(trans[4])
    trans[13] = d12_move_rev(trans[5])
    trans[14] = d12_move_rev(trans[6])
    trans[15] = d12_move_rev(trans[7])


cpdef unsigned long long d12_move_rev(unsigned long long pat):
    return ((((pat >> 39) & 0x1555555) | (((pat >> 38) & 0x1555555) << 1)) << 38) | ((pat >> 12 & 0x3ffffff) << 12) | (pat & 0xfff)


cpdef unsigned long long d12_move_rot90(unsigned long long pat):
    return (((pat & <unsigned long long>0xc03000300c000) << 8) |
            ((pat & <unsigned long long>0x30c0000c30000) << 14) |
            ((pat & <unsigned long long>0x3000000c0000) << 6) |
            ((pat & <unsigned long long>0xc00000300000) >> 4) |
            ((pat & <unsigned long long>0xc03000300c000000) >> 8) |
            ((pat & <unsigned long long>0x30c0000c30000000) >> 14) |
            ((pat & <unsigned long long>0x3000000c0000000) << 4) |
            ((pat & <unsigned long long>0xc00000300000000) >> 6) |
            (pat & <unsigned long long>0xc000003000) |
            ((pat & <unsigned long long>0x21) << 4) |
            ((pat & <unsigned long long>0x12) << 7) |
            ((pat & <unsigned long long>0x4) << 3) |
            ((pat & <unsigned long long>0x8) >> 2) |
            ((pat & <unsigned long long>0x840) >> 4) |
            ((pat & <unsigned long long>0x480) >> 7) |
            ((pat & <unsigned long long>0x100) << 2) |
            ((pat & <unsigned long long>0x200) >> 3))


cpdef unsigned long long d12_move_fliplr(unsigned long long pat):
    return (((pat & <unsigned long long>0x3000c00c0030000) << 4) |
            ((pat & <unsigned long long>0x3000c00c00300000) >> 4) |
            ((pat & <unsigned long long>0x3000000c00000) << 6) |
            ((pat & <unsigned long long>0xc000003000000) << 2) |
            ((pat & <unsigned long long>0x3000000c000000) >> 2) |
            ((pat & <unsigned long long>0xc0000030000000) >> 6) |
            (pat & <unsigned long long>0xcc0033f3000cfa05) |
            ((pat & <unsigned long long>0x102) << 2) |
            ((pat & <unsigned long long>0x408) >> 2) |
            ((pat & <unsigned long long>0x10) << 3) |
            ((pat & <unsigned long long>0x20) << 1) |
            ((pat & <unsigned long long>0x40) >> 1) |
            ((pat & <unsigned long long>0x80) >> 3))


cpdef unsigned long long d12_move_flipud(unsigned long long pat):
    return (((pat & <unsigned long long>0x3000000c000) << 22) |
            ((pat & <unsigned long long>0xfc00003f0000) << 14) |
            ((pat & <unsigned long long>0x3f00000fc0000000) >> 14) |
            ((pat & <unsigned long long>0xc000003000000000) >> 22) |
            (pat & <unsigned long long>0xff00c03fc030f0) |
            ((pat & <unsigned long long>0x1) << 11) |
            ((pat & <unsigned long long>0xe) << 7) |
            ((pat & <unsigned long long>0x700) >> 7) |
            ((pat & <unsigned long long>0x800) >> 11))


cpdef unsigned long long d12_move_transp(unsigned long long pat):
    return (((pat & <unsigned long long>0xc003003000c000) << 8) |
            ((pat & <unsigned long long>0x3030000c0c0000) << 6) |
            ((pat & <unsigned long long>0xc00000300000) << 10) |
            ((pat & <unsigned long long>0xc003003000c00000) >> 8) |
            ((pat & <unsigned long long>0xc0c000303000000) >> 6) |
            ((pat & <unsigned long long>0x3000000c0000000) >> 10) |
            (pat & <unsigned long long>0x30000ccc00033402) |
            ((pat & <unsigned long long>0x81) << 4) |
            ((pat & <unsigned long long>0x44) << 3) |
            ((pat & <unsigned long long>0x8) << 5) |
            ((pat & <unsigned long long>0x810) >> 4) |
            ((pat & <unsigned long long>0x220) >> 3) |
            ((pat & <unsigned long long>0x100) >> 5))


cpdef void print_d12_move(unsigned long long pat3, bint show_bits=True, bint show_board=True):
    buf = []
    stone = ['+', 'B', 'W', '#']
    liberty = [0, 1, 2, 3]
    if show_bits:
        buf.append("0b{:s}".format(bin(pat3)[2:].rjust(64, '0')))
    if show_board:
        if show_bits:
            buf.append("\n")
        buf.append("  {:s}       {:d} \n".format(
            '*' if (pat3 & 1) else stone[(pat3 >> 28 >> 12) & 0x3],
            liberty[(pat3 >> 2 >> 12) & 0x3],
            ))
        buf.append(" {:s}{:s}{:s}     {:d}{:d}{:d}\n".format(
            '*' if (pat3 & 2) else stone[(pat3 >> 30 >> 12) & 0x3],
            '*' if (pat3 & 4) else stone[(pat3 >> 32 >> 12) & 0x3],
            '*' if (pat3 & 8) else stone[(pat3 >> 34 >> 12) & 0x3],
            liberty[(pat3 >> 4 >> 12) & 0x3],
            liberty[(pat3 >> 6 >> 12) & 0x3],
            liberty[(pat3 >> 8 >> 12) & 0x3]
            ))
        buf.append("{:s}{:s}{:s}{:s}{:s}   {:d}{:d}{:d}{:d}{:d}\n".format(
            '*' if (pat3 & 16) else stone[(pat3 >> 36 >> 12) & 0x3],
            '*' if (pat3 & 32) else stone[(pat3 >> 38 >> 12) & 0x3],
            stone[(pat3 >> 26 >> 12) & 0x3],
            '*' if (pat3 & 64) else stone[(pat3 >> 40 >> 12) & 0x3],
            '*' if (pat3 & 128) else stone[(pat3 >> 42 >> 12) & 0x3],
            liberty[(pat3 >> 10 >> 12) & 0x3],
            liberty[(pat3 >> 12 >> 12) & 0x3],
            liberty[pat3 >> 12 & 0x3],
            liberty[(pat3 >> 14 >> 12) & 0x3],
            liberty[(pat3 >> 16 >> 12) & 0x3]
            ))
        buf.append(" {:s}{:s}{:s}     {:d}{:d}{:d}\n".format(
            '*' if (pat3 & 256) else stone[(pat3 >> 44 >> 12) & 0x3],
            '*' if (pat3 & 512) else stone[(pat3 >> 46 >> 12) & 0x3],
            '*' if (pat3 & 1024) else stone[(pat3 >> 48 >> 12) & 0x3],
            liberty[(pat3 >> 18 >> 12) & 0x3],
            liberty[(pat3 >> 20 >> 12) & 0x3],
            liberty[(pat3 >> 22 >> 12) & 0x3]
            ))
        buf.append("  {:s}       {:d} \n".format(
            '*' if (pat3 & 2048) else stone[(pat3 >> 50 >> 12) & 0x3],
            liberty[(pat3 >> 24 >> 12) & 0x3]
            ))
    print(''.join(buf))


cpdef void print_d12_move_trans8(unsigned long long pat, bint show_bits=True, bint show_board=True):
    cdef unsigned long long trans[8]
    cdef unsigned long long tmp_pat
    cdef int i, j

    d12_trans16(pat, trans)

    for i in range(8):
        for j in range(i+1, 8):
            if trans[j] < trans[i]:
                tmp = trans[j]
                trans[j] = trans[i]
                trans[i] = tmp

    for i in range(8):
        print_d12(trans[i], show_bits, show_board)


cpdef void print_d12_move_trans16(unsigned long long pat, bint show_bits=True, bint show_board=True):
    cdef unsigned long long trans[16]
    cdef unsigned long long tmp_pat
    cdef int i, j

    d12_trans16(pat, trans)

    for i in range(16):
        for j in range(i+1, 16):
            if trans[j] < trans[i]:
                tmp = trans[j]
                trans[j] = trans[i]
                trans[i] = tmp

    for i in range(16):
        print_d12(trans[i], show_bits, show_board)


""" 3x3 pattern functions
"""
cdef unsigned long long x33_hash(game_state_t *game, int pos, int color) nogil except? -1:
    cdef int neighbor_pos
    cdef string_t *string
    cdef unsigned long long hash = 0
    cdef int i

    for i in range(8):
        neighbor_pos = neighbor8_seq_pos[pos][i]
        hash ^= color_mt[i][game.board[neighbor_pos]]
        string = &game.string[game.string_id[neighbor_pos]]
        if string.flag:
            hash ^= liberty_mt[i][MIN(string.libs, 3)]
        else:
            hash ^= liberty_mt[i][0]

    return hash ^ player_mt[color]


cpdef unsigned long long x33_hash_from_bits(unsigned long long bits) except? -1:
    cdef int i, j
    cdef unsigned long long hash = 0
    for i in range(8):
        hash ^= color_mt[i][bits >> (18+2*i) & 0x3]
        hash ^= liberty_mt[i][bits >> (2+2*i) & 0x3]
    return hash ^ player_mt[bits & 0x3]


cdef unsigned long long x33_bits(game_state_t *game, int pos, int color) nogil except? -1:
    cdef int neighbor_pos
    cdef string_t *string
    cdef unsigned long long color_pat = 0 
    cdef int lib_pat = 0
    cdef int i

    for i in range(8):
        neighbor_pos = neighbor8_seq_pos[pos][i]
        color_pat |= (game.board[neighbor_pos] << i*2)
        string = &game.string[game.string_id[neighbor_pos]]
        if string.flag:
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
        buf.append("0b{:s}".format(bin(pat3)[2:].rjust(34, '0')))
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


""" Non-response 12 diamond(MD2) Pattern functions
"""
cdef unsigned long long nonres_d12_hash(game_state_t *game, int pos, int color) nogil except? -1:
    cdef int md_pos
    cdef string_t *string
    cdef unsigned long long hash = 0
    cdef int i

    for i in range(12):
        md_pos = md2_pos[pos][i]
        hash ^= color_mt[i][game.board[md_pos]]
        string = &game.string[game.string_id[md_pos]]
        if string.flag:
            hash ^= liberty_mt[i][MIN(string.libs, 3)]
        else:
            hash ^= liberty_mt[i][0]

    return hash ^ player_mt[color]


cdef unsigned long long nonres_d12_bits(game_state_t *game, int pos, int color) nogil except? -1:
    cdef int md_pos
    cdef string_t *string
    cdef unsigned long long color_pat = 0
    cdef int lib_pat = 0
    cdef int i

    for i in range(12):
        md_pos = md2_pos[pos][i]
        color_pat |= (game.board[md_pos] << i*2)
        string = &game.string[game.string_id[md_pos]]
        if string.flag:
            lib_pat |= (MIN(string.libs, 3) << i*2)

    return (((color_pat << 24) | lib_pat) << 2) | color


cpdef unsigned long long nonres_d12_hash_from_bits(unsigned long long bits) except? -1:
    cdef int i, j
    cdef unsigned long long hash = 0

    for i in range(12):
        hash ^= color_mt[i][bits >> (26+2*i) & 0x3]
        hash ^= liberty_mt[i][bits >> (2+2*i) & 0x3]
    return hash ^ player_mt[bits & 0x3]


cpdef unsigned long long nonres_d12_trans8_min(unsigned long long pat):
    cdef unsigned long long trans[8]
    cdef unsigned long long min_pat
    cdef int i

    nonres_d12_trans8(pat, trans)

    min_pat = trans[0]
    for i in range(1, 8):
        if trans[i] < min_pat:
            min_pat = trans[i]

    return min_pat


cpdef unsigned long long nonres_d12_trans16_min(unsigned long long pat):
    cdef unsigned long long trans[16]
    cdef unsigned long long min_pat
    cdef int i

    nonres_d12_trans16(pat, trans)

    min_pat = trans[0]
    for i in range(1, 16):
        if trans[i] < min_pat:
            min_pat = trans[i]

    return min_pat


cdef void nonres_d12_trans8(unsigned long long pat, unsigned long long *trans):
    trans[0] = pat
    trans[1] = nonres_d12_rot90(pat)
    trans[2] = nonres_d12_rot90(trans[1])
    trans[3] = nonres_d12_rot90(trans[2])
    trans[4] = nonres_d12_fliplr(pat)
    trans[5] = nonres_d12_flipud(pat)
    trans[6] = nonres_d12_transp(pat)
    trans[7] = nonres_d12_fliplr(trans[1])


cdef void nonres_d12_trans16(unsigned long long pat, unsigned long long *trans):
    trans[0] = pat
    trans[1] = nonres_d12_rot90(pat)
    trans[2] = nonres_d12_rot90(trans[1])
    trans[3] = nonres_d12_rot90(trans[2])
    trans[4] = nonres_d12_fliplr(pat)
    trans[5] = nonres_d12_flipud(pat)
    trans[6] = nonres_d12_transp(pat)
    trans[7] = nonres_d12_fliplr(trans[1])
    trans[8] = nonres_d12_rev(trans[0])
    trans[9] = nonres_d12_rev(trans[1])
    trans[10] = nonres_d12_rev(trans[2])
    trans[11] = nonres_d12_rev(trans[3])
    trans[12] = nonres_d12_rev(trans[4])
    trans[13] = nonres_d12_rev(trans[5])
    trans[14] = nonres_d12_rev(trans[6])
    trans[15] = nonres_d12_rev(trans[7])


cpdef unsigned long long nonres_d12_rev(unsigned long long pat):
    return ((((pat >> 27) & 0x555555) | (((pat >> 26) & 0x555555) << 1)) << 26) | ((pat >> 2 & 0xffffff) << 2) | (~pat & 0x3)


cpdef unsigned long long nonres_d12_rot90(unsigned long long pat):
    return (((pat & <unsigned long long>0x300c00300c) << 8) |
            ((pat & <unsigned long long>0xc30000c30) << 14) |
            ((pat & <unsigned long long>0xc00000c0) << 6) |
            ((pat & <unsigned long long>0x300000300) >> 4) |
            ((pat & <unsigned long long>0x300c00300c000) >> 8) |
            ((pat & <unsigned long long>0xc30000c30000) >> 14) |
            ((pat & <unsigned long long>0xc00000c0000) << 4) |
            ((pat & <unsigned long long>0x300000300000) >> 6) |
            (pat & 0x3))


cpdef unsigned long long nonres_d12_fliplr(unsigned long long pat):
    return (((pat & <unsigned long long>0xc00300c0030) << 4) |
            ((pat & <unsigned long long>0xc00300c00300) >> 4) |
            ((pat & <unsigned long long>0xc00000c00) << 6) |
            ((pat & <unsigned long long>0x3000003000) << 2) |
            ((pat & <unsigned long long>0xc00000c000) >> 2) |
            ((pat & <unsigned long long>0x30000030000) >> 6) |
            (pat & 0x33000cf3000cf))


cpdef unsigned long long nonres_d12_flipud(unsigned long long pat):
    return (((pat & <unsigned long long>0xc00000c) << 22) |
            ((pat & <unsigned long long>0x3f00003f0) << 14) |
            ((pat & <unsigned long long>0xfc0000fc0000) >> 14) |
            ((pat & <unsigned long long>0x3000003000000) >> 22) |
            (pat & 0x3fc0003fc03))


cpdef unsigned long long nonres_d12_transp(unsigned long long pat):
    return (((pat & <unsigned long long>0x3000c03000c) << 8) |
            ((pat & <unsigned long long>0xc0c000c0c0) << 6) |
            ((pat & <unsigned long long>0x300000300) << 10) |
            ((pat & <unsigned long long>0x3000c03000c00) >> 8) |
            ((pat & <unsigned long long>0x303000303000) >> 6) |
            ((pat & <unsigned long long>0xc00000c0000) >> 10) |
            (pat & 0xc00030c00033))


cpdef void print_nonres_d12(unsigned long long pat, bint show_bits=True, bint show_board=True):
    buf = []
    stone = ['+', 'B', 'W', '#']
    color = ['?', 'x', 'o']
    liberty = [0, 1, 2, 3]
    if show_bits:
        buf.append("0b{:s}".format(bin(pat)[2:].rjust(50, '0')))
    if show_board:
        if show_bits:
            buf.append("\n")
        buf.append("  {:s}       {:d} \n".format(
            stone[(pat >> 26) & 0x3],
            liberty[(pat >> 2) & 0x3]
            ))
        buf.append(" {:s}{:s}{:s}     {:d}{:d}{:d}\n".format(
            stone[(pat >> 28) & 0x3],
            stone[(pat >> 30) & 0x3],
            stone[(pat >> 32) & 0x3],
            liberty[(pat >> 4) & 0x3],
            liberty[(pat >> 6) & 0x3],
            liberty[(pat >> 8) & 0x3]
            ))
        buf.append("{:s}{:s}{:s}{:s}{:s}   {:d}{:d} {:d}{:d}\n".format(
            stone[(pat >> 34) & 0x3],
            stone[(pat >> 36) & 0x3],
            color[pat & 0x3],
            stone[(pat >> 38) & 0x3],
            stone[(pat >> 40) & 0x3],
            liberty[(pat >> 10) & 0x3],
            liberty[(pat >> 12) & 0x3],
            liberty[(pat >> 14) & 0x3],
            liberty[(pat >> 16) & 0x3]
            ))
        buf.append(" {:s}{:s}{:s}     {:d}{:d}{:d}\n".format(
            stone[(pat >> 42) & 0x3],
            stone[(pat >> 44) & 0x3],
            stone[(pat >> 46) & 0x3],
            liberty[(pat >> 18) & 0x3],
            liberty[(pat >> 20) & 0x3],
            liberty[(pat >> 22) & 0x3]
            ))
        buf.append("  {:s}       {:d} \n".format(
            stone[(pat >> 48) & 0x3],
            liberty[(pat >> 24) & 0x3]
            ))
    print(''.join(buf))


cpdef void print_nonres_d12_trans8(unsigned long long pat, bint show_bits=True, bint show_board=True):
    cdef unsigned long long trans[8]
    cdef unsigned long long tmp_pat
    cdef int i, j

    nonres_d12_trans8(pat, trans)

    for i in range(8):
        for j in range(i+1, 8):
            if trans[j] < trans[i]:
                tmp = trans[j]
                trans[j] = trans[i]
                trans[i] = tmp

    for i in range(8):
        print_nonres_d12(trans[i], show_bits, show_board)


cpdef void print_nonres_d12_trans16(unsigned long long pat, bint show_bits=True, bint show_board=True):
    cdef unsigned long long trans[16]
    cdef unsigned long long tmp_pat
    cdef int i, j

    nonres_d12_trans16(pat, trans)

    for i in range(16):
        for j in range(i+1, 16):
            if trans[j] < trans[i]:
                tmp = trans[j]
                trans[j] = trans[i]
                trans[i] = tmp

    for i in range(16):
        print_nonres_d12(trans[i], show_bits, show_board)

