from libc.string cimport memset

from libcpp.string cimport string as cppstring

cimport board 


cdef unsigned int update_mask[12][3]

update_mask[:] = [
  # 3x3
  [ 0, 0x00004000, 0x00008000 ], #  1->8
  [ 0, 0x00001000, 0x00002000 ], #  2->7
  [ 0, 0x00000400, 0x00000800 ], #  3->6
  [ 0, 0x00000100, 0x00000200 ], #  4->5
  [ 0, 0x00000040, 0x00000080 ], #  5->4
  [ 0, 0x00000010, 0x00000020 ], #  6->3
  [ 0, 0x00000004, 0x00000008 ], #  7->2
  [ 0, 0x00000001, 0x00000002 ], #  8->1
  # md2
  [ 0, 0x00100000, 0x00200000 ], #  9->11
  [ 0, 0x00400000, 0x00800000 ], # 10->12
  [ 0, 0x00010000, 0x00020000 ], # 11-> 9
  [ 0, 0x00040000, 0x00080000 ], # 12->10
]


cdef void init_const():
    global N, S, W, E, NN, NE, SS, SW, SE, WW, EE

    N = -board.board_size
    S = board.board_size
    W = 1
    E = -1
    NN = N + N
    NW = N + E
    NE = N + W
    SS = S + S
    SW = S + E
    SE = S + W
    WW = W + W
    EE = E + E


cdef void clear_pattern(unsigned int *pat):
    cdef int y

    memset(pat, 0, sizeof(int) * board.BOARD_MAX)

    for y in range(board.board_start, board.board_end + 1):
      # 1線
      # 上
      pat[board.POS(y, board.board_start, board.board_size)] |= 0x0003003F;  # 1 2 3 9
      # 右
      pat[board.POS(board.board_end, y, board.board_size)] |= 0x000CC330;    # 3 5 8 10
      # 下
      pat[board.POS(y, board.board_end, board.board_size)] |= 0x0030FC00;    # 6 7 8 11
      # 左
      pat[board.POS(board.board_start, y, board.board_size)] |= 0x00C00CC3;  # 1 4 6 12

      # 2線
      # 上
      pat[board.POS(y, board.board_start + 1, board.board_size)] |= 0x00030000;      # 9
      # 右
      pat[board.POS(board.board_end - 1, y, board.board_size)] |= 0x000C0000;        # 10
      # 下
      pat[board.POS(y, board.board_end - 1, board.board_size)] |= 0x00300000;        # 11
      # 左
      pat[board.POS(board.board_start + 1, y, board.board_size)] |= 0x00C00000;      # 12


cdef void update_md2_stone(unsigned int *pat, int pos, char color) nogil:
    pat[pos + NW] |= update_mask[0][<int>color];
    pat[pos +  N] |= update_mask[1][<int>color];
    pat[pos + NE] |= update_mask[2][<int>color];
    pat[pos +  W] |= update_mask[3][<int>color];
    pat[pos +  E] |= update_mask[4][<int>color];
    pat[pos + SW] |= update_mask[5][<int>color];
    pat[pos +  S] |= update_mask[6][<int>color];
    pat[pos + SE] |= update_mask[7][<int>color];
    pat[pos + NN] |= update_mask[8][<int>color];
    pat[pos + EE] |= update_mask[9][<int>color];
    pat[pos + SS] |= update_mask[10][<int>color];
    pat[pos + WW] |= update_mask[11][<int>color];


cdef void update_md2_empty(unsigned int *pat, int pos) nogil:
    pat[pos + NW] &= 0xFF3FFF;
    pat[pos +  N] &= 0xFFCFFF;
    pat[pos + NE] &= 0xFFF3FF;
    pat[pos +  W] &= 0xFFFCFF;
    pat[pos +  E] &= 0xFFFF3F;
    pat[pos + SW] &= 0xFFFFCF;
    pat[pos +  S] &= 0xFFFFF3;
    pat[pos + SE] &= 0xFFFFFC;
    pat[pos + NN] &= 0xCFFFFF;
    pat[pos + EE] &= 0x3FFFFF;
    pat[pos + SS] &= 0xFCFFFF;
    pat[pos + WW] &= 0xF3FFFF;


cdef unsigned int pat3(unsigned int *pat, int pos) nogil:
    return pat[pos] & 0xFFFF


cdef void pat3_transpose8(unsigned int pat3, unsigned int *transp):
    transp[0] = pat3;
    transp[1] = pat3_vertical_mirror(pat3);
    transp[2] = pat3_horizontal_mirror(pat3);
    transp[3] = pat3_vertical_mirror(transp[2]);
    transp[4] = pat3_rotate90(pat3);
    transp[5] = pat3_rotate90(transp[1]);
    transp[6] = pat3_rotate90(transp[2]);
    transp[7] = pat3_rotate90(transp[3]);


cdef void pat3_transpose16(unsigned int pat3, unsigned int *transp):
    transp[0] = pat3
    transp[1] = pat3_vertical_mirror(pat3)
    transp[2] = pat3_horizontal_mirror(pat3)
    transp[3] = pat3_vertical_mirror(transp[2])
    transp[4] = pat3_rotate90(pat3);
    transp[5] = pat3_rotate90(transp[1])
    transp[6] = pat3_rotate90(transp[2])
    transp[7] = pat3_rotate90(transp[3])
    transp[8] = pat3_reverse(transp[0])
    transp[9] = pat3_reverse(transp[1])
    transp[10] = pat3_reverse(transp[2])
    transp[11] = pat3_reverse(transp[3])
    transp[12] = pat3_reverse(transp[4])
    transp[13] = pat3_reverse(transp[5])
    transp[14] = pat3_reverse(transp[6])
    transp[15] = pat3_reverse(transp[7])


cdef unsigned int pat3_reverse(unsigned int pat3):
    return ((pat3 >> 1) & 0x5555) | ((pat3 & 0x5555) << 1)


cdef unsigned int pat3_vertical_mirror(unsigned int pat3):
    return ((pat3 & 0xFC00) >> 10) | (pat3 & 0x03C0) | ((pat3 & 0x003F) << 10)


cdef unsigned int pat3_horizontal_mirror(unsigned int pat3):
    return (REV3((pat3 & 0xFC00) >> 10) << 10) | (REV((pat3 & 0x03C0) >> 6) << 6) | (REV3((pat3 & 0x003F)))


cdef unsigned int pat3_rotate90(unsigned int pat3):
    """
    1 2 3    3 5 8
    4   5 -> 2   7
    6 7 8    1 4 6
    """
    return ((pat3 & 0x0003) << 10) | ((pat3 & 0x0C0C) << 4) | ((pat3 & 0x3030) >> 4) | ((pat3 & 0x00C0) << 6) | ((pat3 & 0x0300) >> 6) | ((pat3 & 0xC000) >> 10)


cdef void print_input_pat3(unsigned int pat3):
    cdef char stone[4]
    cdef list buf = []
    stone = ['+', '@', 'O', '#']
    buf.append("\n")
    buf.append("{:s}{:s}{:s}\n".format(cppstring(1, stone[pat3 & 0x3]), cppstring(1, stone[(pat3 >> 2) & 0x3]), cppstring(1, stone[(pat3 >> 4) & 0x3])))
    buf.append("{:s}*{:s}\n".format(cppstring(1, stone[(pat3 >> 6) & 0x3]), cppstring(1, stone[(pat3 >> 8) & 0x3])))
    buf.append("{:s}{:s}{:s}\n".format(cppstring(1, stone[(pat3 >> 10) & 0x3]), cppstring(1, stone[(pat3 >> 12) & 0x3]), cppstring(1, stone[(pat3 >> 14) & 0x3])))
    print ''.join(buf)
