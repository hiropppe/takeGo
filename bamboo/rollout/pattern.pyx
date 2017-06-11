from libc.stdio cimport printf

from libcpp.string cimport string as cppstring

from bamboo.go.board cimport S_EMPTY, S_BLACK, S_WHITE, S_OB
from bamboo.go.board cimport game_state_t, string_t, get_neighbor8
from bamboo.go.zobrist_hash cimport mt
from bamboo.go.printer cimport print_board


class IllegalState(Exception):
    pass

cdef void initialize_hash():
    cdef int i
    for i in range(8):
        hash_bit[i] = mt()

cdef unsigned int calculate_x33_bit(game_state_t *game, int pos, int color) except? -1:
    cdef int neighbor8[8]
    cdef int neighbor_pos
    cdef int string_id
    cdef string_t *string
    cdef int string_color
    cdef int string_libs
    cdef int i
    cdef int vertex
    cdef int pat = 0 

    get_neighbor8(neighbor8, pos)

    for i in range(8):
        neighbor_pos = neighbor8[i]
        string_id = game.string_id[neighbor_pos]
        # string exists
        if string_id:
            string = &game.string[string_id]
            if string.color == S_BLACK:
                if string.libs == 1:
                    vertex = BLACK1
                elif string.libs == 2:
                    vertex = BLACK2
                elif string.libs >= 3:
                    vertex = BLACK3
                else:
                    raise IllegalState()
            else:
                if string.libs == 1:
                    vertex = WHITE1
                elif string.libs == 2:
                    vertex = WHITE2
                elif string.libs >= 3:
                    vertex = WHITE3
                else:
                    raise IllegalState()
        elif game.board[neighbor_pos] == S_EMPTY:
            vertex = EMPTY
        elif game.board[neighbor_pos] == S_OB:
            vertex = OB
        else:
            raise IllegalState()
        #printf('%d', vertex)
        pat |= (vertex << (i*3))
    #printf(' ')

    pat = (pat << 2) | color

    return pat


cdef void print_x33(unsigned int pat3):
    cdef char stone[8]
    cdef char color[3]
    cdef list buf = []
    stone = ['+', 'B', 'B', 'B', 'W', 'W', 'W', '#']
    color = ['*', 'b', 'w']
    buf.append("\n")
    buf.append("{:s}{:s}{:s}\n".format(cppstring(1, stone[(pat3 >> 2) & 0x7]), cppstring(1, stone[(pat3 >> (2+3)) & 0x7]), cppstring(1, stone[(pat3 >> (2+6)) & 0x7])))
    buf.append("{:s}{:s}{:s}\n".format(cppstring(1, stone[(pat3 >> (2+9)) & 0x7]), cppstring(1, color[pat3 & 0x3]), cppstring(1, stone[(pat3 >> (2+12)) & 0x7])))
    buf.append("{:s}{:s}{:s}\n".format(cppstring(1, stone[(pat3 >> (2+15)) & 0x7]), cppstring(1, stone[(pat3 >> (2+18)) & 0x7]), cppstring(1, stone[(pat3 >> (2+21)) & 0x7])))
    #buf.append("color={:s}\n".format(cppstring(1, color[pat3 & 0x3])))
    print ''.join(buf)

cdef int init_nakade(object nakade_file):
    cdef int nakade_size = 8192
    return nakade_size

cdef int init_x33(object x33_file):
    cdef int x33_size = 69338
    return x33_size

cdef int init_d12(object d12_file):
    cdef int d12_size = 32207
    return d12_size
