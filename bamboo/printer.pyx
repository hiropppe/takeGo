# -*- coding: utf-8 -*-
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

from libcpp.string cimport string as cppstring
from libcpp.string cimport to_string

from libc.string cimport memset
from libc.stdio cimport printf, snprintf, stdout, fflush
from libc.math cimport (
    sqrt as csqrt,
    ceilf,
    roundf,
    powf
)

from bamboo.board cimport S_BLACK, S_WHITE, S_MAX
from bamboo.board cimport POS
from bamboo.board cimport board_start, board_end, board_size, pure_board_size

from bamboo.tree_search cimport EXPLORATION_CONSTANT


cdef void print_board(game_state_t *game) nogil:
    cdef char *stone = [b'+', b'B', b'W', b'#']
    cdef int i, x, y, pos
    cdef char buf[10]
    cdef cppstring s
    cdef char fill_char = b' '

    printf("Prisoner: [Black] >> %d [White] >> %d\n", game.prisoner[<int>S_BLACK], game.prisoner[<int>S_WHITE])
    printf("Move    : %d\n", game.moves)

    printf("    ")
    i = 1
    for _ in range(board_start, board_end + 1):
        printf(" %s", cppstring(1, <char>gogui_x[i]).c_str())
        i += 1
    printf("\n")

    printf("   +")
    for i in range(pure_board_size * 2 + 1):
        printf("-")
    printf("+\n")

    i = pure_board_size
    for y in range(board_start, board_end + 1):
        s = to_string(i)
        s.insert(0, 2 - s.size(), fill_char)
        printf("%s:|", s.c_str())
        for x in range(board_start, board_end + 1):
            pos = POS(x, y, board_size)
            printf(" %s", cppstring(1, stone[<int>game.board[pos]]).c_str())
        printf(" |\n")
        i -= 1

    printf("   +")
    for i in range(1, pure_board_size * 2 + 1 + 1):
        printf("-")
    printf("+\n")


cdef void print_rollout_count(tree_node_t *root) nogil:
    cdef char *stone = [b'#', b'B', b'W']
    cdef int i, x, y, pos
    cdef char buf[10]
    cdef cppstring s
    cdef char fill_char = b' '
    cdef int rollout_count[361]
    cdef tree_node_t *child

    memset(rollout_count, 0, sizeof(int) * 361);

    for i in range(root.num_child):
        pos = root.children_pos[i]
        rollout_count[pos] = <int>root.children[pos].Nr

    printf(">> Number of rollout: %d\n", <int>root.Nr)
    printf("Player: %s\n", cppstring(1, stone[root.player_color]).c_str())

    printf("    ")
    i = 1
    for _ in range(board_start, board_end + 1):
        printf("      %s", cppstring(1, <char>gogui_x[i]).c_str())
        i += 1
    printf("\n")

    printf("   +")
    for i in range(pure_board_size * 7 + 1):
        printf("-")
    printf("+\n")

    for y in range(pure_board_size):
        s = to_string(pure_board_size-y)
        s.insert(0, 2 - s.size(), fill_char)
        printf("%s:|", s.c_str())
        for x in range(pure_board_size):
            pos = POS(x, y, pure_board_size)
            s = to_string(rollout_count[pos])
            s.insert(0, 7 - s.size(), fill_char)
            printf("%s", s.c_str())
        printf(" |\n")

    printf("   +")
    for i in range(1, pure_board_size * 7 + 1 + 1):
        printf("-")
    printf("+\n")


cdef void print_winning_ratio(tree_node_t *root) nogil:
    cdef char *stone = [b'#', b'B', b'W']
    cdef int i, x, y, pos
    cdef char buf[10]
    cdef cppstring s
    cdef char fill_char = b' '
    cdef double winning_ratio[361]
    cdef tree_node_t *child
    cdef float p

    memset(winning_ratio, 0, sizeof(double) * 361);

    if root.Nr > 0:
        for i in range(root.num_child):
            pos = root.children_pos[i]
            child = root.children[pos]
            if child.Nr > .0:
                winning_ratio[pos] = child.Wr/child.Nr
            else:
                winning_ratio[pos] = .0

    printf(">> Winning Ratio: %3.2lf\n", 100.0-(root.Wr*100.0/root.Nr))
    printf("Player: %s\n", cppstring(1, stone[root.player_color]).c_str())

    printf("    ")
    i = 1
    for _ in range(board_start, board_end + 1):
        printf("       %s", cppstring(1, <char>gogui_x[i]).c_str())
        i += 1
    printf("\n")

    printf("   +")
    for i in range(pure_board_size * 8 + 1):
        printf("-")
    printf("+\n")

    for y in range(pure_board_size):
        s = to_string(pure_board_size-y)
        s.insert(0, 2 - s.size(), fill_char)
        printf("%s:|", s.c_str())
        for x in range(pure_board_size):
            pos = POS(x, y, pure_board_size)
            p = ceilf(winning_ratio[pos]*powf(10, 4))/powf(10, 2)
            s = to_string(p).substr(0, 5)
            s.insert(0, 8 - s.size(), fill_char)
            printf("%s", s.c_str())
        printf(" |\n")

    printf("   +")
    for i in range(1, pure_board_size * 8 + 1 + 1):
        printf("-")
    printf("+\n")


cdef void print_PN(tree_node_t *root) nogil:
    cdef char *stone = [b'#', b'B', b'W']
    cdef int i, x, y, pos
    cdef char buf[10]
    cdef cppstring s
    cdef char fill_char = b' '
    cdef double prior_prob[361]
    cdef tree_node_t *child
    cdef float p

    memset(prior_prob, 0, sizeof(double) * 361);

    for i in range(root.num_child):
        pos = root.children_pos[i]
        prior_prob[pos] = root.children[pos].P*100.0

    printf(">> PN evaluation\n")
    printf("Player: %s\n", cppstring(1, stone[root.player_color]).c_str())

    printf("    ")
    i = 1
    for _ in range(board_start, board_end + 1):
        printf("       %s", cppstring(1, <char>gogui_x[i]).c_str())
        i += 1
    printf("\n")

    printf("   +")
    for i in range(pure_board_size * 8 + 1):
        printf("-")
    printf("+\n")

    for y in range(pure_board_size):
        s = to_string(pure_board_size-y)
        s.insert(0, 2 - s.size(), fill_char)
        printf("%s:|", s.c_str())
        for x in range(pure_board_size):
            pos = POS(x, y, pure_board_size)
            p = ceilf(prior_prob[pos]*powf(10, 2))/powf(10, 2)
            s = to_string(p).substr(0, 5)
            s.insert(0, 8 - s.size(), fill_char)
            printf("%s", s.c_str())
        printf(" |\n")

    printf("   +")
    for i in range(1, pure_board_size * 8 + 1 + 1):
        printf("-")
    printf("+\n")


cdef void print_VN(tree_node_t *root) nogil:
    cdef char *stone = [b'#', b'B', b'W']
    cdef int i, x, y, pos
    cdef char buf[10]
    cdef cppstring s
    cdef char fill_char = b' '    
    cdef double values[361]
    cdef tree_node_t *child
    cdef float p

    memset(values, 0, sizeof(double) * 361);

    for i in range(root.num_child):
        pos = root.children_pos[i]
        values[pos] = root.children[pos].Wv*100.0

    printf(">> VN evaluation: %3.2lf\n", root.Wv*100.0)
    printf("Player: %s\n", cppstring(1, stone[root.player_color]).c_str())

    printf("    ")
    i = 1
    for _ in range(board_start, board_end + 1):
        printf("       %s", cppstring(1, <char>gogui_x[i]).c_str())
        i += 1
    printf("\n")

    printf("   +")
    for i in range(pure_board_size * 8 + 1):
        printf("-")
    printf("+\n")

    for y in range(pure_board_size):
        s = to_string(pure_board_size-y)
        s.insert(0, 2 - s.size(), fill_char)
        printf("%s:|", s.c_str())
        for x in range(pure_board_size):
            pos = POS(x, y, pure_board_size)
            p = ceilf(values[pos]*powf(10, 2))/powf(10, 2)
            s = to_string(p).substr(0, 5)
            s.insert(0, 7 - s.size(), fill_char)
            printf("%s", s.c_str())
        printf(" |\n")

    printf("   +")
    for i in range(1, pure_board_size * 8 + 1 + 1):
        printf("-")
    printf("+\n")


cdef void print_Q(tree_node_t *root) nogil:
    cdef char *stone = [b'#', b'B', b'W']
    cdef int i, x, y, pos
    cdef char buf[10]
    cdef cppstring s
    cdef char fill_char = b' '
    cdef double action_value[361]
    cdef tree_node_t *child
    cdef float p

    memset(action_value, 0, sizeof(double) * 361);

    for i in range(root.num_child):
        pos = root.children_pos[i]
        action_value[pos] = root.children[pos].Q

    printf(">> Action-Value(Q)\n")
    printf("Player: %s\n", cppstring(1, stone[root.player_color]).c_str())

    printf("    ")
    i = 1
    for _ in range(board_start, board_end + 1):
        printf("       %s", cppstring(1, <char>gogui_x[i]).c_str())
        i += 1
    printf("\n")

    printf("   +")
    for i in range(pure_board_size * 8 + 1):
        printf("-")
    printf("+\n")

    for y in range(pure_board_size):
        s = to_string(pure_board_size-y)
        s.insert(0, 2 - s.size(), fill_char)
        printf("%s:|", s.c_str())
        for x in range(pure_board_size):
            pos = POS(x, y, pure_board_size)
            p = ceilf(action_value[pos]*powf(10, 4))/powf(10, 4)
            s = to_string(p).substr(0, 5)
            s.insert(0, 8 - s.size(), fill_char)
            printf("%s", s.c_str())
        printf(" |\n")

    printf("   +")
    for i in range(1, pure_board_size * 8 + 1 + 1):
        printf("-")
    printf("+\n")


cdef void print_u(tree_node_t *root) nogil:
    cdef char *stone = [b'#', b'B', b'W']
    cdef int i, x, y, pos
    cdef char buf[10]
    cdef cppstring s
    cdef char fill_char = b' '
    cdef double bonus[361]
    cdef tree_node_t *child
    cdef float p

    memset(bonus, 0, sizeof(double) * 361);

    for i in range(root.num_child):
        pos = root.children_pos[i]
        child = root.children[pos]
        if root.Nr > .0:
            bonus[pos] = EXPLORATION_CONSTANT * child.P * (csqrt(root.Nr) / (1 + child.Nr))
        else:
            bonus[pos] = EXPLORATION_CONSTANT * child.P

    printf(">> Bonus(u)\n")
    printf("Player: %s\n", cppstring(1, stone[root.player_color]).c_str())

    printf("    ")
    i = 1
    for _ in range(board_start, board_end + 1):
        printf("       %s", cppstring(1, <char>gogui_x[i]).c_str())
        i += 1
    printf("\n")

    printf("   +")
    for i in range(pure_board_size * 8 + 1):
        printf("-")
    printf("+\n")

    for y in range(pure_board_size):
        s = to_string(pure_board_size-y)
        s.insert(0, 2 - s.size(), fill_char)
        printf("%s:|", s.c_str())
        for x in range(pure_board_size):
            pos = POS(x, y, pure_board_size)
            p = ceilf(bonus[pos]*powf(10, 4))/powf(10, 4)
            s = to_string(p).substr(0, 5)
            s.insert(0, 8 - s.size(), fill_char)
            printf("%s", s.c_str())
        printf(" |\n")

    printf("   +")
    for i in range(1, pure_board_size * 8 + 1 + 1):
        printf("-")
    printf("+\n")


cdef void print_selection_value(tree_node_t *root) nogil:
    cdef char *stone = [b'#', b'B', b'W']
    cdef int i, x, y, pos
    cdef char buf[10]
    cdef cppstring s
    cdef char fill_char = b' '
    cdef double selection[361]
    cdef tree_node_t *child
    cdef float p

    memset(selection, 0, sizeof(double) * 361);

    for i in range(root.num_child):
        pos = root.children_pos[i]
        child = root.children[pos]
        if root.Nr > .0:
            selection[pos] = child.Q + EXPLORATION_CONSTANT * child.P * (csqrt(root.Nr) / (1 + child.Nr))
        else:
            selection[pos] = child.Q + EXPLORATION_CONSTANT * child.P

    printf(">> Action-Value(Q) + Bonus(u)\n")
    printf("Player: %s\n", cppstring(1, stone[root.player_color]).c_str())

    printf("    ")
    i = 1
    for _ in range(board_start, board_end + 1):
        printf("       %s", cppstring(1, <char>gogui_x[i]).c_str())
        i += 1
    printf("\n")

    printf("   +")
    for i in range(pure_board_size * 8 + 1):
        printf("-")
    printf("+\n")

    for y in range(pure_board_size):
        s = to_string(pure_board_size-y)
        s.insert(0, 2 - s.size(), fill_char)
        printf("%s:|", s.c_str())
        for x in range(pure_board_size):
            pos = POS(x, y, pure_board_size)
            p = ceilf(selection[pos]*powf(10, 4))/powf(10, 4)
            s = to_string(p).substr(0, 5)
            s.insert(0, 8 - s.size(), fill_char)
            printf("%s", s.c_str())
        printf(" |\n")

    printf("   +")
    for i in range(1, pure_board_size * 8 + 1 + 1):
        printf("-")
    printf("+\n")


cdef cppstring *rjust(char *buf, unsigned int width, char *fill_char) nogil:
    cdef unsigned int i
    cdef cppstring val
    val = cppstring(buf)
    s = &val
    if width > s.size():
        for i in range(width - s.size()):
            s.insert(0, fill_char)
    return s
