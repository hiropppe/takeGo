from libc.stdlib cimport malloc, free

from . cimport board 
from . cimport printer
from . cimport parseboard
from . cimport pattern as pat

from .board cimport STRING_EMPTY_END


def test_set_board_size_9():
    board.set_board_size(9)

    assert (board.pure_board_size == 9)
    assert (board.pure_board_max == 81)
    assert (board.board_size == 13)
    assert (board.board_max == 169)
    assert (board.max_string == 64)
    assert (board.max_neighbor == 64)
    assert (board.board_start == 2)
    assert (board.board_end == 10)
    assert (board.string_lib_max == 143)
    assert (board.string_pos_max == 143)
    assert (board.string_end == 142)
    assert (board.liberty_end == 142)
    assert (board.max_records == 243)
    assert (board.max_moves == 242)


def test_set_board_size_19():
    board.set_board_size(19)

    assert (board.pure_board_size == 19)
    assert (board.pure_board_max == 361)
    assert (board.board_size == 23)
    assert (board.board_max == 529)
    assert (board.max_string == 288)
    assert (board.max_neighbor == 288)
    assert (board.board_start == 2)
    assert (board.board_end == 20)
    assert (board.string_lib_max == 483)
    assert (board.string_pos_max == 483)
    assert (board.string_end == 482)
    assert (board.liberty_end == 482)
    assert (board.max_records == 1083)
    assert (board.max_moves == 1082)


def test_add_liberty_to_isolated_one():
    cdef board.game_state_t *game
    cdef board.string_t *string
    cdef int neighbor4[4]

    game = __initialize_game()

    origin = board.POS(4, 4, board.board_size)
    string = __initialize_string(game, origin, board.S_BLACK) 

    neighbor4 = board.neighbor4_pos[origin]

    board.add_liberty(string, neighbor4[0], 0)
    assert (string.lib[0] == neighbor4[0])
    assert (string.libs == 1)

    board.add_liberty(string, neighbor4[1], neighbor4[0])
    assert (string.lib[neighbor4[0]] == neighbor4[1])
    assert (string.libs == 2)

    board.add_liberty(string, neighbor4[2], neighbor4[1])
    assert (string.lib[neighbor4[1]] == neighbor4[2])
    assert (string.libs == 3)

    board.add_liberty(string, neighbor4[3], neighbor4[2])
    assert (string.lib[neighbor4[2]] == neighbor4[3])
    assert (string.libs == 4)

    board.free_game(game)


def test_remove_liberty_of_isolated_one():
    cdef board.game_state_t *game
    cdef board.string_t *string
    cdef int neighbor4[4]

    game = __initialize_game()

    pos = board.POS(4, 4, board.board_size)
    board.make_string(game, pos, board.S_BLACK)
    string = &game.string[1]

    neighbor4 = board.neighbor4_pos[pos]
    north = neighbor4[0]
    west  = neighbor4[1]
    east  = neighbor4[2]
    south = neighbor4[3]

    board.remove_liberty(string, north)
    assert (string.lib[north] == 0)
    assert (string.lib[0] == west)

    board.remove_liberty(string, west)
    assert (string.lib[west] == 0)
    assert (string.lib[0] == east)

    board.remove_liberty(string, east)
    assert (string.lib[east] == 0)
    assert (string.lib[0] == south)

    board.remove_liberty(string, south)
    assert (string.lib[south] == 0)

    board.free_game(game)


def test_make_string_of_isolated_one():
    cdef board.game_state_t *game
    cdef board.string_t *new_string

    game = __initialize_game()

    pos = board.POS(4, 4, board.board_size)
    board.make_string(game, pos, board.S_BLACK)
    new_string = &game.string[1]

    assert (new_string.lib[0] == pos - board.board_size)
    assert (new_string.lib[pos - board.board_size] == pos - 1)
    assert (new_string.lib[pos - 1] == pos + 1)
    assert (new_string.lib[pos + 1] == pos + board.board_size)
    assert (new_string.lib[pos + board.board_size] == board.string_lib_max - 1)
    assert (new_string.libs == 4)

    assert (game.string_id[pos] == 1)
    assert (game.string_next[pos] == board.string_pos_max - 1)

    board.free_game(game)


def test_add_neighbor_to_isolated_one():
    cdef board.game_state_t *game
    cdef board.string_t *string
    cdef int neighbor[4] 

    game = __initialize_game()

    pos = board.POS(4, 4, board.board_size)
    board.make_string(game, pos, board.S_BLACK)

    neighbor4 = board.neighbor4_pos[pos]
    south = neighbor[0]

    string = __initialize_string(game, south, board.S_WHITE)

    string_id = game.string_id[pos]
    neighbor_string_id = game.string_id[south]

    assert (string_id == 1)
    assert (neighbor_string_id == 2)
    assert (game.string[1].neighbor[0] == board.NEIGHBOR_END)
    assert (game.string[2].neighbor[0] == board.NEIGHBOR_END)

    board.add_neighbor(&game.string[string_id], neighbor_string_id, 0)
    board.add_neighbor(&game.string[neighbor_string_id], string_id, 0)

    assert (game.string[1].neighbor[0] == 2)
    assert (game.string[1].neighbor[2] == board.NEIGHBOR_END)
    assert (game.string[2].neighbor[0] == 1)
    assert (game.string[2].neighbor[1] == board.NEIGHBOR_END)

    board.free_game(game)


def test_add_stone_to_string_less_position():
    cdef board.game_state_t *game
    cdef board.string_t *string
    cdef int origin = board.POS(4, 4, board.board_size), add_stone = origin - 1, head = 0

    game = __initialize_game()

    board.make_string(game, origin, board.S_BLACK)

    assert (game.string_id[origin] == 1)

    string = &game.string[1]
    board.add_stone_to_string(game, string, add_stone, head)

    assert (string.size == 2)
    assert (string.origin == add_stone)
    assert (game.string_next[add_stone] == origin)
    assert (game.string_next[origin] == board.string_pos_max - 1)

    board.free_game(game)


def test_add_stone_to_string_larger_position_from_zero_head():
    cdef board.game_state_t *game
    cdef board.string_t *string
    cdef int origin = board.POS(4, 4, board.board_size), add_stone = origin + 1, head = 0

    game = __initialize_game()

    board.make_string(game, origin, board.S_BLACK)

    assert (game.string_id[origin] == 1)

    string = &game.string[1]
    board.add_stone_to_string(game, string, add_stone, head)

    assert (string.size == 2)
    assert (game.string_next[origin] == add_stone)
    assert (game.string_next[add_stone] == board.string_pos_max - 1)

    board.free_game(game)


def test_add_stone_to_string_larger_position_from_nonzero_head():
    pass


def test_add_stone_to_isolated_stone():
    cdef board.game_state_t *game
    cdef board.string_t *string
    cdef int origin = board.POS(4, 4, board.board_size)
    cdef int add_pos = origin + 1

    game = __initialize_game()

    board.make_string(game, origin, board.S_BLACK)

    assert (game.string_id[origin] == 1)
    
    string = &game.string[1]
    assert (string.libs == 4)
    assert (string.size == 1)

    board.add_stone(game, add_pos, board.S_BLACK, 1)

    assert (string.libs == 8)
    assert (string.size == 2)
    assert (game.string_id[add_pos] == 1)
    assert (game.string_next[origin] == add_pos)
    assert (game.string_next[add_pos] == board.string_pos_max - 1)

    board.free_game(game)


def test_merge_string_two():
    cdef board.game_state_t *game
    cdef int add_pos = board.POS(4, 4, board.board_size)
    cdef int first = add_pos - 1
    cdef int second = add_pos + 1
    cdef board.string_t *string[3]

    game = __initialize_game()

    board.make_string(game, first, board.S_BLACK)
    board.make_string(game, second, board.S_BLACK)

    assert (game.string_id[first], 1)
    assert (game.string_id[second], 2)

    assert (game.string_next[first], board.string_pos_max - 1)
    assert (game.string_next[second], board.string_pos_max - 1)

    assert (game.string[1].neighbor[0] == board.NEIGHBOR_END)
    assert (game.string[2].neighbor[0] == board.NEIGHBOR_END)

    assert (game.string[1].lib[first + board.board_size] == board.string_lib_max - 1)
    assert (game.string[2].lib[second + board.board_size] == board.string_lib_max - 1)

    string[0] = &game.string[2]

    board.add_stone(game, add_pos, board.S_BLACK, 1)
    
    assert (game.string_next[first], add_pos)
    assert (game.string_next[add_pos], board.string_pos_max - 1)
    assert (game.string_next[second], board.string_pos_max - 1)

    board.merge_string(game, &game.string[1], string, 1)

    board.free_game(game)


def test_merge_string_three():
    cdef board.game_state_t *game
    cdef int add_pos = board.POS(4, 4, board.board_size)
    cdef int first = add_pos - board.board_size
    cdef int second = add_pos - 1
    cdef int third = add_pos + 1
    cdef board.string_t *string[3]

    game = __initialize_game()

    board.make_string(game, first, board.S_BLACK)
    board.make_string(game, second, board.S_BLACK)
    board.make_string(game, third, board.S_BLACK)

    assert (game.string_id[first], 1)
    assert (game.string_id[second], 2)
    assert (game.string_id[third], 3)

    assert (game.string_next[first], board.string_pos_max - 1)
    assert (game.string_next[second], board.string_pos_max - 1)
    assert (game.string_next[third], board.string_pos_max - 1)

    assert (game.string[1].neighbor[0] == board.NEIGHBOR_END)
    assert (game.string[2].neighbor[0] == board.NEIGHBOR_END)
    assert (game.string[3].neighbor[0] == board.NEIGHBOR_END)

    assert (game.string[1].lib[first + board.board_size] == board.string_lib_max - 1)
    assert (game.string[2].lib[second + board.board_size] == board.string_lib_max - 1)
    assert (game.string[3].lib[third + board.board_size] == board.string_lib_max - 1)

    string[0] = &game.string[2]
    string[1] = &game.string[3]

    board.add_stone(game, add_pos, board.S_BLACK, 1)
    
    assert (game.string_next[first], add_pos)
    assert (game.string_next[add_pos], board.string_pos_max - 1)
    assert (game.string_next[second], board.string_pos_max - 1)
    assert (game.string_next[third], board.string_pos_max - 1)

    board.merge_string(game, &game.string[1], string, 1)

    board.free_game(game)


def test_is_legal_stone_exists():
    cdef board.game_state_t *game
    cdef int pos = board.POS(4, 4, board.board_size)

    game = __initialize_game()

    board.put_stone(game, pos, board.S_BLACK)

    assert (board.is_legal(game, pos, board.S_BLACK) == False)
    assert (board.is_legal(game, pos, board.S_WHITE) == False)

    board.free_game(game)


def test_is_legal_nb4_empty_is_zero():
    game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                                 ". B . . . . .|"
                                 "B a B . . . .|"
                                 ". B . . . . .|"
                                 ". . . . . . .|"
                                 ". . . . . . .|"
                                 ". . . . . . .|"
                                 ". . . . . . .|")

    assert (board.is_legal(game, moves['a'], board.S_WHITE) == False)
    board.free_game(game)


def test_is_legal_nb4_empty_is_zero_edge():
    game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                                 "B a B . . . .|"
                                 ". B W B . . .|"
                                 ". . W B . . .|"
                                 ". . . . . . .|"
                                 ". W . . . . .|"
                                 ". . . . . . .|"
                                 ". . . . . . .|")

    assert (board.is_legal(game, moves['a'], board.S_WHITE) == False)
    board.free_game(game)


def test_string_add_and_remove_empty():
    game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                                 ". . . . . . .|"
                                 ". . . . . . .|"
                                 ". . . . d . .|"
                                 ". . c a . . .|"
                                 ". . . b . . .|"
                                 ". . . . . . .|"
                                 ". . . . . . .|")

    # put B[a]
    a = moves['a']
    board.put_stone(game, a, board.S_BLACK)
    string1 = &game.string[1]
    assert (string1.empties == 8)
    assert (string1.empty[0] == a-board.board_size)
    assert (string1.empty[a-board.board_size] == a-1)
    assert (string1.empty[a-1] == a+1)
    assert (string1.empty[a+1] == a+board.board_size)
    assert (string1.empty[a+board.board_size] == a-board.board_size-1)
    assert (string1.empty[a-board.board_size-1] == a-board.board_size+1)
    assert (string1.empty[a-board.board_size+1] == a+board.board_size-1)
    assert (string1.empty[a+board.board_size-1] == a+board.board_size+1)
    assert (string1.empty[a+board.board_size+1] == STRING_EMPTY_END)

    # put B[b]
    b = moves['b']
    board.put_stone(game, b, board.S_BLACK)
    assert (string1.empties == 10)
    assert (string1.empty[0] == a-board.board_size)
    assert (string1.empty[a-board.board_size] == a-1)
    assert (string1.empty[a-1] == a+1)
    assert (string1.empty[a+1] == a-board.board_size-1)
    assert (string1.empty[a-board.board_size-1] == a+board.board_size*2-1)
    assert (string1.empty[a+board.board_size*2-1] == a-board.board_size+1)
    assert (string1.empty[a-board.board_size+1] == a+board.board_size*2+1)
    assert (string1.empty[a+board.board_size*2+1] == a+board.board_size-1)
    assert (string1.empty[a+board.board_size-1] == a+board.board_size+1)
    assert (string1.empty[a+board.board_size+1] == a+board.board_size*2)
    assert (string1.empty[a+board.board_size*2] == STRING_EMPTY_END)

    # put W[c]
    c = moves['c']
    board.put_stone(game, c, board.S_WHITE)
    assert (string1.empties == 9)
    assert (string1.empty[0] == a-board.board_size)
    assert (string1.empty[a-board.board_size] == a+1)
    assert (string1.empty[a+1] == a-board.board_size-1)
    assert (string1.empty[a-board.board_size-1] == a+board.board_size*2-1)
    assert (string1.empty[a+board.board_size*2-1] == a-board.board_size+1)
    assert (string1.empty[a-board.board_size+1] == a+board.board_size*2+1)
    assert (string1.empty[a+board.board_size*2+1] == a+board.board_size-1)
    assert (string1.empty[a+board.board_size-1] == a+board.board_size+1)
    assert (string1.empty[a+board.board_size+1] == a+board.board_size*2)
    assert (string1.empty[a+board.board_size*2] == STRING_EMPTY_END)
    string2 = &game.string[2]
    assert (string2.empties == 6)
    assert (string2.empty[0] == c-board.board_size)
    assert (string2.empty[c-board.board_size] == c-1)
    assert (string2.empty[c-1] == c+board.board_size)
    assert (string2.empty[c+board.board_size] == c-board.board_size-1)
    assert (string2.empty[c-board.board_size-1] == c-board.board_size+1)
    assert (string2.empty[c-board.board_size+1] == c+board.board_size-1)
    assert (string2.empty[c+board.board_size-1] == STRING_EMPTY_END)

    # put W[d]
    d = moves['d']
    board.put_stone(game, d, board.S_WHITE)
    assert (string1.empties == 8)
    assert (string1.empty[0] == a-board.board_size)
    assert (string1.empty[a-board.board_size] == a+1)
    assert (string1.empty[a+1] == a-board.board_size-1)
    assert (string1.empty[a-board.board_size-1] == a+board.board_size*2-1)
    assert (string1.empty[a+board.board_size*2-1] == a+board.board_size*2+1)
    assert (string1.empty[a+board.board_size*2+1] == a+board.board_size-1)
    assert (string1.empty[a+board.board_size-1] == a+board.board_size+1)
    assert (string1.empty[a+board.board_size+1] == a+board.board_size*2)
    assert (string1.empty[a+board.board_size*2] == STRING_EMPTY_END)
    string3 = &game.string[3]
    assert (string3.empties == 7)
    assert (string3.empty[0] == d-board.board_size)
    assert (string3.empty[d-board.board_size] == d-1)
    assert (string3.empty[d-1] == d+1)
    assert (string3.empty[d+1] == d+board.board_size)
    assert (string3.empty[d+board.board_size] == d-board.board_size-1)
    assert (string3.empty[d-board.board_size-1] == d-board.board_size+1)
    assert (string3.empty[d-board.board_size+1] == d+board.board_size+1)
    assert (string3.empty[d+board.board_size+1] == STRING_EMPTY_END)


def test_string_merge_empty():
    game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                                 ". . . . . . .|"
                                 ". . . . . . .|"
                                 ". . . d . . .|"
                                 ". . . a . . .|"
                                 ". . b e c . .|"
                                 ". . . . . . .|"
                                 ". . . . . . .|")

    # put B[a]
    a = moves['a']
    board.put_stone(game, a, board.S_BLACK)
    string1 = &game.string[1]
    assert (string1.empties == 8)
    assert (string1.empty[0] == a-board.board_size)
    assert (string1.empty[a-board.board_size] == a-1)
    assert (string1.empty[a-1] == a+1)
    assert (string1.empty[a+1] == a+board.board_size)
    assert (string1.empty[a+board.board_size] == a-board.board_size-1)
    assert (string1.empty[a-board.board_size-1] == a-board.board_size+1)
    assert (string1.empty[a-board.board_size+1] == a+board.board_size-1)
    assert (string1.empty[a+board.board_size-1] == a+board.board_size+1)
    assert (string1.empty[a+board.board_size+1] == STRING_EMPTY_END)

    # put B[b]
    b = moves['b']
    board.put_stone(game, b, board.S_BLACK)
    string2 = &game.string[2]
    assert (string2.empties == 7)
    assert (string2.empty[0] == b-board.board_size)
    assert (string2.empty[b-board.board_size] == b-1)
    assert (string2.empty[b-1] == b+1)
    assert (string2.empty[b+1] == b+board.board_size)
    assert (string2.empty[b+board.board_size] == b-board.board_size-1)
    assert (string2.empty[b-board.board_size-1] == b+board.board_size-1)
    assert (string2.empty[b+board.board_size-1] == b+board.board_size+1)
    assert (string1.empties == 7)
    assert (string1.empty[0] == a-board.board_size)
    assert (string1.empty[a-board.board_size] == a-1)
    assert (string1.empty[a-1] == a+1)
    assert (string1.empty[a+1] == a+board.board_size)
    assert (string1.empty[a+board.board_size] == a-board.board_size-1)
    assert (string1.empty[a-board.board_size-1] == a-board.board_size+1)
    assert (string1.empty[a-board.board_size+1] == a+board.board_size+1)
    assert (string1.empty[a+board.board_size+1] == STRING_EMPTY_END)
    assert (string1.empty[a+board.board_size-1] == 0) # removed by 'b'

    # put B[c]
    c = moves['c']
    board.put_stone(game, c, board.S_BLACK)
    string3 = &game.string[3]
    assert (string3.empties == 7)
    assert (string3.empty[0] == c-board.board_size)
    assert (string3.empty[c-board.board_size] == c-1)
    assert (string3.empty[c-1] == c+1)
    assert (string3.empty[c+1] == c+board.board_size)
    assert (string3.empty[c+board.board_size] == c-board.board_size+1)
    assert (string3.empty[c-board.board_size+1] == c+board.board_size-1)
    assert (string3.empty[c+board.board_size-1] == c+board.board_size+1)
    assert (string1.empties == 6)
    assert (string1.empty[0] == a-board.board_size)
    assert (string1.empty[a-board.board_size] == a-1)
    assert (string1.empty[a-1] == a+1)
    assert (string1.empty[a+1] == a+board.board_size)
    assert (string1.empty[a+board.board_size] == a-board.board_size-1)
    assert (string1.empty[a-board.board_size-1] == a-board.board_size+1)
    assert (string1.empty[a-board.board_size+1] == STRING_EMPTY_END)
    assert (string1.empty[a+board.board_size-1] == 0) # removed by 'c'

    # put B[d]
    d = moves['d']
    board.put_stone(game, d, board.S_BLACK)
    string1 = &game.string[1]
    assert (string1.empties == 8)
    assert (string1.empty[0] == d-board.board_size)
    assert (string1.empty[d-board.board_size] == d+board.board_size-1)
    assert (string1.empty[d+board.board_size-1] == d+board.board_size+1)
    assert (string1.empty[d+board.board_size+1] == d+board.board_size*2)
    assert (string1.empty[d+board.board_size*2] == d-1)
    assert (string1.empty[d-1] == d+1)
    assert (string1.empty[d+1] == d-board.board_size-1)
    assert (string1.empty[d-board.board_size-1] == d-board.board_size+1)
    assert (string1.empty[d-board.board_size+1] == STRING_EMPTY_END)

    # put B[e]
    e = moves['e']
    board.put_stone(game, e, board.S_BLACK)
    string1 = &game.string[1]
    assert (string1.empties == 16)
    assert (string1.empty[0] == e+board.board_size)
    assert (string1.empty[e+board.board_size] == e-board.board_size*3)
    assert (string1.empty[e-board.board_size*3] == e+board.board_size+2)
    assert (string1.empty[e+board.board_size+2] == e-board.board_size-1)
    assert (string1.empty[e-board.board_size-1] == e-2)
    assert (string1.empty[e-2] == e-board.board_size+1)
    assert (string1.empty[e-board.board_size+1] == e-board.board_size*2-1)
    assert (string1.empty[e-board.board_size*2-1] == e+2)
    assert (string1.empty[e+2] == e+board.board_size-1)
    assert (string1.empty[e+board.board_size-1] == e-board.board_size*2+1)
    assert (string1.empty[e-board.board_size*2+1] == e-board.board_size*3-1)
    assert (string1.empty[e-board.board_size*3-1] == e-board.board_size-2)
    assert (string1.empty[e-board.board_size-2] == e+board.board_size-2)
    assert (string1.empty[e+board.board_size-2] == e+board.board_size+1)
    assert (string1.empty[e-board.board_size*3+1] == e-board.board_size+2)
    assert (string1.empty[e-board.board_size+2] == STRING_EMPTY_END)

    assert (string1.flag == True)
    assert (string2.flag == False)
    assert (string3.flag == False)

    board.free_game(game)


def test_ko():
    cdef board.game_state_t *game

    game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                                 ". B W . . . .|"
                                 "B b a W . . .|"
                                 ". B W . . . .|"
                                 ". . . c d . .|"
                                 ". . . . . . .|"
                                 ". . . . . . .|"
                                 ". . . . . . .|")

    game.current_color = board.S_BLACK

    board.put_stone(game, moves['a'], game.current_color)
    game.current_color = board.FLIP_COLOR(game.current_color)
    board.put_stone(game, moves['b'], game.current_color)
    game.current_color = board.FLIP_COLOR(game.current_color)

    assert (board.is_legal(game, moves['a'], game.current_color) == False)

    board.put_stone(game, moves['c'], game.current_color)
    game.current_color = board.FLIP_COLOR(game.current_color)
    board.put_stone(game, moves['d'], game.current_color)
    game.current_color = board.FLIP_COLOR(game.current_color)

    assert (board.is_legal(game, moves['a'], game.current_color) == True)

    board.free_game(game)

    game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                                 "W B B B B W B B B|"
                                 "B . B W W W W B a|"
                                 "B B B B B . W W b|"
                                 "W B W B B B B W W|"
                                 "W W W B W W B B B|"
                                 "W B W W W W B B .|"
                                 "B B W B . W W B W|"
                                 "W W W W W W W B .|"
                                 "B W W . W B B B B|")

    game.current_color = board.S_WHITE

    board.put_stone(game, moves['a'], game.current_color)
    game.current_color = board.FLIP_COLOR(game.current_color)
    board.put_stone(game, moves['b'], game.current_color)
    game.current_color = board.FLIP_COLOR(game.current_color)

    assert (board.is_legal(game, moves['a'], game.current_color) == False)


cdef board.game_state_t* __initialize_game(int board_size=9):
    cdef board.game_state_t *game

    board.set_board_size(board_size)

    game = board.allocate_game()
    board.initialize_board(game)

    return game


cdef board.string_t* __initialize_string(board.game_state_t *game, int origin, char color):
    cdef board.string_t *string
    cdef string_id = 1

    while game.string[string_id].flag:
        string_id += 1

    string = &game.string[string_id]

    board.fill_n_short(string.lib, board.string_lib_max, 0)
    board.fill_n_short(string.neighbor, board.max_neighbor, 0)
    string.lib[0] = board.string_lib_max - 1
    string.neighbor[0] = board.NEIGHBOR_END
    string.libs = 0
    string.color = color
    string.origin = origin 
    string.size = 1
    string.neighbors = 0

    game.string_id[origin] = string_id
    game.string_next[origin] = board.string_pos_max - 1

    return string

