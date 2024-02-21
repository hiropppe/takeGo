from bamboo.board cimport S_BLACK, S_WHITE
from bamboo.board cimport FLIP_COLOR
from bamboo.board cimport game_state_t
from bamboo.board cimport allocate_game, free_game, put_stone
from bamboo.parseboard cimport parse
from bamboo.zobrist_hash cimport initialize_hash
from bamboo.nakade cimport initialize_nakade_hash, nakade_at_captured_stone


def setup():
    initialize_hash()


def test_nakade3():
    cdef game_state_t *game = allocate_game()

    (moves, pure_moves) = parse(game,
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . W W W . .|"
                              ". W B a B W .|"
                              ". . W b W . .|"
                              ". . . . . . .|"
                              ". . . . . . .|")
    initialize_nakade_hash()

    game.current_color = S_BLACK
    put_stone(game, moves['a'], game.current_color)

    game.current_color = S_WHITE
    put_stone(game, moves['b'], game.current_color)

    assert (nakade_at_captured_stone(game, <int>S_BLACK) == moves['a'])

    (moves, pure_moves) = parse(game,
                              "W a W b . . .|"
                              "B B B . . . .|"
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . . . . . .|")
    initialize_nakade_hash()

    game.current_color = S_WHITE
    put_stone(game, moves['a'], game.current_color)

    game.current_color = S_BLACK
    put_stone(game, moves['b'], game.current_color)

    assert (nakade_at_captured_stone(game, <int>S_WHITE) == moves['a'])

    (moves, pure_moves) = parse(game,
                              "a W b . . . .|"
                              "W B . . . . .|"
                              "B . . . . . .|"
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . . . . . .|"
                              ". . . . . . .|")
    initialize_nakade_hash()

    game.current_color = S_WHITE
    put_stone(game, moves['a'], game.current_color)

    game.current_color = S_BLACK
    put_stone(game, moves['b'], game.current_color)

    assert (nakade_at_captured_stone(game, <int>S_WHITE) == moves['a'])


def test_nakade4():
    pass


def test_nakade5():
    pass


def test_nakade6():
    pass
