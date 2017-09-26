from bamboo.board cimport S_BLACK, S_WHITE, PASS
from bamboo.board cimport game_state_t
from bamboo.board cimport allocate_game, put_stone
from bamboo.parseboard cimport parse
from bamboo.printer cimport print_board
from bamboo.sgf_util cimport save_gamestate_to_sgf

def test_save_gamestate_to_sgf():
    cdef game_state_t *game = allocate_game()

    moves, pure_moves = parse(game,
        ". . . . . . . . . . . . . . . . . . . |"
        ". . . . W . . . . . . . . W . . . B . |"
        ". . . . . B W W . . . . B W . . W B . |"
        ". . . W . W . B . B . . B W . W W W B |"
        "W W W . . W . . . . . B W W . . . B . |"
        "W B B B . . . . . . B W W W B B B W . |"
        "B . B W . W . . . B . B W B W W W W . |"
        ". B B W . . . B . . B . B B W B W . W |"
        ". . B . . W W B . . . . . . B . B W . |"
        "B . B W W B B W . B . . . . . B B W B |"
        ". B W B B B B W W B . . . . . . W B . |"
        "W W W B . B W . W . . . . . . . B . . |"
        ". . W B B W . . W B . B . . . . . B . |"
        ". . W W B . W . W . . . . B . B B . B |"
        ". . . W B . W B . . . . W B . B . . B |"
        ". W W B . B W B . . . B W W W B B . B |"
        ". W B B . . B W . B B . W . . W W B B |"
        "W B . . . . . . . B W W . . W . W B W |"
        ". B . . . . . . . . . . . . . W . W . |")

    put_stone(game, PASS, S_WHITE)
    put_stone(game, PASS, S_BLACK)

    save_gamestate_to_sgf(game, '/tmp/', 'test.sgf', '', '')
