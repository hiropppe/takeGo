import numpy as np

cimport numpy as np

from libc.stdlib cimport malloc, free

from nose.tools import ok_, eq_

cimport board 
cimport policy_feature
cimport parseboard
cimport printer


def test_captured_1():
    game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                             "d b c . . . .|"
                             "B W a . . . .|"
                             ". B . . . . .|"
                             ". . . . . . .|"
                             ". . . . . . .|"
                             ". . . . . W .|")
    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    game.current_color = board.S_BLACK

    # 'a' should catch white in a ladder, but not 'b'
    policy_feature.update(feature, game)
    eq_(planes[44, pure_moves['a']], 1)
    eq_(planes[44, pure_moves['b']], 0)

    # 'b' should not be an escape move for white after 'a'
    board.do_move(game, moves['a'])
    policy_feature.update(feature, game)
    eq_(planes[45, pure_moves['b']], 0)

    # W at 'b', check 'c' and 'd'
    board.do_move(game, moves['b'])
    policy_feature.update(feature, game)
    eq_(planes[44, pure_moves['c']], 1)
    eq_(planes[44, pure_moves['d']], 0)

    board.free_game(game)
    policy_feature.free_feature(feature)


def test_breaker_1():
    game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                                 ". B . . . . .|"
                                 "B W a . . W .|"
                                 "B b . . . . .|"
                                 ". c . . . . .|"
                                 ". . . . . . .|"
                                 ". . . . . W .|"
                                 ". . . . . . .|")
    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    game.current_color = board.S_BLACK
    
    # 'a' should not be a ladder capture, nor 'b'
    policy_feature.update(feature, game)
    eq_(planes[44, pure_moves['a']], 0)
    eq_(planes[44, pure_moves['b']], 0)

    # after 'a', 'b' should be an escape
    board.do_move(game, moves['a'])
    policy_feature.update(feature, game)
    eq_(planes[45, pure_moves['b']], 1)

    # after 'b', 'c' should not be a capture
    board.do_move(game, moves['b'])
    policy_feature.update(feature, game)
    eq_(planes[44, pure_moves['c']], 0)

    board.free_game(game)
    policy_feature.free_feature(feature)


def test_missing_ladder_breaker_1():
    game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                                 ". B . . . . .|"
                                 "B W B . . W .|"
                                 "B a c . . . .|"
                                 ". b . . . . .|"
                                 ". . . . . . .|"
                                 ". W . . . . .|"
                                 ". . . . . . .|")
    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    game.current_color = board.S_WHITE

    # a should not be an escape move for white
    policy_feature.update(feature, game)
    eq_(planes[45, pure_moves['a']], 0)

    # after 'a', 'b' should still be a capture ...
    # ... but 'c' should not
    board.do_move(game, moves['a'])
    policy_feature.update(feature, game)
    eq_(planes[44, pure_moves['b']], 1)
    eq_(planes[44, pure_moves['c']], 0)

    board.free_game(game)
    policy_feature.free_feature(feature)


def test_capture_to_escape_1():
    cdef int ladder_moves[1]
    ladder_moves[0] = 0 
    game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                                "d O X c . . .|"
                                ". X O X . . .|"
                                ". . O X . . .|"
                                ". . a . . . .|"
                                ". O . . . . .|"
                                ". . . . . . .|")
    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    game.current_color = board.S_BLACK

    # 'a' is not a capture because of ataris
    #policy_feature.is_ladder_capture(game, game.string_id[48], 59, 47, feature.search_games, 0)
    #policy_feature.is_ladder_escape(game, game.string_id[25], 24, False, feature.search_games, 0, ladder_moves)
    policy_feature.update(feature, game)
    eq_(planes[44, pure_moves['a']], 0)
    eq_(planes[45, pure_moves['c']], 1)
    eq_(planes[45, pure_moves['d']], 1)

    board.free_game(game)
    policy_feature.free_feature(feature)


def test_throw_in_1():
    game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                                "X a O X . .|"
                                "b O O X . .|"
                                "O O X X . .|"
                                "X X . . . .|"
                                ". . . . . .|"
                                ". . . O . .|")
    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    game.current_color = board.S_BLACK
    
    # 'a' or 'b' will capture
    policy_feature.update(feature, game)
    eq_(planes[44, pure_moves['a']], 1)
    eq_(planes[44, pure_moves['b']], 1)

    board.do_move(game, moves['a'])
    eq_(planes[45, pure_moves['b']], 0)

    board.free_game(game)
    policy_feature.free_feature(feature)


def test_snapback_1():
    game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                                ". . . . . . . . .|"
                                ". . . . . . . . .|"
                                ". . X X X . . . .|"
                                ". . O . . . . . .|"
                                ". . O X . . . . .|"
                                ". . X O a . . . .|"
                                ". . X O X . . . .|"
                                ". . . X . . . . .|"
                                ". . . . . . . . .|")
    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    game.current_color = board.S_WHITE

    policy_feature.update(feature, game)
    eq_(planes[45, pure_moves['a']], 0)

    board.free_game(game)
    policy_feature.free_feature(feature)


def test_two_captures():
    game = board.allocate_game()
    moves, pure_moves = parseboard.parse(game,
                            ". . . . . .|"
                            ". . . . . .|"
                            ". . a b . .|"
                            ". X O O X .|"
                            ". . X X . .|"
                            ". . . . . .|")
    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    game.current_color = board.S_BLACK

    policy_feature.update(feature, game)
    eq_(planes[44, pure_moves['a']], 1)
    eq_(planes[44, pure_moves['b']], 1)

    board.free_game(game)
    policy_feature.free_feature(feature)


def test_two_escapes():
    game = board.allocate_game()
    moves, pure_moves = parseboard.parse(game,
                            ". . X . . .|"
                            ". X O a . .|"
                            ". X c X . .|"
                            ". O X b . .|"
                            ". . O . . .|"
                            ". . . . . .|")
    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    board.put_stone(game, moves['c'], board.S_WHITE)

    game.current_color = board.S_WHITE

    policy_feature.update(feature, game)
    eq_(planes[45, pure_moves['a']], 1)
    eq_(planes[45, pure_moves['b']], 1)

    board.free_game(game)
    policy_feature.free_feature(feature)


def test_escapes_1():
    game = board.allocate_game()
    moves, pure_moves = parseboard.parse(game,
                            "B . . . . . . . . . .|"
                            ". . . . . . . . . . .|"
                            ". . . . . . . . . . .|"
                            ". . . . . . . . . . .|"
                            ". . . . . . . . . . .|"
                            ". . . . . . . . . . .|"
                            ". . . . . . . B . . .|"
                            ". . . . . . B W W . W|"
                            "B . . . . a W B b W .|"
                            ". . . . . B W W B . .|"
                            ". . . . . . B B . . .|")
    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    game.current_color = board.S_WHITE

    policy_feature.update(feature, game)
    eq_(planes[45, pure_moves['a']], 1)
    eq_(planes[45, pure_moves['b']], 1)

    board.free_game(game)
    policy_feature.free_feature(feature)
