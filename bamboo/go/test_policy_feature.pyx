import numpy as np

cimport numpy as np

from libc.stdlib cimport malloc, free

from nose.tools import ok_, eq_

cimport board 
cimport policy_feature
cimport parseboard
cimport printer

from bamboo.go.board cimport is_legal_not_eye


def test_stone_color():
    game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                             "a . .|"
                             ". b .|"
                             ". . c|")
    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    board.put_stone(game, moves['a'], board.S_BLACK)
    board.put_stone(game, moves['b'], board.S_WHITE)

    game.current_color = board.S_BLACK
    policy_feature.update(feature, game)

    eq_(planes[0, pure_moves['a']], 1)
    eq_(planes[1, pure_moves['b']], 1)
    eq_(planes[2, pure_moves['c']], 1)
    eq_(planes[0].sum(), 1)
    eq_(planes[1].sum(), 1)
    eq_(planes[2].sum(), 7)

    game.current_color = board.S_WHITE
    policy_feature.update(feature, game)

    eq_(planes[0, pure_moves['b']], 1)
    eq_(planes[1, pure_moves['a']], 1)
    eq_(planes[2, pure_moves['c']], 1)
    eq_(planes[0].sum(), 1)
    eq_(planes[1].sum(), 1)
    eq_(planes[2].sum(), 7)

    board.free_game(game)
    policy_feature.free_feature(feature)


def test_turns_since():
    game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                             ". . . . .|"
                             ". a b c .|"
                             ". d e f .|"
                             ". g . . .|"
                             ". . . . .|")
    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    game.current_color = board.S_BLACK

    # move 'a'
    board.do_move(game, moves['a'])
    policy_feature.update(feature, game)
    eq_(planes[4, pure_moves['a']], 1) # 0 age
    eq_(planes[4].sum(), 1)

    # move 'b'
    board.do_move(game, moves['b'])
    policy_feature.update(feature, game)
    eq_(planes[5, pure_moves['a']], 1) # 1 age
    eq_(planes[4, pure_moves['b']], 1) # 0 age
    eq_(planes[5].sum(), 1)
    eq_(planes[4].sum(), 1)

    # PASS
    board.do_move(game, board.PASS)
    board.do_move(game, board.PASS)
    policy_feature.update(feature, game)
    eq_(planes[7, pure_moves['a']], 1) # 3 age
    eq_(planes[6, pure_moves['b']], 1) # 2 age
    eq_(planes[7].sum(), 1)
    eq_(planes[6].sum(), 1)

    board.do_move(game, moves['c'])
    board.do_move(game, moves['d'])
    board.do_move(game, moves['e'])
    board.do_move(game, moves['f'])
    policy_feature.update(feature, game)
    eq_(planes[11, pure_moves['a']], 1) # 7 age
    eq_(planes[10, pure_moves['b']], 1) # 6 age
    eq_(planes[7, pure_moves['c']], 1)  # 3 age
    eq_(planes[6, pure_moves['d']], 1)  # 2 age
    eq_(planes[4, pure_moves['f']], 1)  # 0 age

    board.do_move(game, moves['g'])
    policy_feature.update(feature, game)
    eq_(planes[11, pure_moves['a']], 1) # 7 age
    eq_(planes[11, pure_moves['b']], 1) # 7 age
    eq_(planes[8, pure_moves['c']], 1)  # 4 age
    eq_(planes[7, pure_moves['d']], 1)  # 3 age
    eq_(planes[5, pure_moves['f']], 1)  # 1 age
    eq_(planes[4, pure_moves['g']], 1)  # 0 age

    board.free_game(game)
    policy_feature.free_feature(feature)


def test_liberties():
    game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                             ". . . . .|"
                             ". . a . .|"
                             ". . b . .|"
                             ". . c . .|"
                             ". . d . .|")
    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    board.put_stone(game, moves['a'], board.S_BLACK)
    policy_feature.update(feature, game)
    eq_(planes[12 + 4 - 1, pure_moves['a']], 1)

    board.put_stone(game, moves['b'], board.S_BLACK)
    board.put_stone(game, moves['c'], board.S_BLACK)
    policy_feature.update(feature, game)
    eq_(planes[12 + 8 - 1, pure_moves['a']], 1)
    eq_(planes[12 + 8 - 1, pure_moves['b']], 1)
    eq_(planes[12 + 8 - 1, pure_moves['c']], 1)

    board.put_stone(game, moves['d'], board.S_BLACK)
    policy_feature.update(feature, game)
    eq_(planes[12 + 8 - 1, pure_moves['a']], 1)
    eq_(planes[12 + 8 - 1, pure_moves['b']], 1)
    eq_(planes[12 + 8 - 1, pure_moves['c']], 1)
    eq_(planes[12 + 8 - 1, pure_moves['d']], 1)

    board.free_game(game)
    policy_feature.free_feature(feature)


def test_capture_size():
    game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                             "O a . . O X X X X|"
                             "X . . . . O X X X|"
                             ". . . . . . O X X|"
                             ". . . . . . . O c|"
                             ". . . . . . . . .|"
                             ". . . . . . . . .|"
                             ". . . . . . . . .|"
                             "X X X X X X X X .|"
                             "O O O O O O O O b|")
    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    game.current_color = board.S_BLACK
    policy_feature.update(feature, game)
    eq_(planes[20 + 1, pure_moves['a']], 1)
    eq_(planes[20 + 1].sum(), 1)
    eq_(planes[20 + 7, pure_moves['b']], 1)
    eq_(planes[20 + 7].sum(), 1)

    game.current_color = board.S_WHITE
    policy_feature.update(feature, game)

    eq_(planes[20 + 7, pure_moves['c']], 1)
    eq_(planes[20 + 7].sum(), 1)

    board.free_game(game)
    policy_feature.free_feature(feature)


def test_self_atari_size():
    pass
    """
    game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                             "O a . . O X X X X|"
                             "X . . . . O X X X|"
                             ". . . . . . O X X|"
                             ". . . . . . . O c|"
                             ". . . . . . . . .|"
                             ". . . . . . . . .|"
                             ". . . . . . . . .|"
                             "X X X X X X X X .|"
                             "O O O O O O O O b|")
    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    game.current_color = board.S_WHITE
    policy_feature.update(feature, game)

    eq_(planes[28 + 1 - 1, pure_moves['a']], 1)
    eq_(planes[28 + 1 - 1].sum(), 1)
    eq_(planes[28 + 8 - 1, pure_moves['b']], 1)
    eq_(planes[28 + 8 - 1].sum(), 1)

    game.current_color = board.S_BLACK
    policy_feature.update(feature, game)

    eq_(planes[28 + 8 - 1, pure_moves['c']], 1)
    eq_(planes[28 + 8 - 1].sum(), 1)

    board.free_game(game)
    policy_feature.free_feature(feature)
    """

def test_liberties_after_move():
    game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                             ". . . . .|"
                             ". . a . .|"
                             ". . b . .|"
                             ". . c . .|"
                             ". . d e .|")
    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    game.current_color = board.S_BLACK

    # after B 'a'
    policy_feature.update(feature, game)
    eq_(planes[36 + 4 - 1, pure_moves['a']], 1)

    # after B 'b'
    board.put_stone(game, moves['a'], board.S_BLACK)
    policy_feature.update(feature, game)
    eq_(planes[36 + 6 - 1, pure_moves['b']], 1)

    # after B 'c'
    board.put_stone(game, moves['b'], board.S_BLACK)
    policy_feature.update(feature, game)
    eq_(planes[36 + 8 - 1, pure_moves['c']], 1)

    # after B 'd'
    board.put_stone(game, moves['c'], board.S_BLACK)
    policy_feature.update(feature, game)
    eq_(planes[36 + 8 - 1, pure_moves['d']], 1)

    game.current_color = board.S_WHITE

    # after W 'e'
    board.put_stone(game, moves['d'], board.S_BLACK)
    policy_feature.update(feature, game)
    eq_(planes[36 + 2 - 1, pure_moves['e']], 1)

    board.free_game(game)
    policy_feature.free_feature(feature)


def test_liberties_after_move_1():
    game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                             ". . . . .|"
                             ". W B . .|"
                             ". W a B .|"
                             ". . . . .|"
                             ". . . . .|")
    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    game.current_color = board.S_BLACK

    policy_feature.update(feature, game)
    eq_(planes[36 + 5 - 1, pure_moves['a']], 1)


def test_liberties_after_move_dupe_empty():
    game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                             ". . . . . .|"
                             ". . . W . .|"
                             ". . a W . .|"
                             ". B B W . .|"
                             ". . . B W .|"
                             ". . . . . .|")
    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    game.current_color = board.S_BLACK

    policy_feature.update(feature, game)
    eq_(planes[36 + 5 - 1, pure_moves['a']], 1)


def test_liberties_after_move_captured():
    game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                             ". . . . . .|"
                             ". . B B . .|"
                             ". B W W B .|"
                             ". . a B W .|"
                             ". . B W W .|"
                             ". . . . . .|")
    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    game.current_color = board.S_BLACK

    policy_feature.update(feature, game)
    eq_(planes[36 + 5 - 1, pure_moves['a']], 1)


def test_liberties_after_move_captured_1():
    game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                             "W W B B B B W|"
                             "W B B W . B W|"
                             "B W W W W W W|"
                             "B B W B B B a|"
                             "B W B B W W .|"
                             "B W W B W . .|"
                             ". B . W . . .|")
    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    game.current_color = board.S_WHITE

    policy_feature.update(feature, game)
    eq_(planes[36 + 6 - 1, pure_moves['a']], 1)


def test_liberties_after_move_captured_2():
    game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                             ". . B . B B W . .|"
                             "B B B B W W B B .|"
                             "B W W W W W B B B|"
                             "W W B B W B B W W|"
                             "a W W B W W W B .|"
                             "B W W B B B . B .|"
                             "B W W W W B B . .|"
                             "B B W W B B . . .|"
                             "B W W B B . . . .|")
    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    game.current_color = board.S_WHITE

    policy_feature.update(feature, game)
    eq_(planes[36 + 5 - 1, pure_moves['a']], 1)


def test_sensibleness_eye_1():
    game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                             "B W W W B B .|"
                             "B B . W B . .|"
                             "W B W W B . .|"
                             "B W W B W B .|"
                             "B W B B W W W|"
                             "B B a B B B B|"
                             "W B b c d . .|")
    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    board.put_stone(game, moves['a'], board.S_WHITE)
    board.put_stone(game, moves['b'], board.S_BLACK)
    board.put_stone(game, moves['c'], board.S_WHITE)
    board.put_stone(game, moves['d'], board.S_BLACK)

    game.current_color = board.S_BLACK
    
    policy_feature.update(feature, game)
    eq_(planes[46, pure_moves['a']], 0)
    eq_(planes[46, pure_moves['c']], 0)

