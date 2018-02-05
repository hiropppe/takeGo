import numpy as np

cimport numpy as np

from libc.stdlib cimport malloc, free

from nose.tools import ok_, eq_

cimport board 
cimport policy_feature
cimport parseboard
cimport printer

from bamboo.policy_feature cimport MAX_POLICY_PLANES
from bamboo.tree_search cimport tree_node_t


def test_captured_1():
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    for i in range(361):
        node.children[i] = <tree_node_t *>malloc(sizeof(tree_node_t))
    (moves, pure_moves) = parseboard.parse(node.game,
                             "d b c . . . .|"
                             "B W a . . . .|"
                             ". B . . . . .|"
                             ". . . . . . .|"
                             ". . . . . . .|"
                             ". . . . . W .|")
    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    node.game.current_color = board.S_BLACK

    # 'a' should catch white in a ladder, but not 'b'
    policy_feature.update(feature, node)
    eq_(planes[44, pure_moves['a']], 1)
    eq_(planes[44, pure_moves['b']], 0)

    # 'b' should not be an escape move for white after 'a'
    board.do_move(node.game, moves['a'])
    policy_feature.update(feature, node)
    eq_(planes[45, pure_moves['b']], 0)

    # W at 'b', check 'c' and 'd'
    board.do_move(node.game, moves['b'])
    policy_feature.update(feature, node)
    eq_(planes[44, pure_moves['c']], 1)
    eq_(planes[44, pure_moves['d']], 0)

    board.free_game(node.game)
    policy_feature.free_feature(feature)


def test_breaker_1():
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(node.game,
                                 ". B . . . . .|"
                                 "B W a . . W .|"
                                 "B b . . . . .|"
                                 ". c . . . . .|"
                                 ". . . . . . .|"
                                 ". . . . . W .|"
                                 ". . . . . . .|")
    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    node.game.current_color = board.S_BLACK
    
    # 'a' should not be a ladder capture, nor 'b'
    policy_feature.update(feature, node)
    eq_(planes[44, pure_moves['a']], 0)
    eq_(planes[44, pure_moves['b']], 0)

    # after 'a', 'b' should be an escape
    board.do_move(node.game, moves['a'])
    policy_feature.update(feature, node)
    eq_(planes[45, pure_moves['b']], 1)

    # after 'b', 'c' should not be a capture
    board.do_move(node.game, moves['b'])
    policy_feature.update(feature, node)
    eq_(planes[44, pure_moves['c']], 0)

    board.free_game(node.game)
    policy_feature.free_feature(feature)


def test_missing_ladder_breaker_1():
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    for i in range(361):
        node.children[i] = <tree_node_t *>malloc(sizeof(tree_node_t))
    (moves, pure_moves) = parseboard.parse(node.game,
                                 ". B . . . . .|"
                                 "B W B . . W .|"
                                 "B a c . . . .|"
                                 ". b . . . . .|"
                                 ". . . . . . .|"
                                 ". W . . . . .|"
                                 ". . . . . . .|")
    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    node.game.current_color = board.S_WHITE

    # a should not be an escape move for white
    policy_feature.update(feature, node)
    eq_(planes[45, pure_moves['a']], 0)

    # after 'a', 'b' should still be a capture ...
    # ... but 'c' should not
    board.do_move(node.game, moves['a'])
    policy_feature.update(feature, node)
    eq_(planes[44, pure_moves['b']], 1)
    eq_(planes[44, pure_moves['c']], 0)

    board.free_game(node.game)
    policy_feature.free_feature(feature)


def test_missing_ladder_breaker_2():
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    for i in range(361):
        node.children[i] = <tree_node_t *>malloc(sizeof(tree_node_t))
    # BBS(MCTS-CNN) put 'b' at after 'a' in actual game
    (moves, pure_moves) = parseboard.parse(node.game,
        ". . . . . . . . . . . . . . . . . . .|"
        ". . . W W W B B . . . . . . . . . . .|"
        ". . B . B W B . B . B B . . . . W . .|"
        ". . B . B W B B W W W . . . . B . W .|"
        ". . . . B W W W . . . W . . . . . . .|"
        ". . . . . W . . . B . . . . B . W . .|"
        ". . . B . . . . . . W . . . . . . . .|"
        ". . . . . . . . . . . . . . B . W . .|"
        ". . . . . . . . . . . . . . . . . B .|"
        ". . B B . . . . . . . . . . . B . . .|"
        ". . B W W . . . . . . . . . . . . . .|"
        ". B W . W . . . . . a . . . . . . . .|"
        ". . B W W . . . . b W B B . . . . . .|"
        ". . B B W B . . . B B W W . B . B B .|"
        ". . . W B W B . . B W . . . . . W . .|"
        ". . . W . W B . B W W . . . . W . . .|"
        ". . . . . W . . W . . . . W . . . . .|"
        ". . . . . . . . . . . . . . . . . . .|"
        ". . . . . . . . . . . . . . . . . . .|")

 
    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    node.game.current_color = board.S_BLACK

    # 'a' should catch white in a ladder, but not 'b'
    policy_feature.update(feature, node)
    eq_(planes[44, pure_moves['a']], 1)
    eq_(planes[44, pure_moves['b']], 0)

    # 'b' should not be an escape move for white after 'a'
    board.do_move(node.game, moves['a'])
    policy_feature.update(feature, node)
    eq_(planes[45, pure_moves['b']], 0)

 
def test_capture_to_escape_1():
    cdef tree_node_t *node
    cdef int ladder_moves[1]
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    ladder_moves[0] = 0 
    (moves, pure_moves) = parseboard.parse(node.game,
                                "c O X b . . .|"
                                ". X O X . . .|"
                                ". . O X . . .|"
                                ". . a . . . .|"
                                ". O . . . . .|"
                                ". . . . . . .|")
    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    node.game.current_color = board.S_BLACK

    # 'a' is not a capture because of ataris
    #policy_feature.is_ladder_capture(node.game, game.string_id[48], 59, 47, feature.search_games, 0)
    #policy_feature.is_ladder_escape(node.game, game.string_id[25], 24, False, feature.search_games, 0, ladder_moves)
    policy_feature.update(feature, node)
    eq_(planes[44, pure_moves['a']], 0)
    eq_(planes[45, pure_moves['b']], 1)

    # search only neighbor string of 'c'
    eq_(planes[45, pure_moves['c']], 0)
    # search all escape routes
    #eq_(planes[45, pure_moves['c']], 1)

    board.free_game(node.game)
    policy_feature.free_feature(feature)


def test_throw_in_1():
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(node.game,
                                "X a O X . .|"
                                "b O O X . .|"
                                "O O X X . .|"
                                "X X . . . .|"
                                ". . . . . .|"
                                ". . . O . .|")
    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    node.game.current_color = board.S_BLACK
    
    # 'a' or 'b' will capture
    policy_feature.update(feature, node)
    eq_(planes[44, pure_moves['a']], 1)
    eq_(planes[44, pure_moves['b']], 1)

    board.do_move(node.game, moves['a'])
    eq_(planes[45, pure_moves['b']], 0)

    board.free_game(node.game)
    policy_feature.free_feature(feature)


def test_snapback_1():
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    for i in range(361):
        node.children[i] = <tree_node_t *>malloc(sizeof(tree_node_t))
    (moves, pure_moves) = parseboard.parse(node.game,
                                ". . . . . . . . .|"
                                ". . . . . . . . .|"
                                ". . X X X . . . .|"
                                ". . O . . . . . .|"
                                ". . O X . . . . .|"
                                ". . X O a . . . .|"
                                ". . X O X . . . .|"
                                ". . . X . . . . .|"
                                ". . . . . . . . .|")
    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    node.game.current_color = board.S_WHITE

    policy_feature.update(feature, node)
    eq_(planes[45, pure_moves['a']], 0)

    board.free_game(node.game)
    policy_feature.free_feature(feature)


def test_two_captures():
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    moves, pure_moves = parseboard.parse(node.game,
                            ". . . . . .|"
                            ". . . . . .|"
                            ". . a b . .|"
                            ". X O O X .|"
                            ". . X X . .|"
                            ". . . . . .|")
    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    node.game.current_color = board.S_BLACK

    policy_feature.update(feature, node)
    eq_(planes[44, pure_moves['a']], 1)
    eq_(planes[44, pure_moves['b']], 1)

    board.free_game(node.game)
    policy_feature.free_feature(feature)


def test_two_escapes():
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    moves, pure_moves = parseboard.parse(node.game,
                            ". . X . . .|"
                            ". X O a . .|"
                            ". X c X . .|"
                            ". O X b . .|"
                            ". . O . . .|"
                            ". . . . . .|")
    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    board.put_stone(node.game, moves['c'], board.S_WHITE)

    node.game.current_color = board.S_WHITE

    policy_feature.update(feature, node)
    eq_(planes[45, pure_moves['a']], 1)
    # search only neighbor string of 'b'
    eq_(planes[45, pure_moves['b']], 0)
    # search all escape routes
    #eq_(planes[45, pure_moves['b']], 1)

    board.free_game(node.game)
    policy_feature.free_feature(feature)


def test_escapes_1():
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    moves, pure_moves = parseboard.parse(node.game,
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
    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    node.game.current_color = board.S_WHITE

    policy_feature.update(feature, node)
    eq_(planes[45, pure_moves['a']], 1)
    # search only neighbor string of 'b'
    eq_(planes[45, pure_moves['b']], 0)
    # search all escape routes
    #eq_(planes[45, pure_moves['b']], 1)

    board.free_game(node.game)
    policy_feature.free_feature(feature)


def test_escapes_require_many_moves():
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    moves, pure_moves = parseboard.parse(node.game,
                            ". . . . . . . . . . . . . . . . . . .|"
                            ". . . . . . . . . . . . . . . . . . .|"
                            ". . . W . . . . . . . B . . . . . . .|"
                            ". . . . W . . B . . . . . . . . B . .|"
                            ". . B . . . . . . . . . . . . . . . .|"
                            ". . . . . . . . . . . . . . . . . . .|"
                            ". . . . . . . . . . . . . . . . . . .|"
                            ". . . . . . . . . . . . . . . . . . .|"
                            ". . . . . . . . . . . . . . . . . . .|"
                            ". . W . . . . . . . . . . . . W . . .|"
                            ". . . . . . . . . . . . . . . . . . .|"
                            ". . . . . . . . . . . . . . . . W . .|"
                            ". . . . . . . . . . . . . . . . a . .|"
                            ". . . . . . . . . . . . . . . W B W .|"
                            ". . W . . . . . . . . . . . B B W B .|"
                            ". . . . B . . . . . . . . . W W W B .|"
                            ". . . B . . . . . . . . . W B . B . .|"
                            ". . . . . . . . . . . . . . . . . . .|"
                            ". . . . . . . . . . . . . . . . . . .|")
    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    node.game.current_color = board.S_BLACK

    policy_feature.update(feature, node)
    eq_(planes[45, pure_moves['a']], 1)

    board.free_game(node.game)


def test_captured_require_many_moves():
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    moves, pure_moves = parseboard.parse(node.game,
                            ". . . . . . . . . . . . . . . . . . .|"
                            ". . . . . . . . . . . . . . . . . . .|"
                            ". . . . . B . . . . . . . . . . . . .|"
                            ". . . B . . . . . . . . . . . B . . .|"
                            ". . . . . . . . . . . . . . . . . . .|"
                            ". . W . . . . . . . . . . . . . . . .|"
                            ". . . . . . . . . . . . . . . . . . .|"
                            ". . . . . . . . . . . . . . . . . . .|"
                            ". . . . . . . . . . . . . . . . . . .|"
                            ". . W . . . . . . . . . . . . . . . .|"
                            ". . . . . . . . . . . . . . . . . . .|"
                            ". . . . . . . . . . . . . . . . . . .|"
                            ". . . . . . . . . . . . . . . . . a W|"
                            ". . . . . . . . . . . . . . W . W W B|"
                            ". . B . . . . . . . . . . B B . B B B|"
                            ". . . . . . . . . . . W W W B B B W .|"
                            ". . . B . . . . . . B B . B W W W W .|"
                            ". . . . . W . . . . . . . B W . . . .|"
                            ". . . . . . . . . . . . . . B W . . .|")
    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    node.game.current_color = board.S_BLACK

    policy_feature.update(feature, node)
    eq_(planes[44, pure_moves['a']], 1)

    board.free_game(node.game)
    policy_feature.free_feature(feature)


def test_captured_2():
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    moves, pure_moves = parseboard.parse(node.game,
                            ". . b . . . . . . . . . . . . . . . .|"
                            ". W B a . . . . . . . . . . . . . . .|"
                            ". W B W . . . . . . . . . . . . . . .|"
                            ". . W B B . . W . . . . . . . B . . .|"
                            ". . W B W B . . . . . . . . . . . . .|"
                            ". W B W W . . . . . . . . . . . . . .|"
                            ". B B B W . . . . . . . . . . . . . .|"
                            ". . . . B . . . . . . . . . . . . . .|"
                            ". . . . . . . . . . . . . . . . . . .|"
                            ". . . . . . . . . . . . . . . . . . .|"
                            ". . . . . . . . . . . . . . . . . . .|"
                            ". . . . . . . . . . . . . . . . . . .|"
                            ". . . . . . . . . . . . . . . . . . .|"
                            ". . . . . . . . . . . . . . . . . . .|"
                            ". . . . . . . . . . . . . . . . . . .|"
                            ". . . B . . . . . . . . . . . B . . .|"
                            ". . . . . . . . . . . . . . . . . . .|"
                            ". . . . . . . . . . . . . . . . . . .|"
                            ". . . . . . . . . . . . . . . . . . .|")
    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    node.game.current_color = board.S_WHITE

    policy_feature.update(feature, node)
    eq_(planes[44, pure_moves['a']], 1)
    eq_(planes[44, pure_moves['b']], 0)

    board.free_game(node.game)
    policy_feature.free_feature(feature)


def test_escape_segmentation_fault_1():
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    for i in range(361):
        node.children[i] = <tree_node_t *>malloc(sizeof(tree_node_t))
    (moves, pure_moves) = parseboard.parse(node.game,
        ". . . W W B . . . . B . W . W W . W .|"
        "W W W W B B . W W B W W W . W B W B B|"
        "W . W W W B B W B B B B B W B B B W .|"
        "B W . W W W B W W B . W B B B B . B a|"
        ". B W B B B B . W B W B W W W W W W .|"
        ". B W B . . B . W W B B . B W . B B W|"
        ". W W B . W W W . W B . B B B B B W .|"
        "W . W B B W . B . W B W W W W . . W .|"
        "W W B B . W W . B B W B B B . W W . .|"
        "B B . B B W . W . B W W W B B B W . .|"
        ". . W . W B W . B W W W W W W W W W W|"
        ". B B B B B W W W B W B B . W B B W B|"
        ". B W B . B B B B B B B . B B . B W B|"
        ". . W B B W B . B W W W B . B . B B B|"
        "W W W W W W W . W . W . W B B . W W .|"
        "W B W B W W W W W W W W W B W W W B W|"
        "B B W B W B B B B W B W B B B W B B .|"
        ". B B B B . . B B B B W B W W B B B .|"
        ". . . . . B B W . W W . W W . W B . .|")

    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    board.put_stone(node.game, moves['a'], board.S_WHITE)

    node.game.current_color = board.S_BLACK
    
    policy_feature.update(feature, node)


def test_capture_segmentation_fault_1():
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    for i in range(361):
        node.children[i] = <tree_node_t *>malloc(sizeof(tree_node_t))
    (moves, pure_moves) = parseboard.parse(node.game,
        ". . . W W B . . . . B . W . W W . W .|"
        "W W W W B B . W W B W W W . W B W B B|"
        "W . W W W B B W B B B B B W B B B W .|"
        "B W . W W W B W W B . W B B B B a B W|"
        ". B W B B B B . W B W B W W W W W W .|"
        ". B W B . . B . W W B B . B W . B B W|"
        ". W W B . W W W . W B . B B B B B W .|"
        "W . W B B W . B . W B W W W W . . W .|"
        "W W B B . W W . B B W B B B . W W . .|"
        "B B . B B W . W . B W W W B B B W . .|"
        ". . W . W B W . B W W W W W W W W W W|"
        ". B B B B B W W W B W B B . W B B W B|"
        ". B W B . B B B B B B B . B B . B W B|"
        ". . W B B W B . B W W W B . B . B B B|"
        "W W W W W W W . W . W . W B B . W W .|"
        "W B W B W W W W W W W W W B W W W B W|"
        "B B W B W B B B B W B W B B B W B B .|"
        ". B B B B . . B B B B W B W W B B B .|"
        ". . . . . B B W . W W . W W . W B . .|")

    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    board.put_stone(node.game, moves['a'], board.S_BLACK)

    node.game.current_color = board.S_WHITE
    
    policy_feature.update(feature, node)


def test_segmentation_fault_2():
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    for i in range(361):
        node.children[i] = <tree_node_t *>malloc(sizeof(tree_node_t))
    (moves, pure_moves) = parseboard.parse(node.game,
        ". . . . . . B B B W . W . W W B . . .|"
        "B . B . B B B W B B W W W W B B B B B|"
        "B B W B B W W W W W W B W B W B B . B|"
        "B W W B W B . . . . W B B B W W W B B|"
        "W W W W W . W . W W W B B . B B W B W|"
        ". B . W B . . W W B B B . B B W W W W|"
        ". . W B . B B W . W W W B B W W W B .|"
        ". . W . W W W . . W W W B B B W W . .|"
        ". . . W B B B W . . W B B W W W W W W|"
        "W W W B . B W W . W W W W B W W B B W|"
        "W B B B . B W . W . . W B B B B B B W|"
        "B B . . B W W W B . W W W B . W W B B|"
        ". . . . B B W W B B B B W B W a . W B|"
        ". . . . . W B B W W . W B . B W B W W|"
        ". . . . . . B W B W W W W B W . W W B|"
        ". . B . . . . . B B B B W W W W W B B|"
        ". . . . B . . . . . . . B B W B B B .|"
        ". . . . . . . . . . . . . B B W B W .|"
        ". . . . . . . . . . . . . . . . . . .|")

    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    board.put_stone(node.game, moves['a'], board.S_BLACK)

    node.game.current_color = board.S_WHITE
    
    policy_feature.update(feature, node)


def test_captured_3():
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    for i in range(361):
        node.children[i] = <tree_node_t *>malloc(sizeof(tree_node_t))
    (moves, pure_moves) = parseboard.parse(node.game,
        ". . . . . . . . . . . . . . . . . . .|"
        ". . . . . . . . . . . . . W B B B W .|"
        ". . . . . . . . . . . . W B W B W W .|"
        ". . . . . . . . . . . W B B W W B B .|"
        ". . . . . . . . . . W B B W . . . . .|"
        ". . . . . . . . . W B B W . . . . . .|"
        ". . . . . . . . W B B W . . . . . . .|"
        ". . . . . . . W B B W . . . . . . . .|"
        ". . . . . . W B B W . . . . . . . . .|"
        ". . . . . W B B W . . . . . . . . . .|"
        ". . . . W B B W . . . . . . . . . . .|"
        ". . . W B B W . . . . . . . . . . . .|"
        ". . W B B W . . . . . . . . . . . . .|"
        ". . W B W . . . . . . . . . . . . . .|"
        ". . b B a . . . . . . . . . . . . . .|"
        ". . . W . . . . . . . . . . . . . . .|"
        ". . . . . . . . . . . . . . . . . . .|"
        ". . . . . . . . . . . . . . . . . . .|"
        ". . . . . . . . . . . . . . . . . . .|")

    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    node.game.current_color = board.S_WHITE

    policy_feature.update(feature, node)
    eq_(planes[44, pure_moves['a']], 1)
    #for i in range(46):
    #    print i, planes[i, pure_moves['a']], planes[i, pure_moves['b']]
    eq_(planes[44, pure_moves['b']], 0)
    
    board.free_game(node.game)
    policy_feature.free_feature(feature)
