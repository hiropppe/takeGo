import numpy as np

cimport numpy as np

from libc.stdlib cimport malloc, free

from . cimport board 
from . cimport policy_feature
from . cimport parseboard
from . cimport printer

from .board cimport is_legal
from .policy_feature cimport MAX_POLICY_PLANES
from .tree_search cimport tree_node_t


def test_stone_color():
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(node.game,
                             "a . .|"
                             ". b .|"
                             ". . c|")
    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    board.put_stone(node.game, moves['a'], board.S_BLACK)
    board.put_stone(node.game, moves['b'], board.S_WHITE)

    node.game.current_color = board.S_BLACK
    policy_feature.update(feature, node)

    assert (planes[0, pure_moves['a']] == 1)
    assert (planes[1, pure_moves['b']] == 1)
    assert (planes[2, pure_moves['c']] == 1)
    assert (planes[0].sum() == 1)
    assert (planes[1].sum() == 1)
    assert (planes[2].sum(), 7)

    node.game.current_color = board.S_WHITE
    policy_feature.update(feature, node)

    assert (planes[0, pure_moves['b']] == 1)
    assert (planes[1, pure_moves['a']] == 1)
    assert (planes[2, pure_moves['c']] == 1)
    assert (planes[0].sum() == 1)
    assert (planes[1].sum() == 1)
    assert (planes[2].sum(), 7)


def test_turns_since():
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(node.game,
                             ". . . . .|"
                             ". a b c .|"
                             ". d e f .|"
                             ". g . . .|"
                             ". . . . .|")
    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    node.game.current_color = board.S_BLACK

    # move 'a'
    board.do_move(node.game, moves['a'])
    policy_feature.update(feature, node)
    assert (planes[4, pure_moves['a']] == 1) # 0 age
    assert (planes[4].sum() == 1)

    # move 'b'
    board.do_move(node.game, moves['b'])
    policy_feature.update(feature, node)
    assert (planes[5, pure_moves['a']] == 1) # 1 age
    assert (planes[4, pure_moves['b']] == 1) # 0 age
    assert (planes[5].sum() == 1)
    assert (planes[4].sum() == 1)

    # PASS
    board.do_move(node.game, board.PASS)
    board.do_move(node.game, board.PASS)
    policy_feature.update(feature, node)
    assert (planes[7, pure_moves['a']] == 1) # 3 age
    assert (planes[6, pure_moves['b']] == 1) # 2 age
    assert (planes[7].sum() == 1)
    assert (planes[6].sum() == 1)

    board.do_move(node.game, moves['c'])
    board.do_move(node.game, moves['d'])
    board.do_move(node.game, moves['e'])
    board.do_move(node.game, moves['f'])
    policy_feature.update(feature, node)
    assert (planes[11, pure_moves['a']] == 1) # 7 age
    assert (planes[10, pure_moves['b']] == 1) # 6 age
    assert (planes[7, pure_moves['c']] == 1)  # 3 age
    assert (planes[6, pure_moves['d']] == 1)  # 2 age
    assert (planes[4, pure_moves['f']] == 1)  # 0 age

    board.do_move(node.game, moves['g'])
    policy_feature.update(feature, node)
    assert (planes[11, pure_moves['a']] == 1) # 7 age
    assert (planes[11, pure_moves['b']] == 1) # 7 age
    assert (planes[8, pure_moves['c']] == 1)  # 4 age
    assert (planes[7, pure_moves['d']] == 1)  # 3 age
    assert (planes[5, pure_moves['f']] == 1)  # 1 age
    assert (planes[4, pure_moves['g']] == 1)  # 0 age


def test_liberties():
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(node.game,
                             ". . . . .|"
                             ". . a . .|"
                             ". . b . .|"
                             ". . c . .|"
                             ". . d . .|")
    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    board.put_stone(node.game, moves['a'], board.S_BLACK)
    policy_feature.update(feature, node)
    assert (planes[12 + 4 - 1, pure_moves['a']] == 1)

    board.put_stone(node.game, moves['b'], board.S_BLACK)
    board.put_stone(node.game, moves['c'], board.S_BLACK)
    policy_feature.update(feature, node)
    assert (planes[12 + 8 - 1, pure_moves['a']] == 1)
    assert (planes[12 + 8 - 1, pure_moves['b']] == 1)
    assert (planes[12 + 8 - 1, pure_moves['c']] == 1)

    board.put_stone(node.game, moves['d'], board.S_BLACK)
    policy_feature.update(feature, node)
    assert (planes[12 + 8 - 1, pure_moves['a']] == 1)
    assert (planes[12 + 8 - 1, pure_moves['b']] == 1)
    assert (planes[12 + 8 - 1, pure_moves['c']] == 1)
    assert (planes[12 + 8 - 1, pure_moves['d']] == 1)


def test_capture_size():
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    for i in range(361):
        node.children[i] = <tree_node_t *>malloc(sizeof(tree_node_t))
    (moves, pure_moves) = parseboard.parse(node.game,
                             "O a . . O X X X X|"
                             "X . . . . O X X X|"
                             ". . . . . . O X X|"
                             ". . . . . . . O c|"
                             ". . . . . . . . .|"
                             ". . . . . . . . .|"
                             ". . . . . . . . .|"
                             "X X X X X X X X .|"
                             "O O O O O O O O b|")
    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    node.game.current_color = board.S_BLACK
    policy_feature.update(feature, node)
    assert (planes[20 + 1, pure_moves['a']] == 1)
    assert (planes[20 + 1].sum() == 1)
    assert (planes[20 + 7, pure_moves['b']] == 1)
    assert (planes[20 + 7].sum() == 1)

    node.game.current_color = board.S_WHITE
    policy_feature.update(feature, node)

    assert (planes[20 + 7, pure_moves['c']] == 1)
    assert (planes[20 + 7].sum() == 1)


def test_self_atari_size():
    pass
    """
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(node.game,
                             "O a . . O X X X X|"
                             "X . . . . O X X X|"
                             ". . . . . . O X X|"
                             ". . . . . . . O c|"
                             ". . . . . . . . .|"
                             ". . . . . . . . .|"
                             ". . . . . . . . .|"
                             "X X X X X X X X .|"
                             "O O O O O O O O b|")
    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    node.game.current_color = board.S_WHITE
    policy_feature.update(feature, node)

    assert (planes[28 + 1 - 1, pure_moves['a']] == 1)
    assert (planes[28 + 1 - 1].sum() == 1)
    assert (planes[28 + 8 - 1, pure_moves['b']] == 1)
    assert (planes[28 + 8 - 1].sum() == 1)

    node.game.current_color = board.S_BLACK
    policy_feature.update(feature, node)

    assert (planes[28 + 8 - 1, pure_moves['c']] == 1)
    assert (planes[28 + 8 - 1].sum() == 1)

    board.free_game(node.node)
    policy_feature.free_feature(feature)
    """

def test_liberties_after_move():
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(node.game,
                             ". . . . .|"
                             ". . a . .|"
                             ". . b . .|"
                             ". . c . .|"
                             ". . d e .|")
    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    node.game.current_color = board.S_BLACK

    # after B 'a'
    policy_feature.update(feature, node)
    assert (planes[36 + 4 - 1, pure_moves['a']] == 1)

    # after B 'b'
    board.put_stone(node.game, moves['a'], board.S_BLACK)
    policy_feature.update(feature, node)
    assert (planes[36 + 6 - 1, pure_moves['b']] == 1)

    # after B 'c'
    board.put_stone(node.game, moves['b'], board.S_BLACK)
    policy_feature.update(feature, node)
    assert (planes[36 + 8 - 1, pure_moves['c']] == 1)

    # after B 'd'
    board.put_stone(node.game, moves['c'], board.S_BLACK)
    policy_feature.update(feature, node)
    assert (planes[36 + 8 - 1, pure_moves['d']] == 1)

    node.game.current_color = board.S_WHITE

    # after W 'e'
    board.put_stone(node.game, moves['d'], board.S_BLACK)
    policy_feature.update(feature, node)
    assert (planes[36 + 2 - 1, pure_moves['e']] == 1)


def test_liberties_after_move_1():
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(node.game,
                             ". . . . .|"
                             ". W B . .|"
                             ". W a B .|"
                             ". . . . .|"
                             ". . . . .|")
    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    node.game.current_color = board.S_BLACK

    policy_feature.update(feature, node)
    assert (planes[36 + 5 - 1, pure_moves['a']] == 1)


def test_liberties_after_move_dupe_empty():
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(node.game,
                             ". . . . . .|"
                             ". . . W . .|"
                             ". . a W . .|"
                             ". B B W . .|"
                             ". . . B W .|"
                             ". . . . . .|")
    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    node.game.current_color = board.S_BLACK

    policy_feature.update(feature, node)
    assert (planes[36 + 5 - 1, pure_moves['a']] == 1)


def test_liberties_after_move_captured():
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(node.game,
                             ". . . . . .|"
                             ". . B B . .|"
                             ". B W W B .|"
                             ". . a B W .|"
                             ". . B W W .|"
                             ". . . . . .|")
    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    node.game.current_color = board.S_BLACK

    policy_feature.update(feature, node)
    assert (planes[36 + 5 - 1, pure_moves['a']] == 1)


def test_liberties_after_move_captured_1():
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    for i in range(361):
        node.children[i] = <tree_node_t *>malloc(sizeof(tree_node_t))
    (moves, pure_moves) = parseboard.parse(node.game,
                             "W W B B B B W|"
                             "W B B W . B W|"
                             "B W W W W W W|"
                             "B B W B B B a|"
                             "B W B B W W .|"
                             "B W W B W . .|"
                             ". B . W . . .|")
    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    node.game.current_color = board.S_WHITE

    policy_feature.update(feature, node)
    assert (planes[36 + 6 - 1, pure_moves['a']] == 1)


def test_liberties_after_move_captured_2():
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    for i in range(361):
        node.children[i] = <tree_node_t *>malloc(sizeof(tree_node_t))
    (moves, pure_moves) = parseboard.parse(node.game,
                             ". . B . B B W . .|"
                             "B B B B W W B B .|"
                             "B W W W W W B B B|"
                             "W W B B W B B W W|"
                             "a W W B W W W B .|"
                             "B W W B B B . B .|"
                             "B W W W W B B . .|"
                             "B B W W B B . . .|"
                             "B W W B B . . . .|")
    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    node.game.current_color = board.S_WHITE

    policy_feature.update(feature, node)
    assert (planes[36 + 5 - 1, pure_moves['a']] == 1)


def test_sensibleness_not_suicide():
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    for i in range(361):
        node.children[i] = <tree_node_t *>malloc(sizeof(tree_node_t))
    (moves, pure_moves) = parseboard.parse(node.game,
                            "W W W . . . .|"
                            "W W B W . . .|"
                            "W B B B W . .|"
                            "B a B W . . .|"
                            "B B B W . . .|"
                            "W B W . . . .|"
                            ". W . . . . .|")
    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    node.game.current_color = board.S_WHITE
    
    policy_feature.update(feature, node)
    assert is_legal(node.game, moves['a'], board.S_WHITE)
    assert (planes[46, pure_moves['a']] == 1)


def test_sensibleness_true_eye():
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(node.game,
                            "B a B . B W W|"
                            "B B B W W d W|"
                            "B b B W e W W|"
                            "B B . B W f W|"
                            "B B B W g W W|"
                            "B c B W W W W|"
                            "B B B W h W i|")
    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    node.game.current_color = board.S_BLACK
    
    policy_feature.update(feature, node)
    assert (planes[46, pure_moves['a']] == 0)
    assert (planes[46, pure_moves['b']] == 0)
    assert (planes[46, pure_moves['c']] == 0)
    
    node.game.current_color = board.S_WHITE
    policy_feature.update(feature, node)
    assert (planes[46, pure_moves['d']] == 0)
    assert (planes[46, pure_moves['e']] == 0)
    assert (planes[46, pure_moves['f']] == 0)
    assert (planes[46, pure_moves['g']] == 0)
    assert (planes[46, pure_moves['h']] == 0)
    assert (planes[46, pure_moves['i']] == 0)


def test_sensibleness_not_true_eye():
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(node.game,
                            "B B B . . W c|"
                            "B a B . . . W|"
                            "W B . . . . .|"
                            "W b W . . . .|"
                            "W W W . . . .|"
                            ". . . . . W W|"
                            ". . . . W d W|")
    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    node.game.current_color = board.S_BLACK
    
    policy_feature.update(feature, node)
    assert (planes[46, pure_moves['a']] == 1)
    
    node.game.current_color = board.S_WHITE

    policy_feature.update(feature, node)
    assert (planes[46, pure_moves['b']] == 1)
    assert (planes[46, pure_moves['c']] == 1)
    assert (planes[46, pure_moves['d']] == 1)


def test_sensibleness_true_eye_remove_stone():
    cdef tree_node_t *node
    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    node.game = board.allocate_game()
    (moves, pure_moves) = parseboard.parse(node.game,
                             "B W W W B B .|"
                             "B B . W B . .|"
                             "W B W W B . .|"
                             "B W W B W B .|"
                             "B W B B W W W|"
                             "B B a B B B B|"
                             "W B b c d . .|")
    feature = policy_feature.allocate_feature(MAX_POLICY_PLANES)
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    board.put_stone(node.game, moves['a'], board.S_WHITE)
    board.put_stone(node.game, moves['b'], board.S_BLACK)
    board.put_stone(node.game, moves['c'], board.S_WHITE)
    board.put_stone(node.game, moves['d'], board.S_BLACK)

    node.game.current_color = board.S_BLACK
    
    policy_feature.update(feature, node)
    assert (planes[46, pure_moves['a']] == 0)
    assert (planes[46, pure_moves['c']] == 0)

