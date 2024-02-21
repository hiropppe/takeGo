from __future__ import division

import numpy as np
cimport numpy as np

from bamboo.models.keras_dcnn_policy import CNNPolicy

from libc.stdio cimport printf

from bamboo.board cimport PURE_BOARD_SIZE, BOARD_SIZE, OB_SIZE, S_EMPTY, S_BLACK, S_WHITE, PASS, RESIGN
from bamboo.board cimport FLIP_COLOR, CORRECT_X, CORRECT_Y
from bamboo.board cimport game_state_t, onboard_pos
from bamboo.board cimport set_board_size, initialize_board, allocate_game, free_game, put_stone, copy_game, calculate_score, komi
from bamboo.printer cimport print_board
from bamboo.parseboard cimport parse

from bamboo.zobrist_hash cimport uct_hash_size
from bamboo.zobrist_hash cimport set_hash_size, initialize_hash, initialize_uct_hash, clear_uct_hash, delete_old_hash, search_empty_index, find_same_hash_index
from bamboo.tree_search cimport tree_node_t, MCTS

from bamboo.rollout_preprocess cimport set_debug, initialize_rollout_const, initialize_rollout, update_rollout, set_rollout_parameter
from bamboo.local_pattern cimport read_rands, init_d12_rsp_hash, init_x33_hash
from bamboo.nakade cimport initialize_nakade_hash

from bamboo.gtp import gtp

pn_path = None

def setup_pattern(rands_file, d12_rsp_csv, x33_csv):
    cdef int x33_size, d12_size, nakade_size

    set_hash_size(2**20)
    initialize_hash()

    read_rands(rands_file)
    nakade_size = initialize_nakade_hash()
    x33_size = init_x33_hash(x33_csv)
    d12_rsp_size = init_d12_rsp_hash(d12_rsp_csv)

    initialize_rollout_const(nakade_size, x33_size, d12_rsp_size, 0)


def setup_supervised_policy(path, nogpu=True):
    global pn_path

    pn_path = path


def setup_rollout_policy(weights):
    set_rollout_parameter(weights)

def setup():
    initialize_uct_hash()


def teardown():
    pass


def test_seek_root():
    cdef game_state_t *game = initialize_game()
    cdef MCTS mcts = MCTS()
    cdef unsigned int tmp_root
    cdef tree_node_t *node

    game.current_color = S_BLACK

    # put B[Q16]
    put_stone(game, 132, game.current_color)
    game.current_color = FLIP_COLOR(game.current_color) 

    # corresponding node for game not found. assigned empty index
    assert (mcts.seek_root(game) is False)
    tmp_root = mcts.current_root

    # game state is not updated. seek to existing node index
    assert (mcts.seek_root(game) is True)
    assert (mcts.current_root == tmp_root)

    # put W[D4]
    put_stone(game, 396, game.current_color)
    game.current_color = FLIP_COLOR(game.current_color) 

    assert (mcts.seek_root(game) is False)
    tmp_root = mcts.current_root

    # game state is not updated. seek to existing node index
    assert (mcts.seek_root(game) is True)
    assert (mcts.current_root == tmp_root)

    #node = &mcts.nodes[mcts.current_root]
    #print_board(node.game)

    free_game(game)


def test_expand():
    cdef game_state_t *game = initialize_game()
    cdef MCTS mcts = MCTS()
    cdef tree_node_t *node
    cdef tree_node_t *child

    game.current_color = S_BLACK

    assert (mcts.seek_root(game) is False)
    node = &mcts.nodes[mcts.current_root]

    # expand empty board
    mcts.expand(node, game)

    assert (node.num_child, 361)
    assert (node.is_edge is False)

    # assert child B[Q16]
    child = node.children[72]
    assert (node.children_pos[72] == 72)
    assert (child.time_step == 1)
    assert (child.pos == 132)
    assert (child.color == S_BLACK)
    assert (child.P == 0)
    assert (child.Nv == 0)
    assert (child.Nr == 0)
    assert (child.Wv == 0)
    assert (child.Wr == 0)
    assert (child.Q == 0)
    assert (child.num_child == 0)
    assert (child.is_root is False)
    assert (child.is_edge is True)

    # put B[Q16]
    put_stone(game, 132, game.current_color)
    game.current_color = FLIP_COLOR(game.current_color) 
    update_rollout(game)

    assert (mcts.seek_root(game) is True)

    # put W[D4]
    put_stone(game, 396, game.current_color)
    game.current_color = FLIP_COLOR(game.current_color) 
    update_rollout(game)

    assert (mcts.seek_root(game) is False)
    node = &mcts.nodes[mcts.current_root]

    # expand B[Q16]->W{D4}
    mcts.expand(node, game)

    assert (node.num_child, 359)
    assert (node.is_edge is False)

    # assert child B[Q4]
    child = node.children[300]
    assert (node.children_pos[298] == 300)
    assert (child.time_step == 3)
    assert (child.pos == 408)
    assert (child.color == S_BLACK)

    # put B[D4]
    put_stone(game, 408, game.current_color)
    game.current_color = FLIP_COLOR(game.current_color) 
    update_rollout(game)

    # seek root matches expanded child node
    assert (mcts.seek_root(game) is True)

    #node = &mcts.nodes[mcts.current_root]
    #print_board(node.game)


def test_select():
    cdef game_state_t *game = initialize_game()
    cdef game_state_t *search_game = allocate_game()
    cdef MCTS mcts = MCTS()
    cdef tree_node_t *root_node
    cdef tree_node_t *node
    cdef int max_child_pos
    cdef double max_child_Qu = .0
    cdef int i

    mcts.use_rollout = False

    game.current_color = S_BLACK

    mcts.seek_root(game)
    root_node = &mcts.nodes[mcts.current_root]
    node = root_node

    copy_game(search_game, game)

    mcts.expand(node, search_game)
    # set max_child
    node.children[72].P = 0.99

    # select down the tree
    node = mcts.select(node, search_game)
    assert (node.pos == 132) # B[Q16]
    assert (node.color == S_BLACK)
    assert (node.is_edge is True)

    mcts.expand(node, search_game)
    # set max_child
    max_child_pos = 300
    node.Nr = 1000
    node.children[max_child_pos].Nr = 1000
    node.children[max_child_pos].Q = 0.5
    node.P = 0.99

    node = mcts.select(node, search_game)
    assert (node.pos == onboard_pos[max_child_pos])
    assert (node.color == S_WHITE)
    assert (node.is_edge is True)

    # one more from the beginning
    node = root_node
    copy_game(search_game, game)
    node = mcts.select(node, search_game)
    assert (node.pos == 132) # B[Q16]
    assert (node.color == S_BLACK)
    assert (node.is_edge is False)
    node = mcts.select(node, search_game)
    assert (node.pos == onboard_pos[max_child_pos])
    assert (node.color == S_WHITE)
    assert (node.is_edge is True)

    # put B[Q16]
    put_stone(game, 132, game.current_color)
    game.current_color = FLIP_COLOR(game.current_color) 
    update_rollout(game)

    mcts.seek_root(game)
    root_node = &mcts.nodes[mcts.current_root]
    node = root_node
    node = mcts.select(node, search_game)
    assert (node.pos == onboard_pos[max_child_pos])
    assert (node.color == S_WHITE)
    assert (node.is_edge is True)


def test_rollout():
    cdef game_state_t *game = initialize_game()
    cdef MCTS mcts = MCTS()

    game.current_color = S_BLACK

    put_stone(game, 132, game.current_color)
    game.current_color = FLIP_COLOR(game.current_color) 
    update_rollout(game)

    mcts.rollout(game)

    print_board(game)


def test_eval_leaf_by_policy_network():
    cdef game_state_t *game = initialize_game()
    cdef MCTS mcts
    cdef tree_node_t *node0
    cdef tree_node_t *node1
    cdef tree_node_t *node2
    cdef tree_node_t *eval_node
    cdef tree_node_t *child
    cdef int i
    cdef double prob_sum = 0.0

    mcts = MCTS(nogpu=True)
    mcts.run_pn_session(pn_path)

    game.current_color = S_BLACK

    # seek and expand
    assert (mcts.seek_root(game) is False)
    node0 = &mcts.nodes[mcts.current_root]
    mcts.expand(node0, game)

    assert (mcts.policy_network_queue.size() == 1)

    # eval node0
    eval_node = mcts.policy_network_queue.front()
    assert (node0.node_i, eval_node.node_i)

    mcts.eval_leaf_by_policy_network(eval_node)
    mcts.policy_network_queue.pop()

    prob_sum = .0
    for i in range(node0.num_child):
        prob_sum += node0.children[node0.children_pos[i]].P
    assert (round(prob_sum) == 1.0)

    # put B[Q16]
    put_stone(game, 132, game.current_color)
    game.current_color = FLIP_COLOR(game.current_color) 
    update_rollout(game)

    # seek and expand
    assert (mcts.seek_root(game) is True)
    node1 = &mcts.nodes[mcts.current_root]
    mcts.expand(node1, game)

    assert (mcts.policy_network_queue.size() == 1)

    # put W[D4]
    put_stone(game, 396, game.current_color)
    game.current_color = FLIP_COLOR(game.current_color) 
    update_rollout(game)

    # seek and expand
    assert (mcts.seek_root(game) is True)
    node2 = &mcts.nodes[mcts.current_root]
    mcts.expand(node2, game)

    assert (mcts.policy_network_queue.size() == 2)

    # eval node1
    eval_node = mcts.policy_network_queue.front()
    assert (node1.node_i == eval_node.node_i)

    mcts.eval_leaf_by_policy_network(eval_node)
    mcts.policy_network_queue.pop()

    prob_sum = .0
    for i in range(node1.num_child):
        prob_sum += node1.children[node1.children_pos[i]].P
    assert (round(prob_sum) == 1.0)

    # eval node2
    eval_node = mcts.policy_network_queue.front()
    assert (node2.node_i == eval_node.node_i)

    mcts.eval_leaf_by_policy_network(eval_node)
    mcts.policy_network_queue.pop()

    prob_sum = .0
    for i in range(node2.num_child):
        prob_sum += node2.children[node2.children_pos[i]].P
    assert (round(prob_sum) == 1.0)

    assert mcts.policy_network_queue.empty()


cdef game_state_t* initialize_game(int board_size=19):
    cdef game_state_t *game

    set_board_size(board_size)

    game = allocate_game()
    initialize_board(game)

    return game
