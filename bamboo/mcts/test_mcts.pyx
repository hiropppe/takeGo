from __future__ import division

import numpy as np
cimport numpy as np

from bamboo.models import policy

from libc.stdio cimport printf

from nose.tools import ok_, eq_

from bamboo.go.board cimport PURE_BOARD_SIZE, BOARD_SIZE, OB_SIZE, S_EMPTY, S_BLACK, S_WHITE, PASS, RESIGN
from bamboo.go.board cimport FLIP_COLOR, CORRECT_X, CORRECT_Y
from bamboo.go.board cimport game_state_t, onboard_pos
from bamboo.go.board cimport set_board_size, initialize_board, allocate_game, free_game, put_stone, copy_game, calculate_score, komi
from bamboo.go.printer cimport print_board
from bamboo.go.parseboard cimport parse

from bamboo.go.zobrist_hash cimport uct_hash_size
from bamboo.go.zobrist_hash cimport set_hash_size, initialize_hash, initialize_uct_hash, clear_uct_hash, delete_old_hash, search_empty_index, find_same_hash_index
from bamboo.mcts.tree_search cimport tree_node_t, MCTS

from bamboo.rollout.preprocess cimport set_debug, initialize_const, initialize_rollout, update_rollout, set_rollout_parameter
from bamboo.rollout.pattern cimport read_rands, init_d12_hash, init_x33_hash

from bamboo.gtp import gtp

sl_policy = None

def setup_pattern(rands_file, d12_csv, x33_csv):
    cdef int x33_size, d12_size   

    set_hash_size(2**20)
    initialize_hash()

    read_rands(rands_file)
    x33_size = init_x33_hash(x33_csv)
    d12_size = init_d12_hash(d12_csv)

    initialize_const(0, x33_size, d12_size)


def setup_supervised_policy(model, weights):
    global sl_policy

    sl_policy = policy.CNNPolicy.load_model(model)
    sl_policy.model.load_weights(weights)


def setup_rollout_policy(weights):
    set_rollout_parameter(weights)


def setup():
    initialize_uct_hash()


def teardown():
    pass


def test_seek_root():
    cdef game_state_t *game = initialize_game()
    cdef MCTS mcts = MCTS(None)
    cdef unsigned int tmp_root
    cdef tree_node_t *node

    game.current_color = S_BLACK

    # put B[Q16]
    put_stone(game, 132, game.current_color)
    game.current_color = FLIP_COLOR(game.current_color) 

    # corresponding node for game not found. assigned empty index
    eq_(mcts.seek_root(game), False)
    tmp_root = mcts.current_root

    # game state is not updated. seek to existing node index
    eq_(mcts.seek_root(game), True)
    eq_(mcts.current_root, tmp_root)

    # put W[D4]
    put_stone(game, 396, game.current_color)
    game.current_color = FLIP_COLOR(game.current_color) 

    eq_(mcts.seek_root(game), False)
    tmp_root = mcts.current_root

    # game state is not updated. seek to existing node index
    eq_(mcts.seek_root(game), True)
    eq_(mcts.current_root, tmp_root)

    #node = &mcts.nodes[mcts.current_root]
    #print_board(node.game)

    free_game(game)


def test_expand():
    cdef game_state_t *game = initialize_game()
    cdef MCTS mcts = MCTS(None)
    cdef tree_node_t *node
    cdef tree_node_t *child

    game.current_color = S_BLACK

    eq_(mcts.seek_root(game), False)
    node = &mcts.nodes[mcts.current_root]

    # expand empty board
    mcts.expand(node, game)

    eq_(node.num_child, 361)
    eq_(node.is_edge, False)

    # assert child B[Q16]
    child = node.children[72]
    eq_(node.children_pos[72], 72)
    eq_(child.time_step, 1)
    eq_(child.pos, 132)
    eq_(child.color, S_BLACK)
    eq_(child.P, 0)
    eq_(child.Nv, 0)
    eq_(child.Nr, 0)
    eq_(child.Wv, 0)
    eq_(child.Wr, 0)
    eq_(child.Q, 0)
    eq_(child.num_child, 0)
    eq_(child.is_root, False)
    eq_(child.is_edge, True)

    # put B[Q16]
    put_stone(game, 132, game.current_color)
    game.current_color = FLIP_COLOR(game.current_color) 
    update_rollout(game)

    eq_(mcts.seek_root(game), True)

    # put W[D4]
    put_stone(game, 396, game.current_color)
    game.current_color = FLIP_COLOR(game.current_color) 
    update_rollout(game)

    eq_(mcts.seek_root(game), False)
    node = &mcts.nodes[mcts.current_root]

    # expand B[Q16]->W{D4}
    mcts.expand(node, game)

    eq_(node.num_child, 359)
    eq_(node.is_edge, False)

    # assert child B[Q4]
    child = node.children[300]
    eq_(node.children_pos[298], 300)
    eq_(child.time_step, 3)
    eq_(child.pos, 408)
    eq_(child.color, S_BLACK)

    # put B[D4]
    put_stone(game, 408, game.current_color)
    game.current_color = FLIP_COLOR(game.current_color) 
    update_rollout(game)

    # seek root matches expanded child node
    eq_(mcts.seek_root(game), True)

    #node = &mcts.nodes[mcts.current_root]
    #print_board(node.game)


def test_select():
    cdef game_state_t *game = initialize_game()
    cdef game_state_t *search_game = allocate_game()
    cdef MCTS mcts = MCTS(None)
    cdef tree_node_t *root_node
    cdef tree_node_t *node
    cdef int max_child_pos
    cdef double max_child_Qu = .0
    cdef int i

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
    eq_(node.pos, 132) # B[Q16]
    eq_(node.color, S_BLACK)
    eq_(node.is_edge, True)

    mcts.expand(node, search_game)
    # set max_child
    max_child_pos = 300
    node.Nr = 1000
    node.children[max_child_pos].Nr = 1000
    node.children[max_child_pos].Q = 0.5
    node.P = 0.99

    node = mcts.select(node, search_game)
    eq_(node.pos, onboard_pos[max_child_pos])
    eq_(node.color, S_WHITE)
    eq_(node.is_edge, True)

    # one more from the beginning
    node = root_node
    copy_game(search_game, game)
    node = mcts.select(node, search_game)
    eq_(node.pos, 132) # B[Q16]
    eq_(node.color, S_BLACK)
    eq_(node.is_edge, False)
    node = mcts.select(node, search_game)
    eq_(node.pos, onboard_pos[max_child_pos])
    eq_(node.color, S_WHITE)
    eq_(node.is_edge, True)

    # put B[Q16]
    put_stone(game, 132, game.current_color)
    game.current_color = FLIP_COLOR(game.current_color) 
    update_rollout(game)

    mcts.seek_root(game)
    root_node = &mcts.nodes[mcts.current_root]
    node = root_node
    node = mcts.select(node, search_game)
    eq_(node.pos, onboard_pos[max_child_pos])
    eq_(node.color, S_WHITE)
    eq_(node.is_edge, True)


def test_rollout():
    cdef game_state_t *game = initialize_game()
    cdef MCTS mcts = MCTS(None)

    game.current_color = S_BLACK

    put_stone(game, 132, game.current_color)
    game.current_color = FLIP_COLOR(game.current_color) 
    update_rollout(game)

    mcts.rollout(game)

    print_board(game)


def test_eval_leafs_by_policy_network():
    cdef game_state_t *game = initialize_game()
    cdef MCTS mcts
    cdef tree_node_t *node0
    cdef tree_node_t *node1
    cdef tree_node_t *node2
    cdef tree_node_t *eval_node
    cdef tree_node_t *child
    cdef int i
    cdef double prob_sum = 0.0

    mcts = MCTS(sl_policy)

    game.current_color = S_BLACK

    # seek and expand
    eq_(mcts.seek_root(game), False)
    node0 = &mcts.nodes[mcts.current_root]
    mcts.expand(node0, game)

    eq_(mcts.policy_network_queue.size(), 1)

    # eval node0
    eval_node = mcts.policy_network_queue.front()
    eq_(node0.node_i, eval_node.node_i)

    mcts.eval_leafs_by_policy_network(eval_node)
    mcts.policy_network_queue.pop()

    prob_sum = .0
    for i in range(node0.num_child):
        prob_sum += node0.children[node0.children_pos[i]].P
    eq_(round(prob_sum), 1.0)

    # put B[Q16]
    put_stone(game, 132, game.current_color)
    game.current_color = FLIP_COLOR(game.current_color) 
    update_rollout(game)

    # seek and expand
    eq_(mcts.seek_root(game), True)
    node1 = &mcts.nodes[mcts.current_root]
    mcts.expand(node1, game)

    eq_(mcts.policy_network_queue.size(), 1)

    # put W[D4]
    put_stone(game, 396, game.current_color)
    game.current_color = FLIP_COLOR(game.current_color) 
    update_rollout(game)

    # seek and expand
    eq_(mcts.seek_root(game), True)
    node2 = &mcts.nodes[mcts.current_root]
    mcts.expand(node2, game)

    eq_(mcts.policy_network_queue.size(), 2)

    # eval node1
    eval_node = mcts.policy_network_queue.front()
    eq_(node1.node_i, eval_node.node_i)

    mcts.eval_leafs_by_policy_network(eval_node)
    mcts.policy_network_queue.pop()

    prob_sum = .0
    for i in range(node1.num_child):
        prob_sum += node1.children[node1.children_pos[i]].P
    eq_(round(prob_sum), 1.0)

    # eval node2
    eval_node = mcts.policy_network_queue.front()
    eq_(node2.node_i, eval_node.node_i)

    mcts.eval_leafs_by_policy_network(eval_node)
    mcts.policy_network_queue.pop()

    prob_sum = .0
    for i in range(node2.num_child):
        prob_sum += node2.children[node2.children_pos[i]].P
    eq_(round(prob_sum), 1.0)

    ok_(mcts.policy_network_queue.empty())


def test_running():
    cdef game_state_t *game = initialize_game()
    cdef MCTS mcts = MCTS(sl_policy, playout_limit=5000)
    cdef tree_node_t *node
    cdef int pass_count = 0
    cdef int pos
    cdef int i

    initialize_rollout(game)

    game.current_color = S_BLACK

    while True:
        mcts.start_search_thread(game)

        while True:
            if mcts.policy_network_queue.empty():
                break
            node = mcts.policy_network_queue.front()
            mcts.eval_leafs_by_policy_network(node)
            free_game(node.game)
            mcts.policy_network_queue.pop()

        pos = mcts.genmove(game)

        put_stone(game, pos, game.current_color)
        game.current_color = FLIP_COLOR(game.current_color) 
        update_rollout(game)

        print_board(game)

        if pos == PASS or pos == RESIGN:
            print(gtp.gtp_vertex(pos))
        else:
            x = CORRECT_X(pos, BOARD_SIZE, OB_SIZE) + 1
            y = PURE_BOARD_SIZE-CORRECT_Y(pos, BOARD_SIZE, OB_SIZE)
            print(gtp.gtp_vertex((x, y)))

        if pos == PASS:
            pass_count += 1
            if pass_count == 2:
                break
        else:
            pass_count = 0

    print('Score: {:s}'.format(str(calculate_score(game) - komi)))


cdef game_state_t* initialize_game(int board_size=19):
    cdef game_state_t *game

    set_board_size(board_size)

    game = allocate_game()
    initialize_board(game)

    return game
