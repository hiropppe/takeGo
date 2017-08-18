# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

import numpy as np
import os
cimport numpy as np

from bamboo.models import policy

from libc.stdio cimport printf

from nose.tools import ok_, eq_

from bamboo.go.board cimport PURE_BOARD_SIZE, BOARD_SIZE, OB_SIZE, S_EMPTY, S_BLACK, S_WHITE, PASS, RESIGN
from bamboo.go.board cimport FLIP_COLOR, CORRECT_X, CORRECT_Y
from bamboo.go.board cimport game_state_t, onboard_pos
from bamboo.go.board cimport set_board_size, initialize_board, allocate_game, free_game, put_stone, copy_game, calculate_score, komi, is_legal_not_eye
from bamboo.go.printer cimport print_board
from bamboo.go.parseboard cimport parse

from bamboo.go.zobrist_hash cimport uct_hash_size
from bamboo.go.zobrist_hash cimport set_hash_size, initialize_hash, initialize_uct_hash, clear_uct_hash, delete_old_hash, search_empty_index, find_same_hash_index
from bamboo.mcts.tree_search cimport tree_node_t, PyMCTS

from bamboo.rollout.preprocess cimport set_debug, initialize_const, initialize_rollout, update_rollout, set_rollout_parameter
from bamboo.rollout.pattern cimport read_rands, init_d12_hash, init_x33_hash

from bamboo.gtp import gtp


def self_play(time_limit=60.0, playout_limit=10000, n_games=1, n_threads=1):
    cdef game_state_t *game
    cdef PyMCTS mcts
    cdef tree_node_t *node
    cdef int pass_count = 0
    cdef int pos
    cdef int i

    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    # config = tf.ConfigProto(device_count={"GPU": 0})
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    set_session(tf.Session(config=config))

    d = os.path.dirname(os.path.abspath(__file__))
    # supervised policy
    model = os.path.join(d, '../../params/policy/policy.json')
    weights = os.path.join(d, '../../params/policy/weights.00088.hdf5')
    # rollout policy
    rollout_weights = os.path.join(d, '../../params/rollout/sample.hdf5')
    # pattern hash for rollout
    rands_txt = os.path.join(d, '../../params/rollout/mt_rands.txt')
    d12_csv = os.path.join(d, '../../params/rollout/d12.csv')
    x33_csv = os.path.join(d, '../../params/rollout/x33.csv')

    set_hash_size(2**20)
    initialize_hash()

    read_rands(rands_txt)
    x33_size = init_x33_hash(x33_csv)
    d12_size = init_d12_hash(d12_csv)

    initialize_const(0, x33_size, d12_size)

    sl_policy = policy.CNNPolicy.load_model(model)
    sl_policy.model.load_weights(weights)

    set_rollout_parameter(rollout_weights)

    set_board_size(19)

    mcts = PyMCTS(sl_policy,
                  time_limit=time_limit,
                  playout_limit=playout_limit,
                  n_threads=n_threads,
                  read_ahead=False)
    mcts.clear()
    game = mcts.game
    for i in range(n_games):
        while True:
            pos = mcts.genmove(game.current_color)

            if pos == RESIGN:
                break

            mcts.play(pos, game.current_color)

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

        print('Score({:s}): {:s}'.format(str(i), str(calculate_score(game) - komi)))

        mcts.clear()
