# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

import numpy as np
import os

from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras import backend as K

from bamboo.models.keras_dcnn_policy import CNNPolicy
from bamboo.gtp import gtp

cimport numpy as np

from libc.stdio cimport printf

from bamboo.board cimport PURE_BOARD_SIZE, BOARD_SIZE, OB_SIZE, S_EMPTY, S_BLACK, S_WHITE, PASS, RESIGN
from bamboo.board cimport FLIP_COLOR, CORRECT_X, CORRECT_Y
from bamboo.board cimport game_state_t, onboard_pos
from bamboo.board cimport set_board_size, initialize_board, allocate_game, free_game, put_stone, copy_game, calculate_score, komi, \
        set_check_superko, set_japanese_rule, set_check_seki, set_use_lgrf2
from bamboo.zobrist_hash cimport uct_hash_size
from bamboo.zobrist_hash cimport set_hash_size, initialize_hash, initialize_uct_hash, clear_uct_hash, delete_old_hash, search_empty_index, find_same_hash_index
from bamboo.tree_search cimport tree_node_t, PyMCTS

from bamboo.rollout_preprocess cimport set_debug, initialize_rollout_const, initialize_rollout, update_rollout, set_rollout_parameter, set_tree_parameter
from bamboo.local_pattern cimport read_rands, init_d12_rsp_hash, init_x33_hash, init_d12_hash
from bamboo.nakade cimport initialize_nakade_hash
from bamboo.printer cimport print_board
from bamboo.parseboard cimport parse


def self_play(const_time=5.0,
              const_playout=0,
              n_games=1,
              n_threads=1,
              intuition=False,
              use_pn=True,
              use_vn=True,
              use_rollout=True,
              use_tree=True,
              superko=False,
              seki=False,
              japanese_rule=False,
              lgrf2=True):
    cdef game_state_t *game
    cdef PyMCTS mcts
    cdef tree_node_t *node
    cdef int pass_count = 0
    cdef int pos
    cdef int i

    if not (use_vn or use_rollout):
        print('No evaluation component enabled.')
        return

    # Parameters
    d = os.path.dirname(os.path.abspath(__file__))
    # Policy Net
    pn_path = os.path.join(d, '../params/policy/kihuu.hdf5')
    # Value Net
    vn_path = os.path.join(d, '../logs/model.ckpt-9')
    # Rollout Policy
    rollout_path = os.path.join(d, '../params/rollout/rollout.hdf5')
    # Tree Policy
    tree_path = os.path.join(d, '../params/rollout/tree.hdf5')
    # pattern hash for rollout
    rands_txt = os.path.join(d, '../params/rollout/mt_rands.txt')
    d12_rsp_csv = os.path.join(d, '../params/rollout/d12_rsp.csv')
    x33_csv = os.path.join(d, '../params/rollout/x33.csv')
    d12_csv = os.path.join(d, '../params/rollout/d12.csv')

    initialize_hash()
    initialize_nakade_hash()

    set_board_size(19)
    set_check_superko(superko)
    set_check_seki(seki)
    set_japanese_rule(japanese_rule)
    set_use_lgrf2(lgrf2)

    mcts = PyMCTS(const_time=const_time,
                  const_playout=const_playout,
                  n_threads=n_threads,
                  intuition=intuition,
                  read_ahead=False,
                  self_play=True)

    if use_pn:
        mcts.run_pn_session(pn_path, temperature=0.67)

    if use_vn:
        mcts.run_vn_session(vn_path)

    if use_rollout or use_tree:
        read_rands(rands_txt)
        initialize_rollout_const(8,
            init_x33_hash(x33_csv),
            init_d12_rsp_hash(d12_rsp_csv),
            init_d12_hash(d12_csv),
            pos_aware_d12=False)
        if use_rollout:
            mcts.set_rollout_parameter(rollout_path)
        if use_tree:
            mcts.set_tree_parameter(tree_path)

    for i in range(n_games):
        mcts.clear()
        game = mcts.game
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
