# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
from __future__ import division

import glob
import numpy as np
import os

from bamboo.gtp import gtp

cimport numpy as np

from libc.stdio cimport printf

from bamboo.sgf_util cimport SGFMoveIterator
from bamboo.board cimport PURE_BOARD_SIZE, BOARD_SIZE, OB_SIZE, S_EMPTY, S_BLACK, S_WHITE, PASS, RESIGN
from bamboo.board cimport FLIP_COLOR, CORRECT_X, CORRECT_Y
from bamboo.board cimport game_state_t, onboard_pos
from bamboo.board cimport set_board_size, initialize_board, allocate_game, free_game, put_stone, copy_game, calculate_score, komi, \
        set_check_superko, set_japanese_rule, set_check_seki, set_use_lgrf2
from bamboo.zobrist_hash cimport uct_hash_size
from bamboo.zobrist_hash cimport set_hash_size, initialize_hash, initialize_uct_hash, clear_uct_hash, delete_old_hash, search_empty_index, find_same_hash_index

from bamboo.rollout_preprocess cimport set_debug, initialize_rollout_const, initialize_rollout, set_rollout_parameter, set_tree_parameter, get_rollout_probs
from bamboo.local_pattern cimport read_rands, init_d12_rsp_hash, init_x33_hash, init_d12_hash
from bamboo.nakade cimport initialize_nakade_hash
from bamboo.printer cimport print_board
from bamboo.parseboard cimport parse


def play(sgf_glob):
    cdef int i, j
    cdef SGFMoveIterator sgf_iter
    cdef double probs[361]
    cdef int max_pos
    cdef double max_prob

    # Parameters
    d = os.path.dirname(os.path.abspath(__file__))
    # Rollout Policy
    rollout_path = os.path.join(d, '../../../params/rollout/rollout.hdf5')
    # Tree Policy
    tree_path = os.path.join(d, '../../../params/rollout/tree.hdf5')
    # pattern hash for rollout
    rands_txt = os.path.join(d, '../../../params/rollout/mt_rands.txt')
    d12_rsp_csv = os.path.join(d, '../../../params/rollout/d12_rsp.csv')
    x33_csv = os.path.join(d, '../../../params/rollout/x33.csv')
    d12_csv = os.path.join(d, '../../../params/rollout/d12.csv')

    initialize_hash()
    initialize_nakade_hash()

    set_board_size(19)

    read_rands(rands_txt)
    initialize_rollout_const(8,
        init_x33_hash(x33_csv),
        init_d12_rsp_hash(d12_rsp_csv),
        init_d12_hash(d12_csv),
        pos_aware_d12=False)
    set_rollout_parameter(rollout_path)
    set_tree_parameter(tree_path)

    n_total_state = 0
    n_total_correct = 0
    for sgf in glob.glob(sgf_glob):
        n_game_state = 0
        n_game_correct = 0
        try:
            with open(sgf, 'r') as sgf_object:
                sgf_iter = SGFMoveIterator(19,
                                           sgf_object.read(),
                                           rollout=True,
                                           ignore_not_legal=False,
                                           ignore_no_result=True)

            initialize_rollout(sgf_iter.game)
            for move in sgf_iter:
                # print_board(sgf_iter.game)
                n_total_state += 1
                n_game_state += 1
                max_pos = -1
                max_prob = -1.0
                if move[0] != PASS:
                    get_rollout_probs(sgf_iter.game, probs)
                    for i in range(361):
                        if probs[i] > max_prob:
                            max_pos = onboard_pos[i]
                            max_prob = probs[i]

                    if move[0] == max_pos:
                        n_total_correct += 1
                        n_game_correct += 1
        except:
            pass

        print('Game Acc: {:3.2f} % ({:d}/{:d})'. \
            format(n_game_correct/n_game_state, n_game_correct, n_game_state))

    print('Total Acc: {:3.2f} % ({:d}/{:d})'. \
        format(n_total_correct/n_total_state, n_total_correct, n_total_state))
