# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
from __future__ import division

import glob
import numpy as np
import os
import sgf
import sys
import traceback
import time

from bamboo.gtp import gtp

cimport numpy as np

from libc.stdio cimport printf

from bamboo.board cimport PURE_BOARD_SIZE, BOARD_SIZE, OB_SIZE, S_EMPTY, S_BLACK, S_WHITE, PASS, RESIGN
from bamboo.board cimport POS, FLIP_COLOR, CORRECT_X, CORRECT_Y
from bamboo.board cimport game_state_t, onboard_pos
from bamboo.board cimport set_board_size, initialize_board, allocate_game, free_game, put_stone, copy_game, is_legal, calculate_score, komi, \
        set_check_superko, set_japanese_rule, set_check_seki, set_use_lgrf2
from bamboo.zobrist_hash cimport uct_hash_size
from bamboo.zobrist_hash cimport set_hash_size, initialize_hash, initialize_uct_hash, clear_uct_hash, delete_old_hash, search_empty_index, find_same_hash_index

from bamboo.rollout_preprocess cimport set_debug, initialize_rollout_const, initialize_rollout, set_rollout_parameter, set_tree_parameter, get_rollout_probs, update_rollout, update_tree_planes_all, get_tree_probs
from bamboo.local_pattern cimport read_rands, init_d12_rsp_hash, init_x33_hash, init_d12_hash
from bamboo.nakade cimport initialize_nakade_hash
from bamboo.printer cimport print_board
from bamboo.tree_search cimport PyMCTS 


LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def parse_sgf_move(node_value):
    """Given a well-formed move string, return either PASS_MOVE or the (x, y) position
    """
    if node_value == '' or node_value == 'tt':
        return PASS
    else:
        x = LETTERS.index(node_value[0].upper())
        y = LETTERS.index(node_value[1].upper())
        return POS(x+OB_SIZE, y+OB_SIZE, BOARD_SIZE)


def debug():
    cdef PyMCTS mcts
    import argparse
    parser = argparse.ArgumentParser(description='Run GTP')
    parser.add_argument("--sgf", "-s", type=str, default=None, required=True,
                        help="Moves to show evaluation.")
    parser.add_argument("--eval_move", type=int, default=None,
                        help="Move to show evaluation.")
    parser.add_argument("--policy_temperature", type=float, default=0.67,
                        help="Distribution temperature of players using policies (Default: 0.67)")
    parser.add_argument("--threads", "-t", type=int, default=1,
                        help="Number of search threads (Default: 1)")
    parser.add_argument("--node_hash_size", "-n", type=int, default=1048576,
                        help="MCT node hash size (Default: 2**20)")
    parser.add_argument("--const_time", type=float, default=0.0,
                        help="Constant time of simulation for each move. Enable if non-zero value set. (Default: 0.0)")
    parser.add_argument("--const_playout", type=int, default=10000,
                        help="Constant number of simulations for eval move. (Default: 10000)")
    parser.add_argument("--superko", "-superko", default=True, action="store_true",
                        help="Use positional superko. (Default: True)")
    parser.add_argument("--japanese_rule", "-jp", default=False, action="store_true",
                        help="Use japanese rule. (Default: False)")
    parser.add_argument("--seki", "-seki", default=False, action="store_true",
                        help="Check seki at tree expansion. (Default: False)")
    parser.add_argument("--lgrf2", "-lgrf2", default=False, action="store_true",
                        help="Use LGRF2. (Default: False)")
    parser.add_argument("--nogpu", default=False, action="store_true",
                        help="Play using CPU only. (Default: False)")

    args = parser.parse_args()

    set_board_size(19)
    set_check_superko(args.superko)
    set_japanese_rule(args.japanese_rule)
    set_check_seki(args.seki)
    set_use_lgrf2(args.lgrf2)
    set_hash_size(args.node_hash_size)

    # init tree hash
    initialize_hash()

    d = os.path.dirname(os.path.abspath(__file__))

    # local pattern hash for rollout
    rands_txt = os.path.join(d, '../params/rollout/mt_rands.txt')
    d12_rsp_csv = os.path.join(d, '../params/rollout/d12_rsp.csv')
    x33_csv = os.path.join(d, '../params/rollout/x33.csv')
    d12_csv = os.path.join(d, '../params/rollout/d12.csv')

    read_rands(rands_txt)
    nakade_size = initialize_nakade_hash()
    x33_size = init_x33_hash(x33_csv)
    d12_rsp_size = init_d12_rsp_hash(d12_rsp_csv)
    d12_size = init_d12_hash(d12_csv)

    # init rollout const
    initialize_rollout_const(nakade_size,
        x33_size,
        d12_rsp_size,
        d12_size,
        pos_aware_d12=False)

    # Policy Network
    pn_path = os.path.join(d, '../params/policy/weights.hdf5')
    # Rollout Policy
    rollout_path = os.path.join(d, '../params/rollout/rollout.hdf5')
    # Tree Policy
    tree_path = os.path.join(d, '../params/rollout/tree.hdf5')

    mcts = PyMCTS(const_time=args.const_time,
                  const_playout=args.const_playout,
                  n_threads=args.threads,
                  nogpu=args.nogpu,
                  self_play=True)

    mcts.run_pn_session(pn_path, args.policy_temperature)
    mcts.set_rollout_parameter(rollout_path)
    mcts.set_tree_parameter(tree_path)

    mcts.set_size(19)
    mcts.clear()

    sgf_string = open(args.sgf).read()  
    try:
        collection = sgf.parse(sgf_string)
    except sgf.ParseException:
        err, msg, _ = sys.exc_info()
        sys.stderr.write("{:s} {:s}\n{:s}".format(err, msg, sgf_string))
        sys.stderr.write(traceback.format_exc())

    sgf_game = collection[0]

    for i, node in enumerate(sgf_game.rest):
        if i+1 == args.eval_move:
            break
        props = node.properties
        if 'W' in props:
            pos = parse_sgf_move(props['W'][0])
            mcts.play(pos, S_WHITE)
        elif 'B' in props:
            pos = parse_sgf_move(props['B'][0])
            mcts.play(pos, S_BLACK)

    mcts.genmove(mcts.game.current_color)
