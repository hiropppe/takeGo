# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

import numpy as np

cimport numpy as np
np.set_printoptions(suppress=True, linewidth=200, precision=3)

from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy

from libcpp.vector cimport vector as cppvector

import os
import subprocess
import time

from .models.keras_dcnn_policy import KerasPolicy, cnn_policy

from .gtp import gtp

from .board cimport PURE_BOARD_SIZE, PURE_BOARD_MAX, BOARD_SIZE, OB_SIZE, S_BLACK, S_WHITE, PASS, RESIGN
from .board cimport FLIP_COLOR, CORRECT_X, CORRECT_Y
from .board cimport game_state_t, set_board_size, initialize_board, allocate_game, free_game
from .board cimport put_stone, is_legal, do_move, get_legal_moves_mask, onboard_pos, calculate_score, komi

from .printer cimport print_board, print_PN

from . cimport policy_feature as pf
from .policy_feature cimport PolicyFeature
from .policy_feature cimport MAX_POLICY_PLANES

from .player cimport PolicyPlayer

from .sgf_util cimport save_gamestate_to_sgf

from .goscorer import final_territory_score, territory_scoring

ctypedef np.int32_t INT_t


cdef np.ndarray[INT_t, ndim=4] state_to_tensor(PolicyFeature policy_feature, game_state_t *game):
    cdef np.ndarray[INT_t, ndim=4] state_tensor

    pf.update(policy_feature, game)

    state_tensor = np.asarray(policy_feature.planes) \
        .reshape((1, MAX_POLICY_PLANES, PURE_BOARD_SIZE, PURE_BOARD_SIZE))
    state_tensor = np.transpose(state_tensor, (0, 2, 3, 1)) 

    return state_tensor.copy()


cdef double territory_score(game_state_t *game, double komi):
    cdef int pos, ob_pos, color

    stones = np.zeros(PURE_BOARD_MAX)
    for pos in range(PURE_BOARD_MAX):
        ob_pos = onboard_pos[pos]
        color = game.board[ob_pos]
        stones[pos] = color
    stones = stones.reshape(PURE_BOARD_SIZE, PURE_BOARD_SIZE)
    marked_dead = np.full((PURE_BOARD_SIZE, PURE_BOARD_SIZE), False)
    final_score = final_territory_score(
        stones,
        marked_dead,
        black_points_from_captures=0,
        white_points_from_captures=0,
        komi=komi,
    )
    return final_score[S_BLACK] - final_score[S_WHITE]


cpdef run_n_games(object player_pn,
                  object opponent_pn,
                  int n_games=20,
                  int move_limit=361,
                  double temperature=0.67,
                  bint greedy=0,
                  int verbose=0):
    cdef int i, j
    cdef game_state_t *games, *game
    cdef bint legal
    cdef int pos, ob_pos
    cdef double score
    cdef PolicyPlayer player, opponent, current, other
    cdef np.ndarray[INT_t, ndim=4] state_tensor
    cdef PolicyFeature policy_feature
    cdef list states = []
    #cdef list masks = []
    cdef int[:] moves
    cdef int[:] games_in_play = np.ones(n_games, dtype=np.int32)
    cdef list i_games_in_play = []
    cdef int n_games_in_play = n_games
    cdef np.ndarray[np.npy_bool, ndim=1] legal_moves_mask
    cdef np.ndarray[np.npy_bool, ndim=2] legal_moves_masks

    set_board_size(19)

    games = <game_state_t *>malloc(n_games * sizeof(game_state_t))
    memset(games, 0, n_games * sizeof(game_state_t))

    for i in range(n_games):
        initialize_board(&games[i])

    feature = pf.allocate_feature(MAX_POLICY_PLANES)
    pf.initialize_feature(feature)

    # Create one list of features (aka state tensors) and one of moves for each game being played.
    state_tensors = [[] for _ in range(n_games)]
    move_tensors  = [[] for _ in range(n_games)]

    # List of booleans indicating whether the 'learner' player won.
    learner_won = [None] * n_games

    #  Even games will have 'learner' black.
    learner_color = [S_BLACK if i % 2 == 0 else S_WHITE for i in range(n_games)]

    player = PolicyPlayer(player_pn, temperature=temperature, greedy=greedy)
    opponent = PolicyPlayer(opponent_pn, temperature=temperature, greedy=greedy)
    
    # Start all odd games with moves by 'opponent' because First current player is learner.
    if n_games > 1:
        states = []
        for i in range(1, n_games, 2):
            game = &games[i]
            state_tensor = state_to_tensor(feature, game)
            states.append(state_tensor)
        
        moves = opponent.genmove(np.vstack(states))

        for j, i in enumerate(range(1, n_games, 2)):
            game = &games[i]
            pos = moves[j]
            ob_pos = onboard_pos[pos]
            put_stone(game, ob_pos, game.current_color)
            game.current_color = FLIP_COLOR(game.current_color)

    current, other = player, opponent

    while n_games_in_play > 0:
        del states[:]
        del i_games_in_play[:]
        #del masks[:]
        for i in range(n_games):
            if games_in_play[i]:
                i_games_in_play.append(i)
                game = &games[i]
                state_tensor = state_to_tensor(feature, game)
                states.append(state_tensor)

                #legal_moves_mask = get_legal_moves_mask(game, game.current_color)
                #masks.append(legal_moves_mask)

        moves = current.genmove(np.vstack(states))

        #legal_moves_masks = np.vstack(masks)
        #moves = current.gen_masked_move(np.vstack(states), legal_moves_masks)

        assert len(states) == len(moves) == n_games_in_play

        for j in range(n_games_in_play):
            i = i_games_in_play[j]
            pos = moves[j]
            ob_pos = onboard_pos[pos]
            game = &games[i]
            state_tensor = states[j]

            if verbose >= 4:
                ## display moves
                print_board(game)
                print("")
                probs = current.model.eval_state(state_tensor)
                probs = probs[0]
                print(probs.reshape(19, 19))
                print("")

                cx = CORRECT_X(ob_pos, BOARD_SIZE, OB_SIZE)
                cy = CORRECT_Y(ob_pos, BOARD_SIZE, OB_SIZE)
                px = cx + 1
                py = PURE_BOARD_SIZE - cy
                vtx = gtp.gtp_vertex((px, py))
                print(f"#{str(i).zfill(3)}. Moves: {game.moves} {'Black' if game.current_color == S_BLACK else 'White'} > {vtx} ({cx}, {cy})")
                print("")

            if learner_color[i] == game.current_color:
                state_tensors[i].append(state_tensor)
                move_tensor = np.zeros((1, PURE_BOARD_MAX))
                move_tensor[(0, pos)] = 1                    
                move_tensors[i].append(move_tensor)

            if is_legal(game, ob_pos, game.current_color) and pos != RESIGN and game.moves <= move_limit:
                put_stone(game, ob_pos, game.current_color)

                if verbose == 3:
                    print_board(game)

                game.current_color = FLIP_COLOR(game.current_color)
            else:
                games_in_play[i] = 0
                n_games_in_play -= 1

                score = territory_score(game, komi)

                if score > 0:
                    winner = S_BLACK
                else:
                    winner = S_WHITE

                #if game.moves <= 100:
                #    winner = FLIP_COLOR(game.current_color)

                learner_won[i] = winner == learner_color[i]

                if verbose:
                    cx = CORRECT_X(ob_pos, BOARD_SIZE, OB_SIZE)
                    cy = CORRECT_Y(ob_pos, BOARD_SIZE, OB_SIZE)
                    px = cx + 1
                    py = PURE_BOARD_SIZE - cy
                    last_move = gtp.gtp_vertex((px, py))
                    print(f"#{str(i).zfill(3)}. {'Won' if learner_won[i] else 'Lost'} ({'B' if learner_color[i] == S_BLACK else 'W'}) Score: {score} Moves: {game.moves} Last Color: {'B' if game.current_color == S_BLACK else 'W'} Last Move: {last_move} ({cx}, {cy})")

                    #print(f"#{str(i).zfill(3)}. {'*' if np.sign(score) != np.sign(calculate_score(game) - komi) else ''}{'Won' if learner_won[i] else 'Lost'} ({'B' if learner_color[i] == S_BLACK else 'W'}) Score: {score} ({calculate_score(game) - komi}) Moves: {game.moves} Last Color: {'B' if game.current_color == S_BLACK else 'W'} Last Move: {last_move} ({cx}, {cy})")

                    #save_gamestate_to_sgf(game, '/tmp', f'self_play_{i}.sgf', 'B', 'W')
                    #print(subprocess.check_output(["gnugo", "--score", "aftermath", "-l", f"/tmp/self_play_{i}.sgf"]))

                if verbose >= 2:
                    print_board(game)
                    #print(state_tensor[0, :, :, 46])  # Whether a move is legal and does not fill its own eyes
                    probs = current.model.eval_state(state_tensor)
                    probs = probs[0]
                    print(probs.reshape(19, 19))

            current, other = other, current

    free_game(games)

    win_ratio = sum(learner_won) / n_games
    
    return state_tensors, move_tensors, learner_won, win_ratio
