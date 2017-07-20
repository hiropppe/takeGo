# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

import numpy as np

cimport numpy as np

from libc.stdio cimport printf
from libc.stdlib cimport abort, malloc, free, rand
from libc.math cimport sqrt as csqrt

from libcpp.queue cimport queue as cppqueue

from cython import cdivision
from cython.parallel import prange

from bamboo.go.board cimport PURE_BOARD_SIZE, PURE_BOARD_MAX, MAX_RECORDS, MAX_MOVES, S_BLACK, S_WHITE, PASS
from bamboo.go.board cimport FLIP_COLOR
from bamboo.go.board cimport onboard_pos
from bamboo.go.board cimport game_state_t
from bamboo.go.board cimport set_board_size, initialize_board
from bamboo.go.board cimport put_stone, is_legal, is_legal_not_eye, do_move, allocate_game, free_game, copy_game, calculate_score
from bamboo.go.zobrist_hash cimport uct_hash_size, uct_hash_limit, hash_bit
from bamboo.go.zobrist_hash cimport mt, delete_old_hash, find_same_hash_index, search_empty_index

from bamboo.go.policy_feature cimport policy_feature_t
from bamboo.go.policy_feature cimport allocate_feature, initialize_feature, free_feature, update

from bamboo.rollout.preprocess cimport initialize_rollout, update_planes, update_planes_all, update_probs, update_probs_all

cimport openmp


cdef class MCTS(object):

    def __cinit__(self, object policy, int n_threads = 1):
        self.nodes = <tree_node_t *>malloc(uct_hash_size * sizeof(tree_node_t))
        self.current_root = uct_hash_size
        self.policy = policy
        self.policy_feature = allocate_feature()
        self.pondering = False
        self.n_threads = n_threads

        initialize_feature(self.policy_feature) 

    def __dealloc__(self):
        free_feature(self.policy_feature)
        if self.nodes:
            free(self.nodes)

    cdef void start_search_thread(self, game_state_t *game, int player_color):
        cdef int i
        cdef tree_node_t *node

        self.pondering = True

        delete_old_hash(game)

        self.seek_root(game)

        node = &self.nodes[self.current_root]
        if node.is_edge:
            self.expand(node, game)

        if self.n_threads <= 1:
            self.run_search(game, player_color) 
        else:
            for i in prange(self.n_threads, nogil=True):
                self.run_search(game, player_color) 

    cdef void stop_search_thread(self):
        self.pondering = False

    cdef void run_search(self, game_state_t *game, int player_color) nogil:
        cdef game_state_t *search_game
        cdef tree_node_t *node
        cdef int pos
        cdef unsigned long long previous_hash = 0

        search_game = allocate_game()

        while self.pondering:
            if previous_hash != game.current_hash:
                previous_hash = game.current_hash
                self.seek_root(game)
                node = &self.nodes[self.current_root]

            copy_game(search_game, game)

            self.search(node, search_game, player_color)

        free_game(search_game)

    cdef void search(self,
                     tree_node_t *node,
                     game_state_t *search_game,
                     int player_color) nogil:
        # selection
        while True:
            if node.is_edge:
                break
            else:
                node = self.select(node, search_game)

        # expansion
        if node.Nr >= EXPANSION_THRESHOLD:
            self.expand(node, search_game)
            node = self.select(node, search_game)

        # evaluation then backup
        self.evaluate_and_backup(node, search_game, player_color)

    cdef void seek_root(self, game_state_t *game) nogil:
        cdef tree_node_t *node
        cdef int pos
        cdef int color

        color = <int>game.current_color

        self.current_root = find_same_hash_index(game.current_hash, color, game.moves) 

        if self.current_root == uct_hash_size:
            self.current_root = search_empty_index(game.current_hash, color, game.moves)

            node = &self.nodes[self.current_root]
            node.node_i = self.current_root
            node.time_step = game.moves
            if game.moves > 0:
                node.pos = game.record[game.moves - 1].pos
                node.color = <int>game.record[game.moves - 1].color

            node.P = 0  # evaluate tree policy
            node.Nv = 0
            node.Nr = 0
            node.Wv = 0
            node.Wr = 0
            node.Q = 0
            node.u = 0
            node.num_child = 0
            node.is_root = True
            node.is_edge = True

            node.game = allocate_game()
            copy_game(node.game, game)

            self.policy_network_queue.push(node)
        else:
            node = &self.nodes[self.current_root]
            node.time_step = game.moves

    cdef tree_node_t *select(self, tree_node_t *node, game_state_t *game) nogil:
        cdef int i
        cdef tree_node_t *child
        cdef tree_node_t *max_child
        cdef float Qu_tmp = 0, Qu_max = 0

        for i in range(node.num_child):
            child = &node.children[i]
            Qu_tmp = child.Q + child.u
            if Qu_tmp > Qu_max:
                Qu_max = Qu_tmp
                max_child = child

        do_move(game, max_child.pos)

        return max_child

    cdef int expand(self, tree_node_t *node, game_state_t *game) nogil:
        cdef tree_node_t child
        cdef int child_pos, child_i
        cdef int child_moves = game.moves + 1
        cdef char other = FLIP_COLOR(game.current_color)
        cdef unsigned long long child_hash
        cdef int i

        for i in range(PURE_BOARD_MAX):
            child_pos = onboard_pos[i]
            if is_legal(game, child_pos, game.current_color):
                child_hash = game.current_hash ^ hash_bit[child_pos][<int>game.current_color]
                child_i = search_empty_index(child_hash, other, child_moves)
                # initialize new edge
                child = self.nodes[child_i]
                child.node_i = child_i
                child.time_step = child_moves
                child.pos = child_pos
                child.color = game.current_color
                child.P = 0  # evaluate tree policy
                child.Nv = 0
                child.Nr = 0
                child.Wv = 0
                child.Wr = 0
                child.Q = 0
                child.u = 0
                child.num_child = 0
                child.is_root = False
                child.is_edge = True
                child.parent = node

                node.children[i] = child
                node.children_pos[i] = child_pos
                node.num_child += 1
                node.is_edge = False

                node.game = allocate_game()
                copy_game(node.game, game)

                self.policy_network_queue.push(node)

    cdef void evaluate_and_backup(self,
                                  tree_node_t *node,
                                  game_state_t *game,
                                  int player_color) nogil:
        cdef int score
        cdef char winner
        cdef int Wr

        self.rollout(game)

        score = calculate_score(game)

        if score > 0:
            winner = S_BLACK
        elif score < 0:
            winner = S_WHITE

        if winner == player_color:
            Wr = 1
        else:
            Wr = 0

        self.backup(node, 1, Wr)

    cdef void rollout(self, game_state_t *game) nogil:
        cdef int color
        cdef int pos
        cdef int pass_count = 0
        cdef int moves_remain = MAX_MOVES - game.moves

        if moves_remain < 0:
            return

        initialize_rollout(game)

        color = game.current_color
        while moves_remain and pass_count < 2:
            update_planes(game)
            while True:
                pos = update_probs(game)
                if is_legal_not_eye(game, pos, color):
                    break
            put_stone(game, pos, color)
            pass_count = pass_count + 1 if pos == PASS else 0
            color = FLIP_COLOR(color)
            moves_remain -= 1

    cdef void backup(self, tree_node_t *edge_node, int Nr, int Wr) nogil:
        cdef tree_node_t *node = edge_node

        while True:
            node.Nr += 1
            node.Wr += Wr
            node.Q = (1 - MIXING_PARAMETER) * node.Wv/node.Nv + MIXING_PARAMETER * node.Wr/node.Nr
            node.u = EXPLORATION_CONSTANT * node.P * csqrt(node.parent.Nr) / (1 + node.Nr) 
            if node.is_root:
                break
            else:
                node = node.parent

    def run_policy_network(self):
        cdef tree_node_t *node
        cdef int i, pos

        while True:
            if self.policy_network_queue.empty():
                continue

            node = self.policy_network_queue.back()

            update(self.policy_feature, node.game)
            tensor = np.asarray(self.policy_feature.planes)
            tensor = tensor.reshape((1, 48, PURE_BOARD_SIZE, PURE_BOARD_SIZE))

            probs = self.policy.eval_state(tensor)
            for i in range(node.num_child):
                pos = node.children_pos[i]
                node.children[pos].P = probs[pos]

            free_game(node.game)

            self.policy_network_queue.pop()


# for random rollout test    
cdef void set_moves(int moves[], int size) nogil:
    cdef int i, j, t
    for i in range(361):
        moves[i] = i
    for i in range(361):
        j = rand() % size
        t = moves[i]
        moves[i] = moves[j]
        moves[j] = t

def testes(model, weights):
    cdef game_state_t *game
    from bamboo.models import policy
    policy = policy.CNNPolicy.load_model(model)
    policy.model.load_weights(weights)
    mcts = MCTS(policy)

    game = allocate_game()

    set_board_size(19)
    initialize_board(game)

    mcts.start_search_thread(game, S_BLACK)
