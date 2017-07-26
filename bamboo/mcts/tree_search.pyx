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
from bamboo.go.board cimport onboard_pos, komi
from bamboo.go.board cimport game_state_t
from bamboo.go.board cimport set_board_size, initialize_board
from bamboo.go.board cimport put_stone, is_legal, is_legal_not_eye, do_move, allocate_game, free_game, copy_game, calculate_score
from bamboo.go.zobrist_hash cimport uct_hash_size, uct_hash_limit, hash_bit
from bamboo.go.zobrist_hash cimport mt, delete_old_hash, find_same_hash_index, search_empty_index

from bamboo.go.policy_feature cimport policy_feature_t
from bamboo.go.policy_feature cimport allocate_feature, initialize_feature, free_feature, update

from bamboo.rollout.preprocess cimport initialize_rollout, update_rollout, update_planes, update_probs, set_illegal, choice_rollout_move

cimport openmp


cdef class MCTS(object):

    def __cinit__(self, object policy, int playout_limit=1000, int n_threads=1):
        self.nodes = <tree_node_t *>malloc(uct_hash_size * sizeof(tree_node_t))
        self.current_root = uct_hash_size
        self.policy = policy
        self.policy_feature = allocate_feature()
        self.pondering = False
        self.playout_limit = playout_limit
        self.n_threads = n_threads

        initialize_feature(self.policy_feature) 

    def __dealloc__(self):
        free_feature(self.policy_feature)
        if self.nodes:
            free(self.nodes)

    cdef int genmove(self, game_state_t *game):
        cdef tree_node_t *node
        cdef tree_node_t *child
        cdef int max_Nr = 0
        cdef double max_P = .0
        cdef int max_pos
        cdef int i

        self.seek_root(game)

        node = &self.nodes[self.current_root]
        max_Nr = 0
        max_pos = PASS

        if node.is_edge:
            self.expand(node, game)
            self.eval_leafs_by_policy_network(node)
            for i in range(node.num_child):
                child = node.children[node.children_pos[i]]
                if child.P > max_P:
                    max_pos = child.pos
                    max_P = child.P
        elif node.num_child > 0:
            for i in range(node.num_child):
                child = node.children[node.children_pos[i]]
                if child.Nr > max_Nr:
                    max_pos = child.pos
                    max_nr = child.Nr

        return max_pos


    cdef void start_search_thread(self, game_state_t *game):
        cdef int i
        cdef tree_node_t *node

        self.pondering = True

        delete_old_hash(game)

        self.seek_root(game)

        node = &self.nodes[self.current_root]
        if node.is_edge:
            self.expand(node, game)
            self.eval_leafs_by_policy_network(node)

        if self.n_threads <= 1:
            self.run_search(game) 
        else:
            for i in prange(self.n_threads, nogil=True):
                self.run_search(game) 

    cdef void stop_search_thread(self):
        self.pondering = False

    cdef void run_search(self, game_state_t *game) nogil:
        cdef game_state_t *search_game
        cdef tree_node_t *node
        cdef int pos
        cdef unsigned long long previous_hash = 0
        cdef int n_playout = 0

        search_game = allocate_game()

        while self.pondering:
            if previous_hash != game.current_hash:
                previous_hash = game.current_hash
                self.seek_root(game)
                node = &self.nodes[self.current_root]

            copy_game(search_game, game)

            self.search(node, search_game)

            n_playout += 1
            if n_playout > self.playout_limit:
                self.pondering = False
                break

        free_game(search_game)

    cdef void search(self,
                     tree_node_t *node,
                     game_state_t *search_game) nogil:
        cdef tree_node_t *current_node

        current_node = node

        # selection
        while True:
            if current_node.is_edge:
                break
            else:
                current_node = self.select(current_node, search_game)

        # expansion
        if current_node.Nr >= EXPANSION_THRESHOLD:
            self.expand(current_node, search_game)
            current_node = self.select(current_node, search_game)

        # evaluation then backup
        self.evaluate_and_backup(current_node, search_game)

    cdef bint seek_root(self, game_state_t *game) nogil:
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

            node.P = 0
            node.Nv = 0
            node.Wv = 0
            node.Nr = 0
            node.Wr = 0
            node.Q = 0
            node.u = 0
            node.Qu = 0
            node.num_child = 0
            node.is_root = True
            node.is_edge = True

            node.game = allocate_game()
            copy_game(node.game, game)
            node.has_game = True

            return False
        else:
            node = &self.nodes[self.current_root]
            node.is_root = True
            node.time_step = game.moves

            if not node.has_game:
                node.game = allocate_game()
                copy_game(node.game, game)
                node.has_game = True

            return True

    cdef tree_node_t *select(self, tree_node_t *node, game_state_t *game) nogil:
        cdef char color = game.current_color
        cdef tree_node_t *child
        cdef tree_node_t *max_child
        cdef double max_Qu = -1.0 
        cdef int i

        for i in range(node.num_child):
            child = node.children[node.children_pos[i]]
            if child.Qu > max_Qu:
                max_Qu = child.Qu
                max_child = child

        put_stone(game, max_child.pos, color)
        game.current_color = FLIP_COLOR(color)
        update_rollout(game)

        return max_child

    cdef void expand(self, tree_node_t *node, game_state_t *game) nogil:
        cdef tree_node_t *child
        cdef int child_pos, child_i
        cdef int child_moves = game.moves + 1
        cdef char color = game.current_color
        cdef char other_color = FLIP_COLOR(color)
        cdef unsigned long long child_hash
        cdef double *move_probs
        cdef int i

        # tree policy not implemented yet. substitutes by rollout policy
        move_probs = game.rollout_probs[color]

        for i in range(PURE_BOARD_MAX):
            child_pos = onboard_pos[i]
            if is_legal_not_eye(game, child_pos, color):
                child_hash = game.current_hash ^ hash_bit[child_pos][<int>color]
                child_i = search_empty_index(child_hash, color, child_moves)
                # initialize new edge
                child = &self.nodes[child_i]
                child.node_i = child_i
                child.time_step = child_moves
                child.pos = child_pos
                child.color = color
                child.P = move_probs[i]
                child.Nv = 0
                child.Wv = 0
                child.Nr = 0
                child.Wr = 0
                child.Q = 0
                child.u = child.P
                child.Qu = child.Q + child.u
                child.num_child = 0
                child.is_root = False
                child.is_edge = True
                child.parent = node
                child.has_game = False

                node.children[i] = child
                node.children_pos[node.num_child] = i
                node.num_child += 1

        node.is_edge = False

        node.game = allocate_game()
        copy_game(node.game, game)

        self.policy_network_queue.push(node)

    cdef void evaluate_and_backup(self,
                                  tree_node_t *node,
                                  game_state_t *game) nogil:
        cdef double score
        cdef int winner
        cdef int Wr

        self.rollout(game)

        score = <double>calculate_score(game)

        if score - komi > 0:
            winner = <int>S_BLACK
        else:
            winner = <int>S_WHITE

        self.backup(node, winner)

    cdef void rollout(self, game_state_t *game) nogil:
        cdef int color
        cdef int pos
        cdef int pass_count = 0
        cdef int moves_remain = MAX_MOVES - game.moves

        if moves_remain < 0:
            return

        color = game.current_color
        while moves_remain and pass_count < 2:
            while True:
                pos = choice_rollout_move(game)
                if is_legal_not_eye(game, pos, color):
                    break
                else:
                    set_illegal(game, pos)

            put_stone(game, pos, color)
            game.current_color = FLIP_COLOR(color)
            update_rollout(game)

            color = game.current_color

            pass_count = pass_count + 1 if pos == PASS else 0
            moves_remain -= 1

    cdef void backup(self, tree_node_t *edge_node, int winner) nogil:
        cdef tree_node_t *node

        node = edge_node
        while True:
            node.Nr += 1
            if node.color == winner:
                node.Wr += 1

            if node.is_root:
                break
            else:
                node.Q = node.Wr/node.Nr
                #node.Q = (1 - MIXING_PARAMETER) * node.Wv/node.Nv + MIXING_PARAMETER * node.Wr/node.Nr
                node.u = EXPLORATION_CONSTANT * node.P * csqrt(node.parent.Nr + 1) / (1 + node.Nr) 
                node.Qu = node.Q + node.u

                node = node.parent

    def run_policy_network(self):
        cdef tree_node_t *node
        cdef int i, pos

        while True:
            if self.policy_network_queue.empty():
                continue

            node = self.policy_network_queue.front()

            self.eval_leafs_by_policy_network(node)

            free_game(node.game)

            self.policy_network_queue.pop()

    cdef void eval_leafs_by_policy_network(self, tree_node_t *node):
        cdef int i, pos
        cdef tree_node_t *child

        update(self.policy_feature, node.game)

        tensor = np.asarray(self.policy_feature.planes)
        tensor = tensor.reshape((1, 48, PURE_BOARD_SIZE, PURE_BOARD_SIZE))

        probs = self.policy.eval_state(tensor)
        for i in range(node.num_child):
            pos = node.children_pos[i]
            child = node.children[pos]
            child.P = probs[pos]
            if child.parent.Nr > 0:
                child.u = EXPLORATION_CONSTANT * child.P * csqrt(child.parent.Nr) / (1 + child.Nr) 
            else:
                child.u = child.P 
            child.Qu = child.Q + child.u
