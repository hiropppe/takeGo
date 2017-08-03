# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

import time
import numpy as np

cimport numpy as np

from libc.stdio cimport printf
from libc.stdlib cimport abort, malloc, free, rand
from libc.math cimport sqrt as csqrt
from libc.time cimport clock as cclock
from libcpp.queue cimport queue as cppqueue
from libcpp.string cimport string as cppstring
from posix.time cimport gettimeofday, timeval, timezone

from cython import cdivision
from cython.parallel import prange

from bamboo.go.board cimport PURE_BOARD_SIZE, PURE_BOARD_MAX, MAX_RECORDS, MAX_MOVES, S_BLACK, S_WHITE, PASS
from bamboo.go.board cimport FLIP_COLOR
from bamboo.go.board cimport onboard_pos, komi
from bamboo.go.board cimport game_state_t
from bamboo.go.board cimport set_board_size, set_komi, initialize_board
from bamboo.go.board cimport put_stone, is_legal, is_legal_not_eye, do_move, allocate_game, free_game, copy_game, calculate_score
from bamboo.go.zobrist_hash cimport uct_hash_size, uct_hash_limit, hash_bit, used
from bamboo.go.zobrist_hash cimport mt, delete_old_hash, find_same_hash_index, search_empty_index, check_remaining_hash_size
from bamboo.go.policy_feature cimport policy_feature_t
from bamboo.go.policy_feature cimport allocate_feature, initialize_feature, free_feature, update
from bamboo.rollout.preprocess cimport set_debug, initialize_rollout, update_rollout, update_planes, update_probs, set_illegal, choice_rollout_move

from bamboo.go.printer cimport print_board, print_prior_probability, print_rollout_count, print_winning_ratio, print_action_value, print_bonus, print_selection_value
from bamboo.util cimport save_gamestate_to_sgf 

cimport openmp


cdef class MCTS(object):

    def __cinit__(self,
                  object policy,
                  double temperature=0.67,
                  int playout_limit=5000,
                  int n_threads=1):
        self.nodes = <tree_node_t *>malloc(uct_hash_size * sizeof(tree_node_t))
        self.current_root = uct_hash_size
        self.policy = policy
        self.policy_feature = allocate_feature()
        self.pondering = False
        self.policy_queue_running = False
        self.n_playout = 0
        self.beta = 1.0/temperature
        self.playout_limit = playout_limit
        self.n_threads = n_threads
        self.max_queue_size_P = 0
        self.debug = False

        initialize_feature(self.policy_feature) 

    def __dealloc__(self):
        free_feature(self.policy_feature)
        if self.nodes:
            free(self.nodes)

    cdef int genmove(self, game_state_t *game) nogil:
        cdef tree_node_t *node
        cdef tree_node_t *child
        cdef double max_Nr = .0
        cdef double max_P = .0
        cdef int max_pos
        cdef int i

        self.seek_root(game)

        node = &self.nodes[self.current_root]
        max_Nr = 0
        max_pos = PASS

        print_selection_value(node)
        print_prior_probability(node)
        print_bonus(node)
        print_winning_ratio(node)
        # print_action_value(node)
        print_rollout_count(node)

        for i in range(node.num_child):
            child = node.children[node.children_pos[i]]
            if child.Nr > max_Nr:
                max_pos = child.pos
                max_Nr = child.Nr

        return max_pos

    cdef void start_search_thread(self, game_state_t *game):
        cdef char *stone = ['#', 'B', 'W']
        cdef int i
        cdef tree_node_t *node
        cdef bint expanded
        cdef timeval start_time, end_time
        cdef double elapsed

        self.pondering = True

        gettimeofday(&start_time, NULL)

        delete_old_hash(game)

        self.seek_root(game)

        node = &self.nodes[self.current_root]

        printf(">> Playout (%s)\n", cppstring(1, stone[node.player_color]).c_str())
        if node.Nr != 0.0:
            printf('Pre Playouts       : %d\n', <int>node.Nr)
            printf('Pre Winning ratio  : %3.2lf %\n', 100.0-(node.Wr*100.0/node.Nr))
        else:
            printf('No playout information found for current state.\n')

        if node.is_edge:
            expanded = self.expand(node, game)
            if expanded:
                self.eval_leafs_by_policy_network(node)
        else:
            print_rollout_count(node)

        if self.n_threads <= 1:
            self.run_search(game) 
        else:
            for i in prange(self.n_threads, nogil=True):
                self.run_search(game) 

        gettimeofday(&end_time, NULL)
        elapsed = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1000000.0;

        printf('Playouts           : %d\n', self.n_playout)
        printf('Elapsed            : %2.3lf sec\n', elapsed)
        if elapsed != 0.0:
            printf('Playout Speed      : %d PO/sec\n', <int>(self.n_playout/elapsed))
        printf('Total Playouts     : %d\n', <int>node.Nr)
        printf("Winning Ratio      : %3.2lf %\n", 100.0-(node.Wr*100.0/node.Nr))
        printf('Queue size (P)     : %d\n', self.policy_network_queue.size())
        printf('Queue size max (P) : %d\n', self.max_queue_size_P)
        printf('Hash status of use : %3.2lf % (%u/%u)\n', used*100.0/uct_hash_size, used, uct_hash_size)

        self.pondering = False

    cdef void stop_search_thread(self):
        self.pondering = False

    cdef void run_search(self, game_state_t *game) nogil:
        cdef game_state_t *search_game
        cdef tree_node_t *node
        cdef int pos
        cdef unsigned long long previous_hash = 0

        self.n_playout = 0

        search_game = allocate_game()

        while (self.pondering and
               check_remaining_hash_size() and
               self.n_playout < self.playout_limit):

            if game.moves == 0 or previous_hash != game.current_hash:
                previous_hash = game.current_hash
                self.seek_root(game)
                node = &self.nodes[self.current_root]

            copy_game(search_game, game)

            self.search(node, search_game)

            self.n_playout += 1

            # workaround. CPU cannot be assigned.
            if self.policy_network_queue.size() > 10:
                with gil:
                    time.sleep(.05)

        free_game(search_game)

    cdef void search(self,
                     tree_node_t *node,
                     game_state_t *search_game) nogil:
        cdef tree_node_t *current_node
        cdef bint expanded

        current_node = node

        # selection
        while True:
            if current_node.is_edge:
                break
            else:
                current_node = self.select(current_node, search_game)

        # expansion
        if current_node.Nr >= EXPANSION_THRESHOLD:
            expanded = self.expand(current_node, search_game)
            if expanded:
                current_node = self.select(current_node, search_game)

        # evaluation then backup
        self.evaluate_and_backup(current_node, search_game)

    cdef bint seek_root(self, game_state_t *game) nogil:
        cdef tree_node_t *node
        cdef int pos
        cdef char color, other_color

        color = game.current_color
        other_color = FLIP_COLOR(color)

        self.current_root = find_same_hash_index(game.current_hash, color, game.moves) 

        if self.current_root == uct_hash_size:
            self.current_root = search_empty_index(game.current_hash, color, game.moves)
            node = &self.nodes[self.current_root]
            node.node_i = self.current_root
            node.time_step = game.moves
            if game.moves == 0:
                node.color = other_color
            else:
                node.pos = game.record[game.moves - 1].pos
                node.color = <int>game.record[game.moves - 1].color
            node.player_color = color
            node.P = 0
            node.Nv = 0
            node.Wv = 0
            node.Nr = 0
            node.Wr = 0
            node.Q = 0
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
        cdef char color
        cdef tree_node_t *child
        cdef tree_node_t *max_child
        cdef double child_u, child_Qu
        cdef double max_Qu = -1.0 
        cdef int i

        color = game.current_color

        for i in range(node.num_child):
            child = node.children[node.children_pos[i]]
            if node.Nr > .0:
                child_u = EXPLORATION_CONSTANT * child.P * (csqrt(node.Nr) / (1 + child.Nr))
            else:
                child_u = EXPLORATION_CONSTANT * child.P
            child_Qu = child.Q + child_u
            if child_Qu > max_Qu:
                max_Qu = child_Qu
                max_child = child

        put_stone(game, max_child.pos, color)
        game.current_color = FLIP_COLOR(color)
        update_rollout(game)

        return max_child

    cdef bint expand(self, tree_node_t *node, game_state_t *game) nogil:
        cdef tree_node_t *child
        cdef int child_pos, child_i
        cdef int child_moves = game.moves + 1
        cdef char color = game.current_color
        cdef char other_color = FLIP_COLOR(color)
        cdef unsigned long long child_hash
        cdef double *move_probs
        cdef int queue_size
        cdef int i

        # tree policy not implemented yet. use rollout policy instead.
        move_probs = game.rollout_probs[color]

        for i in range(PURE_BOARD_MAX):
            child_pos = onboard_pos[i]
            if is_legal_not_eye(game, child_pos, color):
                child_hash = game.current_hash ^ hash_bit[child_pos][<int>color]
                child_i = search_empty_index(child_hash, other_color, child_moves)
                # initialize new edge
                child = &self.nodes[child_i]
                child.node_i = child_i
                child.time_step = child_moves
                child.pos = child_pos
                child.color = color
                child.player_color = other_color
                child.P = move_probs[i]
                child.Nv = 0
                child.Wv = 0
                child.Nr = 0
                child.Wr = 0
                child.Q = 0
                child.num_child = 0
                child.is_root = False
                child.is_edge = True
                child.parent = node
                child.has_game = False

                node.children[i] = child
                node.children_pos[node.num_child] = i
                node.num_child += 1

        if node.num_child > 0:
            node.is_edge = False
            node.game = allocate_game()
            copy_game(node.game, game)
            self.policy_network_queue.push(node)
            queue_size = self.policy_network_queue.size()
            if queue_size > self.max_queue_size_P:
                self.max_queue_size_P = queue_size
            return True
        else:
            return False

    cdef void evaluate_and_backup(self,
                                  tree_node_t *node,
                                  game_state_t *game) nogil:
        cdef double score
        cdef int winner

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
                node = node.parent

    def start_policy_network_queue(self):
        cdef tree_node_t *node
        cdef int i, pos
        cdef int n_eval = 0

        if self.policy_queue_running:
            printf('>> Policy network queue already running.\n')
            return

        self.policy_queue_running = True

        printf('>> Starting policy network queue...\n')

        while self.policy_queue_running:
            if self.policy_network_queue.empty():
                if n_eval > 0:
                    printf('>> Policy Network Queue evaluated: %d node\n', n_eval)
                n_eval = 0
                time.sleep(.5)
                continue

            node = self.policy_network_queue.front()

            self.eval_leafs_by_policy_network(node)

            free_game(node.game)

            self.policy_network_queue.pop()

            n_eval += 1

        printf('>> Policy network queue shut down.\n')

    def stop_policy_network_queue(self):
        print('>> Shutting down policy network queue...')
        self.policy_queue_running = False

    def eval_all_leafs_by_policy_network(self):
        cdef tree_node_t *node
        cdef int i, pos
        cdef int n_eval = 0

        while not self.policy_network_queue.empty():

            node = self.policy_network_queue.front()

            self.eval_leafs_by_policy_network(node)

            free_game(node.game)

            self.policy_network_queue.pop()

            n_eval += 1

        printf('>> Policy Network Queue evaluated: %d node\n', n_eval)

    cdef void eval_leafs_by_policy_network(self, tree_node_t *node):
        cdef int i, pos
        cdef tree_node_t *child

        update(self.policy_feature, node.game)

        tensor = np.asarray(self.policy_feature.planes)
        tensor = tensor.reshape((1, 48, PURE_BOARD_SIZE, PURE_BOARD_SIZE))

        probs = self.policy.eval_state(tensor)
        probs = self.apply_temperature(probs)
        for i in range(node.num_child):
            pos = node.children_pos[i]
            child = node.children[pos]
            child.P = probs[pos]

    def apply_temperature(self, distribution):
        log_probabilities = np.log(distribution)
        log_probabilities = log_probabilities * self.beta
        log_probabilities = log_probabilities - log_probabilities.max()
        probabilities = np.exp(log_probabilities)
        return probabilities / probabilities.sum()


cdef class PyMCTS(object):

    def __cinit__(self,
                  object policy,
                  double temperature=0.67,
                  int playout_limit=8000,
                  int n_threads=1):
        self.mcts = MCTS(policy, temperature, playout_limit, n_threads)

        self.game = allocate_game()
        initialize_board(self.game)
        initialize_rollout(self.game)

    def clear(self):
        self.mcts.stop_search_thread()
        initialize_board(self.game)
        initialize_rollout(self.game)

    def start_pondering(self):
        self.mcts.start_search_thread(self.game)

    def stop_pondering(self):
        self.mcts.stop_search_thread()

    def genmove(self, color):
        self.game.current_color = color
        pos = self.mcts.genmove(self.game)
        return pos

    def play(self, pos, color):
        cdef bint legal
        legal = put_stone(self.game, pos, color)
        if legal:
            self.game.current_color = FLIP_COLOR(self.game.current_color)
            update_rollout(self.game)
            print_board(self.game)
            return True
        else:
            return False

    def start_policy_network_queue(self):
        self.mcts.start_policy_network_queue()

    def stop_policy_network_queue(self):
        self.mcts.stop_policy_network_queue()

    def eval_all_leafs_by_policy_network(self):
        self.mcts.eval_all_leafs_by_policy_network()

    def set_size(self, bsize):
        set_board_size(bsize)

    def set_komi(self, new_komi):
        set_komi(new_komi)

    def set_time(self, m, b, stone):
        pass

    def set_time_left(self, color, time, stone):
        pass

    def set_playout_limit(self, limit):
        self.mcts.playout_limit = limit

    def showboard(self):
        print_board(self.game)

    def save_sgf(self, black_name, white_name):
        from tempfile import NamedTemporaryFile
        temp_file = NamedTemporaryFile(delete=False)
        temp_file_name = temp_file.name + '.sgf'
        save_gamestate_to_sgf(self.game, '/tmp/', temp_file_name, black_name, white_name)
        return temp_file_name

    def quit(self):
        self.mcts.stop_search_thread()
        initialize_board(self.game)
        initialize_rollout(self.game)

