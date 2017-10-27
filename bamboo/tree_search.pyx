# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

import time
import numpy as np
import tensorflow as tf

from tensorflow.contrib.keras.python.keras.backend import set_session

from bamboo.models.keras_dcnn_policy import CNNPolicy
from bamboo.models.dcnn_resnet_value import inference_agz

cimport numpy as np

from libc.stdio cimport printf
from libc.stdlib cimport abort, malloc, free, rand
from libc.math cimport sqrt as csqrt
from libc.time cimport clock as cclock
from libcpp.queue cimport queue as cppqueue
from libcpp.string cimport string as cppstring
from libcpp.vector cimport vector as cppvector
from posix.time cimport gettimeofday, timeval, timezone

from cython import cdivision
from cython.parallel import prange
from cython.operator cimport dereference as deref, preincrement as inc

from bamboo.board cimport PURE_BOARD_SIZE, PURE_BOARD_MAX, BOARD_MAX, MAX_RECORDS, MAX_MOVES, S_BLACK, S_WHITE, PASS, RESIGN
from bamboo.board cimport FLIP_COLOR, DMAX, DMIN
from bamboo.board cimport onboard_pos, komi
from bamboo.board cimport game_state_t
from bamboo.board cimport set_board_size, set_komi, initialize_board
from bamboo.board cimport put_stone, is_legal, is_legal_not_eye, do_move, allocate_game, free_game, copy_game, calculate_score
from bamboo.board cimport use_lgrf2_flag, check_seki_flag
from bamboo.seki cimport check_seki
from bamboo.zobrist_hash cimport uct_hash_size, uct_hash_limit, hash_bit, used
from bamboo.zobrist_hash cimport mt, initialize_uct_hash, delete_old_hash, find_same_hash_index, search_empty_index, check_remaining_hash_size
from bamboo.policy_feature cimport MAX_POLICY_PLANES, MAX_VALUE_PLANES
from bamboo.policy_feature cimport policy_feature_t
from bamboo.policy_feature cimport allocate_feature, initialize_feature, free_feature, update
from bamboo.rollout_preprocess cimport set_rollout_parameter, set_tree_parameter
from bamboo.rollout_preprocess cimport set_debug, initialize_rollout, update_rollout, update_planes, \
        update_probs, set_illegal, choice_rollout_move, update_tree_planes_all, \
        get_tree_probs, get_rollout_probs

from bamboo.printer cimport print_board, print_PN, print_VN, print_rollout_count, print_winning_ratio, print_Q, print_u, print_selection_value
from bamboo.sgf_util cimport save_gamestate_to_sgf 

cimport openmp


cdef class MCTS(object):

    def __cinit__(self,
                  double const_time=5.0,
                  int const_playout=0,
                  int n_threads=1,
                  bint intuition=False,
                  bint self_play=False):
        cdef int i, j, k
        cdef tree_node_t *node
        cdef game_state_t *queue_entry

        self.game = NULL
        self.player_color = 0
        self.nodes = <tree_node_t *>malloc(uct_hash_size * sizeof(tree_node_t))
        for i in range(uct_hash_size):
            node = &self.nodes[i]
            node.node_i = 0
            node.time_step = 0 
            node.pos = 0
            node.color = 0
            node.player_color = 0
            node.P = .0
            node.Nv = .0
            node.Wv = .0
            node.Nr = .0
            node.Wr = .0
            node.Q = .0
            node.is_root = False
            node.is_edge = False
            node.parent = NULL
            node.num_child = 0
            node.game = NULL
            node.has_game = False

        self.current_root = uct_hash_size
        self.policy_feature = allocate_feature(MAX_POLICY_PLANES)
        self.value_feature = allocate_feature(MAX_VALUE_PLANES)
        self.intuition = intuition
        self.use_vn = False
        self.use_rollout = False
        self.use_tree = False
        self.pondered = False
        self.pondering = False
        self.pondering_stopped = True
        self.pondering_suspending = False
        self.pondering_suspended = True
        self.policy_queue_running = False
        self.value_queue_running = False
        self.n_playout = 0
        self.winning_ratio = 0.5
        self.main_time = 0.0
        self.byoyomi_time = 0.0
        self.time_left = 0.0
        self.can_extend = False
        self.const_time = const_time
        self.const_playout = const_playout
        self.n_threads = n_threads
        self.max_queue_size_P = 0
        self.max_queue_size_V = 0
        self.debug = False
        self.self_play = self_play

        initialize_feature(self.policy_feature) 
        initialize_feature(self.value_feature) 

        for i in range(2):
            for j in range(BOARD_MAX):
                for k in range(BOARD_MAX):
                    self.lgr2[i+1][j][k] = PASS

        for i in range(n_threads):
            self.n_threads_playout[i] = 0

        # pre allocate 500 game for VN
        #for i in range(500):
        #    self.game_queue.push(allocate_game())

        openmp.omp_init_lock(&self.tree_lock)
        openmp.omp_init_lock(&self.expand_lock)
        openmp.omp_init_lock(&self.policy_queue_lock)
        openmp.omp_init_lock(&self.value_queue_lock)
        #openmp.omp_init_lock(&self.game_queue_lock)

    def __dealloc__(self):
        cdef game_state_t *game

        if self.nodes:
            free(self.nodes)

        free_feature(self.policy_feature)
        free_feature(self.value_feature)

        #while not self.game_queue.empty():
        #    game = self.game_queue.front()
        #    if game != NULL:
        #        free_game(game)
        #    self.game_queue.pop()

        if self.vn_session:
            self.vn_session.close()

    def clear(self):
        # time.sleep(3.)

        openmp.omp_set_lock(&self.policy_queue_lock)
        while not self.policy_network_queue.empty():
            self.policy_network_queue.pop()
        openmp.omp_unset_lock(&self.policy_queue_lock)

        openmp.omp_set_lock(&self.value_queue_lock)
        while not self.value_network_queue.empty():
            self.value_network_queue.pop()
        openmp.omp_unset_lock(&self.value_queue_lock)

        self.player_color = 0

        self.initialize_nodes()

        self.current_root = uct_hash_size
        self.pondered = False
        self.pondering = True
        self.pondering_stopped = False
        self.pondering_suspending = False
        self.pondering_suspended = True
        self.winning_ratio = 0.5
        self.time_left = self.main_time
        self.can_extend = False
        self.n_playout = 0
        self.max_queue_size_P = 0
        self.max_queue_size_V = 0

        initialize_feature(self.policy_feature) 
        initialize_feature(self.value_feature) 

        openmp.omp_init_lock(&self.tree_lock)
        openmp.omp_init_lock(&self.expand_lock)
        openmp.omp_init_lock(&self.policy_queue_lock)
        openmp.omp_init_lock(&self.value_queue_lock)

    def initialize_nodes(self):
        cdef int i
        cdef tree_node_t *node

        for i in range(uct_hash_size):
            node = &self.nodes[i]
            node.node_i = 0
            node.time_step = 0 
            node.pos = 0
            node.color = 0
            node.player_color = 0
            node.P = .0
            node.Nv = .0
            node.Wv = .0
            node.Nr = .0
            node.Wr = .0
            node.Q = .0
            node.is_root = False
            node.is_edge = False
            node.parent = NULL
            node.num_child = 0
            if node.game != NULL:
                free_game(node.game)
            node.game = NULL
            node.has_game = False

            openmp.omp_init_lock(&node.lock)

    cdef int genmove(self, game_state_t *game) nogil:
        cdef tree_node_t *node
        cdef tree_node_t *child
        cdef tree_node_t *max_child
        cdef double max_Nr
        cdef tree_node_t *second_child
        cdef double second_Nr
        cdef double max_P
        cdef int max_pos
        cdef int i, j
        cdef game_state_t *rollout_game
        cdef int winner
        cdef double rollout_wins = 0
        cdef timeval end_time

        self.seek_root(game)

        node = &self.nodes[self.current_root]

        if self.pondered:
            print_PN(node)
            if self.use_vn:
                print_VN(node)
            print_winning_ratio(node)
            print_rollout_count(node)

            if node.Nr >= 500.0 and 1.0-node.Wr/node.Nr < RESIGN_THRESHOLD:
                return RESIGN

        self.pondered = False

        if self.intuition:
            for i in range(node.num_child):
                max_P = .0
                max_pos = PASS
                for j in range(node.num_child):
                    child = node.children[node.children_pos[j]]
                    if child.P > max_P:
                        max_child = child
                        max_P = child.P
                        max_pos = child.pos

                if is_legal_not_eye(game, max_pos, game.current_color):
                    return max_pos
                else:
                    printf('>> Candidated pos is not legal. p=%d c=%d\n', max_pos, game.current_color)
                    max_child.P = .0
        else:
            for i in range(node.num_child):
                max_Nr = .0
                max_child = NULL
                max_pos = PASS
                second_Nr = .0
                second_child = NULL
                for j in range(node.num_child):
                    child = node.children[node.children_pos[j]]
                    if child.Nr > max_Nr:
                        second_child = max_child
                        second_Nr = max_Nr
                        max_child = child
                        max_Nr = child.Nr
                        max_pos = child.pos
                    elif child.Nr > second_Nr:
                        second_child = child
                        second_Nr = child.Nr

                printf('>> Max visit: %d > 2nd visit: %d\n', <int>max_Nr, <int>second_Nr),
                # extend thinking time when 1st move rollout count less than 2nd rollout count.
                if self.can_extend and max_Nr <= second_Nr*1.5:
                    printf('>> Extend pondering. %5.2lf (1st node) <= %5.2lf (2nd node*1.5)\n',
                        max_Nr, second_Nr*1.5)
                    gettimeofday(&end_time, NULL)
                    elapsed = ((end_time.tv_sec - self.search_start_time.tv_sec) +
                               (end_time.tv_usec - self.search_start_time.tv_usec) / 1000000.0)
                    self.time_left = DMAX(self.time_left - elapsed, 0.0)
                    printf('Time left: %3.2lf sec\n', self.time_left)
                    gettimeofday(&self.search_start_time, NULL)

                    self.ponder(game, True)

                    max_pos = self.genmove(game)

                if is_legal_not_eye(game, max_pos, game.current_color):
                    return max_pos
                else:
                    printf('>> Candidated pos is not legal. p=%d c=%d\n', max_pos, game.current_color)
                    max_child.Nr = .0
    
        return PASS

    cdef void start_pondering(self) nogil:
        printf('Starting pondering main thread ...\n')

        gettimeofday(&self.search_start_time, NULL)
        
        self.pondering = True
        while self.pondering:
            if self.pondering_suspended == True:
                with gil:
                    time.sleep(.001)
                continue

            self.ponder(self.game, False)

            self.pondering_suspending = False
            self.pondering_suspended = True

        self.pondering_stopped = True

    cdef void stop_pondering(self) nogil:
        printf('>> Stopping pondering ...\n')
        if not self.pondering_stopped:
            self.pondering = False
            while not self.pondering_stopped:
                with gil:
                    time.sleep(.001)
                continue
        printf('>> Pondering stopped.\n')

    cdef void suspend_pondering(self) nogil:
        printf('>> Suspending pondering ...\n')
        if not self.pondering_suspended:
            self.pondering_suspending = True
            while not self.pondering_suspended:
                with gil:
                    time.sleep(.001)
                continue
        printf('>> Pondering suspended.\n')

    cdef void resume_pondering(self) nogil:
        printf('>> Resume pondering.\n')
        self.pondering_suspended = False

    cdef void ponder(self, game_state_t *game, bint extend) nogil:
        cdef char *stone = ['#', 'B', 'W']
        cdef int playout_limit = PLAYOUT_LIMIT
        cdef double thinking_time = THINKING_TIME_LIMIT
        cdef int i
        cdef tree_node_t *node
        cdef bint expanded
        cdef int n_threads
        cdef timeval end_time
        cdef double elapsed
        cdef bint const_playout = False

        self.can_extend = False

        self.n_playout = 0
        for i in range(self.n_threads):
            self.n_threads_playout[i] = 0

        delete_old_hash(game)

        self.seek_root(game)

        node = &self.nodes[self.current_root]

        printf(">> Root Node (%s)\n", cppstring(1, stone[node.player_color]).c_str())
        if node.Nr != 0.0:
            printf('Playouts           : %d\n', <int>node.Nr)
            printf('Winning Ratio (RO) : %3.2lf %\n', 100.0-(node.Wr*100.0/node.Nr))
            if self.use_vn:
                printf('Winning Ratio (VN) : %3.2lf %\n', 100.0-(node.Wv*100.0))
        else:
            printf('>> No playout information found for current node.\n')

        if node.is_edge:
            expanded = self.expand(node, game)
            if expanded:
                with gil:
                    self.eval_leaf_by_policy_network(node)
        else:
            print_rollout_count(node)

        if self.intuition and \
           (game.moves < 200 or (game.moves + node.player_color) % 10 != 1):
            printf(">> Skip node evaluation.\n")
            return

        # determine thinking time
        if self.self_play or node.player_color == self.player_color:
            printf("\n>> Starting pondering ... ... ... :-)\n")
            # no time settings (const playout or const time)
            if self.main_time == 0.0 and self.byoyomi_time == 0.0:
                if self.const_playout > 0:
                    playout_limit = self.const_playout
                    const_playout = True
                else:
                    thinking_time = self.const_time
            # sudden death and no time left
            elif self.byoyomi_time == 0.0 and self.time_left < 60.0:
                thinking_time = 0.0
            # enough time left
            else:
                # no main time setting
                if self.main_time == 0.0:
                    thinking_time = DMAX(self.byoyomi_time, 0.1)
                else:
                    if self.time_left < self.byoyomi_time * 2.0:
                        # take 1sec margin
                        thinking_time = DMAX(self.byoyomi_time - 1.0, 1.0)
                    else:
                        thinking_time = DMAX(
                            self.time_left/(60.0 + DMAX(60.0 - game.moves, 0.0)),
                            self.byoyomi_time * (1.5 - DMAX(50.0 - game.moves, 0.0)/100.0)
                        )

                    # check if extend thnking_time
                    if not extend:
                        self.can_extend = game.moves > 4 and (self.time_left - thinking_time > self.main_time * 0.15)

                if game.moves < 4:
                    thinking_time = DMIN(thinking_time, 5.0)

                if self.winning_ratio > 0.95:
                    thinking_time = DMIN(thinking_time, 1.0)

            if thinking_time > 0.0:
                if const_playout:
                    printf('Number of simulations: %d\n', playout_limit)
                else:
                    printf('Pondering time: %3.2lf sec\n', thinking_time)
        else:
            printf("\n>> Starting read-ahead pondering ... ... ... :-)\n")

        if thinking_time > 0.0:
            n_threads = self.n_threads + (2 if self.use_vn else 1)
            for i in prange(n_threads, nogil=True):
                if i == n_threads - 1:
                    self.start_policy_network_queue()
                elif self.use_vn and i == n_threads - 2:
                    self.start_value_network_queue()
                else:
                    self.run_search(i, game, thinking_time, playout_limit) 
            
            self.winning_ratio = 1.0-(node.Wr/node.Nr)

            gettimeofday(&end_time, NULL)

            elapsed = ((end_time.tv_sec - self.search_start_time.tv_sec) +
                       (end_time.tv_usec - self.search_start_time.tv_usec) / 1000000.0)

            if node.player_color == self.player_color:
                printf(">> Time is over :-)\n")
            else:
                printf(">> Stopped read-ahead pondering.\n")

            printf('Elapsed             : %2.3lf sec\n', elapsed)
            printf('Playouts            : %d\n', self.n_playout)
            for i in range(self.n_threads):
                printf('  T%d                : %d\n', i, self.n_threads_playout[i])
            if elapsed != 0.0:
                printf('Playout Speed       : %d PO/sec\n', <int>(self.n_playout/elapsed))
            printf('Total Playouts      : %d\n', <int>node.Nr)
            printf("Winning Ratio (RO)  : %3.2lf %\n", self.winning_ratio*100.0)
            if self.use_vn:
                printf("Winning Ratio (VN)  : %3.2lf %\n", node.Wv*100.0)
            printf('Queue size (PN)     : %d\n', self.policy_network_queue.size())
            printf('Queue size max (PN) : %d\n', self.max_queue_size_P)
            printf('Queue size (VN)     : %d\n', self.value_network_queue.size())
            printf('Queue size max (VN) : %d\n', self.max_queue_size_V)
            #printf('Queue size (Game)   : %d\n', self.game_queue.size())
            printf('Hash status of use  : %3.2lf % (%u/%u)\n', used*100.0/uct_hash_size, used, uct_hash_size)

            self.pondered = True
        else:
            gettimeofday(&end_time, NULL)

            elapsed = ((end_time.tv_sec - self.search_start_time.tv_sec) +
                       (end_time.tv_usec - self.search_start_time.tv_usec) / 1000000.0)

            printf(">> No time left (%3.2lfsec) :-)\n", self.time_left)
            printf('Elapsed: %2.3lf sec\n', elapsed)

    cdef void run_search(self,
                         int thread_id,
                         game_state_t *game,
                         double thinking_time,
                         int playout_limit) nogil:
        cdef game_state_t *search_game
        cdef tree_node_t *node
        cdef int pos
        cdef unsigned long long previous_hash = 0
        cdef timeval current_time
        cdef double elapsed = .0

        printf('>> Search thread (T%d) started ...\n', thread_id)

        search_game = allocate_game()

        while (self.pondering == True and
               self.pondering_suspending == False and
               check_remaining_hash_size() and
               self.n_playout <= playout_limit and
               elapsed < thinking_time):

            if game.moves == 0 or previous_hash != game.current_hash:
                previous_hash = game.current_hash
                self.seek_root(game)
                node = &self.nodes[self.current_root]

            copy_game(search_game, game)

            self.search(node, search_game)

            self.n_threads_playout[thread_id] += 1
            self.n_playout += 1

            gettimeofday(&current_time, NULL)

            elapsed = ((current_time.tv_sec - self.search_start_time.tv_sec) + 
                       (current_time.tv_usec - self.search_start_time.tv_usec) / 1000000.0)

        self.policy_queue_running = False
        self.value_queue_running = False

        free_game(search_game)

        printf('>> T%d shut down.\n', thread_id)

    cdef void search(self,
                     tree_node_t *node,
                     game_state_t *search_game) nogil:
        cdef tree_node_t *current_node
        cdef bint expanded
        cdef int queue_size
        cdef int winner

        current_node = node

        openmp.omp_set_lock(&node.lock)
        
        # selection
        while True:
            current_node.Nr += VIRTUAL_LOSS
            if current_node.is_edge:
                break
            else:
                current_node = self.select(current_node, search_game)

        # expansion
        if current_node.Nr >= EXPANSION_THRESHOLD:
            openmp.omp_set_lock(&self.expand_lock)
            expanded = self.expand(current_node, search_game)
            openmp.omp_unset_lock(&self.expand_lock)
            if expanded:
                current_node = self.select(current_node, search_game)
                current_node.Nr += VIRTUAL_LOSS

        openmp.omp_unset_lock(&node.lock)

        # VN evaluation
        if self.use_vn and current_node.Nv == 0.0:
            # openmp.omp_set_lock(&self.game_queue_lock)
            # current_node.game = self.game_queue.front()
            current_node.game = allocate_game()
            copy_game(current_node.game, search_game)
            current_node.has_game = True
            self.value_network_queue.push(current_node)
            queue_size = self.value_network_queue.size()
            if queue_size > self.max_queue_size_V:
                self.max_queue_size_V = queue_size
            # self.game_queue.pop()
            # openmp.omp_unset_lock(&self.game_queue_lock)

        # Rollout evaluation
        if self.use_rollout:
            winner = self.rollout(search_game)
            self.backup(current_node, winner)

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
        cdef double move_probs[361]
        cdef int queue_size
        cdef int i

        # other thread may expand
        if node.num_child > 0:
            return True

        if self.use_tree:
            update_tree_planes_all(game)
            get_tree_probs(game, move_probs)
        else:
            get_rollout_probs(game, move_probs)

        if check_seki_flag:
            check_seki(game, game.seki)

        for i in range(PURE_BOARD_MAX):
            child_pos = onboard_pos[i]
            if is_legal_not_eye(game, child_pos, color) and game.seki[child_pos] == 0:
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
            node.has_game = True
            self.policy_network_queue.push(node)
            queue_size = self.policy_network_queue.size()
            if queue_size > self.max_queue_size_P:
                self.max_queue_size_P = queue_size
            return True
        else:
            return False

    cdef int rollout(self, game_state_t *game) nogil:
        cdef int winner, looser
        cdef double score
        cdef int color, other_color
        cdef int pos
        cdef int pass_count = 0
        cdef int moves_remain = MAX_MOVES - game.moves
        cdef lgr2_seed_t lgr2_seed
        cdef cppvector[lgr2_seed_t] lgr2_rollout[3]
        cdef cppvector[lgr2_seed_t].iterator it
        #cdef bint debug = False
        #if game.moves in (0,1,2,3,4,5,6,7,8,9,10) and self.n_playout == 100:
        #    debug = True

        color = game.current_color
        other_color = FLIP_COLOR(game.current_color)
        while moves_remain and pass_count < 2:
            pos = PASS
            if use_lgrf2_flag and game.moves > 1:
                lgr2_seed.prev_pos = game.record[game.moves-1].pos
                lgr2_seed.prev2_pos = game.record[game.moves-2].pos
                pos = self.lgr2[color][lgr2_seed.prev_pos][lgr2_seed.prev2_pos]

            if pos == PASS or is_legal_not_eye(game, pos, color) == False:
                while True:
                    pos = choice_rollout_move(game)
                    if is_legal_not_eye(game, pos, color):
                        break
                    else:
                        set_illegal(game, pos)

            put_stone(game, pos, color)
            game.current_color = other_color

            update_rollout(game)

            #if debug:
            #    print_board(game)

            if use_lgrf2_flag and game.moves > 2:
                lgr2_seed.pos = pos
                if (lgr2_seed.prev_pos != PASS and
                    lgr2_seed.prev2_pos != PASS and
                    lgr2_seed.pos != PASS):
                    lgr2_rollout[color].push_back(lgr2_seed)

            other_color = color 
            color = game.current_color

            pass_count = pass_count + 1 if pos == PASS else 0
            moves_remain -= 1

        score = <double>calculate_score(game)

        if score - komi > 0:
            winner = <int>S_BLACK
            looser = <int>S_WHITE
        else:
            winner = <int>S_WHITE
            looser = <int>S_BLACK

        if use_lgrf2_flag:
            it = lgr2_rollout[winner].begin()
            while it != lgr2_rollout[winner].end():
                lgr2_seed = deref(it)
                self.lgr2[winner][lgr2_seed.prev_pos][lgr2_seed.prev2_pos] = lgr2_seed.pos
                inc(it)

            it = lgr2_rollout[looser].begin()
            while it != lgr2_rollout[looser].end():
                lgr2_seed = deref(it)
                self.lgr2[looser][lgr2_seed.prev_pos][lgr2_seed.prev2_pos] = PASS
                inc(it)

        return winner

    cdef void backup(self, tree_node_t *edge_node, int winner) nogil:
        cdef tree_node_t *node

        node = edge_node
        while True:
            openmp.omp_set_lock(&node.lock)

            node.Nr += (1 - VIRTUAL_LOSS)

            if node.color == winner:
                node.Wr += 1

            if node.is_root:
                openmp.omp_unset_lock(&node.lock)
                break
            else:
                if self.use_vn is False or node.Nv == 0.0:
                    node.Q = node.Wr/node.Nr
                else:
                    node.Q = (1-MIXING_PARAMETER)*node.Wv + MIXING_PARAMETER*node.Wr/node.Nr
                openmp.omp_unset_lock(&node.lock)
                node = node.parent

    cdef void start_policy_network_queue(self) nogil:
        cdef tree_node_t *node
        cdef int i, pos

        if self.policy_queue_running:
            printf('>> Policy network queue already running.\n')
            return

        self.policy_queue_running = True

        printf('>> Starting policy network queue ...\n')

        while self.policy_queue_running:
            openmp.omp_set_lock(&self.policy_queue_lock)

            if self.policy_network_queue.empty():
                openmp.omp_unset_lock(&self.policy_queue_lock)
                continue

            node = self.policy_network_queue.front()

            if node.game != NULL:
                with gil:
                    self.eval_leaf_by_policy_network(node)

                free_game(node.game)
                node.game = NULL
                node.has_game = False

            self.policy_network_queue.pop()

            openmp.omp_unset_lock(&self.policy_queue_lock)

        printf('>> Policy network queue shut down.\n')

    cdef void stop_policy_network_queue(self) nogil:
        printf('>> Shutting down policy network queue ...\n')
        self.policy_queue_running = False

    cdef void clear_policy_network_queue(self) nogil:
        cdef tree_node_t *node

        openmp.omp_set_lock(&self.policy_queue_lock)
        while not self.policy_network_queue.empty():
            node = self.policy_network_queue.front()
            if node.game != NULL:
                free_game(node.game)
                node.game = NULL
                node.has_game = False
            self.policy_network_queue.pop()
        printf('>> Policy Network Queue cleared: %d\n', self.policy_network_queue.size())
        openmp.omp_unset_lock(&self.policy_queue_lock)

    cdef void eval_all_leaf_by_policy_network(self) nogil:
        cdef tree_node_t *node
        cdef int i, pos
        cdef int n_eval = 0

        openmp.omp_set_lock(&self.policy_queue_lock)

        while not self.policy_network_queue.empty():
            node = self.policy_network_queue.front()

            if node.game != NULL:
                with gil:
                    self.eval_leaf_by_policy_network(node)

                free_game(node.game)
                node.game = NULL
                node.has_game = False

            self.policy_network_queue.pop()

            n_eval += 1

        openmp.omp_unset_lock(&self.policy_queue_lock)

        printf('>> Policy Network Queue evaluated: %d node\n', n_eval)

    cdef void eval_leaf_by_policy_network(self, tree_node_t *node):
        cdef int i, pos
        cdef tree_node_t *child

        update(self.policy_feature, node.game)

        tensor = np.asarray(self.policy_feature.planes)
        tensor = tensor.reshape((1, MAX_POLICY_PLANES, PURE_BOARD_SIZE, PURE_BOARD_SIZE))

        probs = self.pn.eval_state(tensor)
        probs = self.apply_temperature(probs)
        for i in range(node.num_child):
            pos = node.children_pos[i]
            child = node.children[pos]
            child.P = probs[pos]

    cdef void start_value_network_queue(self) nogil:
        cdef tree_node_t *node
        cdef int i, pos

        if self.value_queue_running:
            printf('>> Value network queue already running.\n')
            return

        self.value_queue_running = True

        printf('>> Starting value network queue ...\n')

        while self.value_queue_running:
            openmp.omp_set_lock(&self.value_queue_lock)

            if self.value_network_queue.empty():
                openmp.omp_unset_lock(&self.value_queue_lock)
                continue

            node = self.value_network_queue.front()

            if node.game != NULL:
                with gil:
                    self.eval_leaf_by_value_network(node)

                free_game(node.game)
                node.game = NULL
                node.has_game = False

            self.value_network_queue.pop()

            openmp.omp_unset_lock(&self.value_queue_lock)

        printf('>> Value network queue shut down.\n')

    cdef void stop_value_network_queue(self) nogil:
        printf('>> Shutting down value network queue ...\n')
        self.value_queue_running = False

    cdef void eval_leaf_by_value_network(self, tree_node_t *node):
        cdef double vn_out

        update(self.value_feature, node.game)

        tensor = np.asarray(self.value_feature.planes)
        tensor = tensor.reshape((1, MAX_VALUE_PLANES, PURE_BOARD_SIZE, PURE_BOARD_SIZE))

        vn_out = self.vn_session.run(self.vn_op, feed_dict={self.vn_inputs: tensor})
        # rescale [-1.0, 1.0] to [0.0, 1.0]
        node.Wv = (vn_out + 1.0) / 2.0
        node.Nv += 1.0
        if node.Nr == 0.0:
            node.Q = node.Wv
        else:
            node.Q = (1-MIXING_PARAMETER)*node.Wv + MIXING_PARAMETER*node.Wr/node.Nr

        # return game to queue.
        # openmp.omp_set_lock(&self.game_queue_lock)
        # self.game_queue.push(node.game)
        # openmp.omp_unset_lock(&self.game_queue_lock)

        # print_board(node.game)
        # printf(">> VN: %3.2lf >> Q: %3.2lf\n", node.Wv, node.Q)

    cdef void clear_value_network_queue(self) nogil:
        cdef tree_node_t *node

        openmp.omp_set_lock(&self.value_queue_lock)
        while not self.value_network_queue.empty():
            node = self.value_network_queue.front()
            if node.game != NULL:
                free_game(node.game)
                node.game = NULL
                node.has_game = False
            self.value_network_queue.pop()
        printf('>> Value Network Queue cleared: %d\n', self.value_network_queue.size())
        openmp.omp_unset_lock(&self.value_queue_lock)

    def apply_temperature(self, distribution):
        log_probabilities = np.log(distribution)
        log_probabilities = log_probabilities * self.beta
        log_probabilities = log_probabilities - log_probabilities.max()
        probabilities = np.exp(log_probabilities)
        return probabilities / probabilities.sum()

    def run_pn_session(self, policy_net, temperature=0.67):
        printf('>> Set PN Session\n')
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.2
        set_session(tf.Session(config=config))
        pn = CNNPolicy(init_network=True)
        pn.model.load_weights(policy_net)
        self.pn = pn
        self.beta = 1.0/temperature

    def run_vn_session(self, value_net):
        printf('>> Set VN Session\n')
        with tf.Graph().as_default() as graph:
            self.vn_inputs = tf.placeholder(tf.float32,
                    [None, MAX_VALUE_PLANES, PURE_BOARD_SIZE, PURE_BOARD_SIZE])
            # TODO return -1 if is_training is False
            self.vn_op = inference_agz(self.vn_inputs, is_training=True)
            saver = tf.train.Saver()
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.3
            self.vn_session = tf.Session(config=config, graph=graph)
            saver.restore(self.vn_session, value_net)

        self.use_vn = True

    def set_rollout_parameter(self, rollout):
        printf('>> Set Rollout weights\n')
        set_rollout_parameter(rollout)
        self.use_rollout = True

    def set_tree_parameter(self, tree):
        printf('>> Set Tree weights\n')
        set_tree_parameter(tree)
        self.use_tree = True


cdef class PyMCTS(object):

    def __cinit__(self,
                  double const_time=5.0,
                  int const_playout=0,
                  int n_threads=1,
                  bint intuition=False,
                  bint read_ahead=False,
                  bint self_play=False):
        self.mcts = MCTS(const_time=const_time,
                         const_playout=const_playout,
                         n_threads=n_threads,
                         intuition=intuition,
                         self_play=self_play)
        self.game = allocate_game()
        self.const_time = const_time
        self.const_playout = const_playout
        self.read_ahead = read_ahead

        initialize_board(self.game)
        initialize_rollout(self.game)
        initialize_uct_hash()

    def __dealloc__(self):
        free_game(self.game)

    def run_pn_session(self, policy_net, temperature=0.67):
        self.mcts.run_pn_session(policy_net, temperature)

    def run_vn_session(self, value_net):
        self.mcts.run_vn_session(value_net)

    def set_rollout_parameter(self, rollout):
        self.mcts.set_rollout_parameter(rollout)

    def set_tree_parameter(self, tree):
        self.mcts.set_tree_parameter(tree)

    def clear(self):
        printf('>> Initializing Tree and Board ...\n')
        if self.read_ahead:
            self.suspend_pondering()
        self.mcts.clear()
        free_game(self.game)
        self.game = allocate_game()
        initialize_board(self.game)
        initialize_rollout(self.game)
        initialize_uct_hash()
        printf('>> O.K.\n')

    def start_pondering(self):
        self.mcts.start_pondering()

    def stop_pondering(self):
        self.mcts.stop_pondering()

    def suspend_pondering(self):
        self.mcts.suspend_pondering()
    
    def resume_pondering(self):
        self.mcts.game = self.game
        self.mcts.const_time = 86400
        self.mcts.const_playout = 1000000
        self.mcts.resume_pondering()

    def genmove(self, color):
        cdef timeval end_time
        cdef double elapsed

        gettimeofday(&self.mcts.search_start_time, NULL)

        # autodetect player color on gtp game.
        if self.mcts.self_play is False and self.mcts.player_color == 0:
             self.mcts.player_color = color

        self.mcts.const_time = self.const_time
        self.mcts.const_playout = self.const_playout
        self.mcts.ponder(self.game, False)

        self.game.current_color = color
        pos = self.mcts.genmove(self.game)

        gettimeofday(&end_time, NULL)

        elapsed = ((end_time.tv_sec - self.mcts.search_start_time.tv_sec) +
                   (end_time.tv_usec - self.mcts.search_start_time.tv_usec) / 1000000.0)

        printf('>> Genmove Elapsed: %2.3lf sec\n', elapsed)

        if self.mcts.main_time > 0.0:
            self.mcts.time_left = DMAX(self.mcts.time_left - elapsed, 0.0)
            printf('Time left: %3.2lf sec\n', self.mcts.time_left)

        return pos

    def play(self, pos, color):
        cdef bint legal

        # autodetect player color on gtp game.
        if self.mcts.self_play is False and self.mcts.player_color == 0:
             self.mcts.player_color = FLIP_COLOR(color)

        if self.read_ahead:
            self.suspend_pondering()

        legal = put_stone(self.game, pos, color)

        if legal:
            self.game.current_color = FLIP_COLOR(self.game.current_color)
            update_rollout(self.game)
            print_board(self.game)

            if self.read_ahead and self.mcts.player_color == color:
                self.resume_pondering()

            return True
        else:
            return False

    def start_policy_network_queue(self):
        self.mcts.start_policy_network_queue()

    def stop_policy_network_queue(self):
        self.mcts.stop_policy_network_queue()

    def eval_all_leaf_by_policy_network(self):
        self.mcts.eval_all_leaf_by_policy_network()

    def start_value_network_queue(self):
        self.mcts.start_value_network_queue()

    def stop_value_network_queue(self):
        self.mcts.stop_value_network_queue()

    def set_size(self, bsize):
        self.clear()
        set_board_size(bsize)

    def set_komi(self, new_komi):
        set_komi(new_komi)

    def set_time_settings(self, main_time, byoyomi_time, byoyomi_stones):
        self.mcts.main_time = main_time
        self.mcts.time_left = self.mcts.main_time

        # const time for each stone.
        if byoyomi_stones > 0:
            self.mcts.byoyomi_time = byoyomi_time/<double>byoyomi_stones
        else:
            self.mcts.byoyomi_time = byoyomi_time

    def set_time_left(self, color, time, stones):
        if self.mcts.player_color == color:
            if stones == 0:
                self.mcts.time_left = time 
            else:
                self.mcts.byoyomi_time = time/<double>stones
                self.mcts.time_left = 0.0 

    def set_const_time(self, limit):
        self.const_time = limit

    def set_const_playout(self, limit):
        self.const_playout = limit

    def showboard(self):
        print_board(self.game)

    def save_sgf(self, black_name, white_name):
        from tempfile import NamedTemporaryFile
        temp_file = NamedTemporaryFile(delete=False)
        temp_file_name = temp_file.name + '.sgf'
        save_gamestate_to_sgf(self.game, '/tmp/', temp_file_name, black_name, white_name)
        return temp_file_name
