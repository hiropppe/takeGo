# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

import msgpackrpc

from libc.stdio cimport printf

from bamboo.go.board cimport S_BLACK 
from bamboo.go.board cimport FLIP_COLOR 
from bamboo.go.board cimport game_state_t
from bamboo.go.board cimport set_board_size, set_komi, komi, allocate_game, initialize_board, put_stone
from bamboo.go.printer cimport print_board
from bamboo.rollout.preprocess cimport initialize_rollout, update_rollout
from bamboo.mcts.tree_search cimport MCTS
from bamboo.util cimport save_gamestate_to_sgf 


cdef class MCTSServer(object):

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
        self.mcts.stop_policy_network_queue()
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
        self.mcts.stop_policy_network_queue()
        initialize_board(self.game)
        initialize_rollout(self.game)

