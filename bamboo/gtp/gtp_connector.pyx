# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

import pyjsonrpc 

from bamboo.gtp import gtp

from bamboo.board cimport PURE_BOARD_SIZE, BOARD_SIZE, OB_SIZE, PASS, RESIGN, S_BLACK, S_WHITE
from bamboo.board cimport POS, X, Y, CORRECT_X, CORRECT_Y
from bamboo.board cimport game_state_t
from bamboo.board cimport allocate_game, free_game, initialize_board, set_board_size
from bamboo.board cimport do_move
from bamboo.models.keras_dcnn_policy import CNNPolicy
from bamboo.tree_search import PyMCTS

from bamboo.printer cimport print_board


class MCTSConnector(object):

    def __init__(self,
                 pn_path,
                 vn_path=None,
                 rollout_path=None,
                 tree_path=None,
                 temperature=0.67,
                 const_time=5.0,
                 const_playout=0,
                 n_threads=1,
                 intuition=False):
        self.mcts = PyMCTS(
                const_time=const_time,
                const_playout=const_playout,
                n_threads=n_threads,
                intuition=intuition,
                read_ahead=False,
                self_play=False)
        if pn_path:
            self.mcts.run_pn_session(pn_path, temperature)

        if vn_path:
            self.mcts.run_vn_session(vn_path)

        if rollout_path:
            self.mcts.set_rollout_parameter(rollout_path)

        if tree_path:
            self.mcts.set_tree_parameter(tree_path)

    def clear(self):
        self.mcts.clear()

    def get_move(self, color):
        cdef int x, y, pos

        pos = self.mcts.genmove(color)

        if pos == PASS:
            return gtp.PASS
        elif pos == RESIGN:
            return gtp.RESIGN
        else:
            x = CORRECT_X(pos, BOARD_SIZE, OB_SIZE) + 1
            y = PURE_BOARD_SIZE-CORRECT_Y(pos, BOARD_SIZE, OB_SIZE)
            return (x, y)

    def make_move(self, color, vertex):
        # vertex in GTP language is 1-indexed, whereas GameState's are zero-indexed
        cdef int pos, x, y
        cdef bint is_legal

        if vertex == gtp.PASS:
            self.mcts.play(PASS, color)
            return True

        pos = POS(OB_SIZE+vertex[0]-1, OB_SIZE+PURE_BOARD_SIZE-vertex[1], BOARD_SIZE)

        if self.mcts.play(pos, color):
            return True
        else:
            return False

    def set_size(self, bsize):
        self.mcts.set_size(bsize)

    def set_komi(self, komi):
        self.mcts.set_komi(komi)

    def set_time_settings(self, main_time, byoyomi_time, byoyomi_stones):
        self.mcts.set_time_settings(main_time, byoyomi_time, byoyomi_stones)

    def set_time_left(self, color, time, stones):
        self.mcts.set_time_left(color, time, stones)

    def set_const_time(self, limit):
        self.mcts.set_const_time(limit)

    def set_const_playout(self, limit):
        self.mcts.set_const_playout(limit)

    def get_current_state_as_sgf(self):
        return self.mcts.save_sgf('Unknown', 'Unknown')

    def place_handicaps(self, vertices):
        # TODO
        pass

    def showboard(self):
        self.mcts.showboard()

    def quit(self):
        self.mcts.clear()


class RemoteMCTSConnector(object):

    def __init__(self, host='localhost', port=6000):
        self.client = pyjsonrpc.HttpClient(
            url='http://{:s}:{:d}/'.format(host, port),
            timeout=24*60*60)

    def clear(self):
        self.client.clear()

    def get_move(self, color):
        cdef int x, y, pos

        pos = self.client.genmove(color)

        if pos == PASS:
            return gtp.PASS
        elif pos == RESIGN:
            return gtp.RESIGN
        else:
            x = CORRECT_X(pos, BOARD_SIZE, OB_SIZE) + 1
            y = PURE_BOARD_SIZE-CORRECT_Y(pos, BOARD_SIZE, OB_SIZE)
            return (x, y)

    def make_move(self, color, vertex):
        # vertex in GTP language is 1-indexed, whereas GameState's are zero-indexed
        cdef int pos, x, y
        cdef bint is_legal

        if vertex == gtp.PASS:
            self.client.play(PASS, color)
            return True

        pos = POS(OB_SIZE+vertex[0]-1, OB_SIZE+PURE_BOARD_SIZE-vertex[1], BOARD_SIZE)

        if self.client.play(pos, color):
            return True
        else:
            return False

    def set_size(self, bsize):
        self.client.set_size(bsize)

    def set_komi(self, komi):
        self.client.set_komi(komi)

    def set_time(self, m, b, stone):
        self.client.set_time(m, b, stone)

    def set_time_left(self, color, time, stone):
        self.client.set_time_left(color, time, stone)

    def set_const_time(self, limit):
        self.client.set_const_time(limit)

    def set_const_playout(self, limit):
        self.client.set_const_playout(limit)

    def get_current_state_as_sgf(self):
        return self.client.save_sgf('Unknown', 'Unknown')

    def place_handicaps(self, vertices):
        # TODO
        pass

    def showboard(self):
        self.client.showboard()

    def quit(self):
        self.client.clear()
