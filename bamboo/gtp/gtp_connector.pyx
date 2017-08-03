# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

import msgpackrpc

from bamboo.gtp import gtp

from bamboo.go.board cimport PURE_BOARD_SIZE, BOARD_SIZE, OB_SIZE, PASS, S_BLACK, S_WHITE
from bamboo.go.board cimport POS, X, Y, CORRECT_X, CORRECT_Y
from bamboo.go.board cimport game_state_t
from bamboo.go.board cimport allocate_game, free_game, initialize_board, set_board_size
from bamboo.go.board cimport do_move

from bamboo.ai.greedy cimport GreedyPolicyPlayer
from bamboo.ai.rollout cimport RolloutPolicyPlayer

from bamboo.go.printer cimport print_board


class MCTSConnector(object):

    def __init__(self, host='localhost', port=6000):
        self.client = msgpackrpc.Client(
                        msgpackrpc.Address(host, port),
                        timeout=60*10)

    def clear(self):
        self.client.call('clear')

    def get_move(self, color):
        cdef int x, y, pos

        self.client.call('start_pondering')
        self.client.call('eval_all_leafs_by_policy_network')

        pos = self.client.call('genmove', color)

        if pos == PASS:
            return gtp.PASS
        else:
            x = CORRECT_X(pos, BOARD_SIZE, OB_SIZE) + 1
            y = PURE_BOARD_SIZE-CORRECT_Y(pos, BOARD_SIZE, OB_SIZE)
            return (x, y)

    def make_move(self, color, vertex):
        # vertex in GTP language is 1-indexed, whereas GameState's are zero-indexed
        cdef int pos, x, y
        cdef bint is_legal

        if vertex == gtp.PASS:
            self.client.call('play', PASS, color)
            return True

        pos = POS(OB_SIZE+vertex[0]-1, OB_SIZE+PURE_BOARD_SIZE-vertex[1], BOARD_SIZE)

        if self.client.call('play', pos, color):
            return True
        else:
            return False

    def set_size(self, bsize):
        self.client.call('set_size', bsize)

    def set_komi(self, komi):
        self.client.call('set_komi', komi)

    def set_time(self, m, b, stone):
        self.client.call('set_time', m, b, stone)

    def set_time_left(self, color, time, stone):
        self.client.call('set_time_left', color, time, stone)

    def set_playout_limit(self, limit):
        self.client.call('set_playout_limit', limit)

    def get_current_state_as_sgf(self):
        return self.client.call('save_sgf', 'Unknown', 'Unknown')

    def place_handicaps(self, vertices):
        # TODO
        pass

    def showboard(self):
        self.client.call('print_board')

    def quit(self):
        self.client.call('quit')


cdef class GTPGameConnector(object):
    """A class implementing the functions of a 'game' object required by the GTP
    Engine by wrapping a GameState and Player instance
    """

    cdef game_state_t *game
    cdef GreedyPolicyPlayer greedy_player
    cdef RolloutPolicyPlayer rollout_player
    cdef object player_name

    def __cinit__(self):
        self.game = allocate_game()
        self.set_size(19)

    def __dealloc(self):
        free_game(self.game)

    def clear(self):
        free_game(self.game)
        self.game = allocate_game()
        self.set_size(19)

    def set_greedy(self, GreedyPolicyPlayer player):
        self.greedy_player = player
        self.player_name = 'greedy'

    def set_rollout(self, RolloutPolicyPlayer player):
        self.rollout_player = player
        self.player_name = 'rollout'

    def get_move(self, color):
        cdef int x, y, pos
        self.game.current_color = color

        if self.player_name == 'greedy':
            pos = self.greedy_player.get_move(self.game)
        elif self.player_name == 'rollout':
            pos = self.rollout_player.get_move(self.game)
        else:
            raise Exception()

        if pos == PASS:
            return gtp.PASS
        else:
            x = X(pos, PURE_BOARD_SIZE) + 1
            y = PURE_BOARD_SIZE-Y(pos, PURE_BOARD_SIZE)
            return (x, y)

    def make_move(self, color, vertex):
        # vertex in GTP language is 1-indexed, whereas GameState's are zero-indexed
        cdef int pos, x, y
        cdef bint is_legal

        self.game.current_color = color

        if vertex == gtp.PASS:
            do_move(self.game, PASS)
            return True

        (x, y) = vertex
        pos = POS(OB_SIZE+x-1, OB_SIZE+PURE_BOARD_SIZE-y, BOARD_SIZE)
        is_legal = do_move(self.game, pos)
        print_board(self.game)
        if is_legal:
            return True
        else:
            return False

    def set_size(self, n):
        set_board_size(n)
        initialize_board(self.game)

    def set_komi(self, k):
        pass

    def set_time(self, m, b, stone):
        pass

    def set_time_left(self, color, time, stone):
        pass

    def get_current_state_as_sgf(self):
        pass

    def place_handicaps(self, vertices):
        pass

    def showboard(self):
        print_board(self.game)

    def quit(self):
        pass
