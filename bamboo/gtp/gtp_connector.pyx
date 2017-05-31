from bamboo.gtp import gtp

from bamboo.go.board cimport PURE_BOARD_SIZE, BOARD_SIZE, OB_SIZE, PASS, S_BLACK, S_WHITE
from bamboo.go.board cimport POS, X, Y
from bamboo.go.board cimport game_state_t
from bamboo.go.board cimport allocate_game, free_game, initialize_board, set_board_size
from bamboo.go.board cimport do_move

from bamboo.ai.greedy cimport GreedyPolicyPlayer

from bamboo.go.printer cimport print_board

from bamboo.util import save_gamestate_to_sgf


cdef class GTPGameConnector(object):
    """A class implementing the functions of a 'game' object required by the GTP
    Engine by wrapping a GameState and Player instance
    """

    cdef game_state_t *game
    cdef GreedyPolicyPlayer player

    def __cinit__(self, GreedyPolicyPlayer player):
        self.game = allocate_game()
        self.set_size(19)
        self.player = player

    def __dealloc(self):
        free_game(self.game)

    def clear(self):
        free_game(self.game)
        self.game = allocate_game()
        self.set_size(19)

    def get_move(self, color):
        cdef int x, y, pos
        self.game.current_color = color
        pos = self.player.get_move(self.game)
        if pos == PASS:
            return gtp.PASS
        else:
            x = X(pos, PURE_BOARD_SIZE) + 1
            y = PURE_BOARD_SIZE-Y(pos, PURE_BOARD_SIZE)
            print pos, x, y
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
        if is_legal:
            return True
        else:
            return False

    def set_size(self, n):
        set_board_size(n)
        initialize_board(self.game, False)

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
