import os
import itertools
import numpy as np
import re
import sgf

from bamboo.go.board cimport S_BLACK, S_WHITE, PASS, OB_SIZE, BOARD_MAX
from bamboo.go.board cimport POS
from bamboo.go.board cimport game_state_t, board_size
from bamboo.go.board cimport allocate_game, set_board_size, initialize_const, clear_const, initialize_board, free_game, put_stone

# for board location indexing
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


class SizeMismatchError(Exception):
    pass


def _parse_sgf_move(node_value):
    """Given a well-formed move string, return either PASS_MOVE or the (x, y) position
    """
    if node_value == '' or node_value == 'tt':
        return PASS
    else:
        x = LETTERS.index(node_value[0].upper())
        y = LETTERS.index(node_value[1].upper())
        return POS(x+OB_SIZE, y+OB_SIZE, board_size)


cdef class SGFIterator(object):

    def __cinit__(self, int bsize):
        self.bsize = bsize
        self.game = allocate_game()
        self.moves = list()
        self.current_moves = 0
        self.next_move = 0

        set_board_size(bsize)
        initialize_board(self.game, False)

    def __dealloc__(self):
        free_game(self.game)

    cdef int read(self, object sgf_string) except? -1:
        sgf_string = re.sub(r'\s', '', sgf_string)
        if sgf_string[0] == '(' and sgf_string[1] != ';':
            sgf_string = sgf_string[:1] + ';' + sgf_string[1:]
        collection = sgf.parse(sgf_string)
        game = collection[0]

        self.sgf_init_gamestate(game.root)

        if game.rest is not None:
            for node in game.rest:
                props = node.properties
                if 'W' in props:
                    pos = _parse_sgf_move(props['W'][0])
                    self.moves.append((pos, S_WHITE))
                elif 'B' in props:
                    pos = _parse_sgf_move(props['B'][0])
                    self.moves.append((pos, S_BLACK))

        self.current_moves = 0
        self.next_move = self.moves[self.current_moves][0]

    cdef bint has_next(self):
        return self.current_moves < len(self.moves)

    cdef game_state_t *move_next(self):
        cdef int pos = self.moves[self.current_moves][0]
        cdef char color = self.moves[self.current_moves][1]
        put_stone(self.game, pos, color)
        self.current_moves += 1
        if self.has_next():
            self.next_move = self.moves[self.current_moves][0]
        else:
            self.next_move = BOARD_MAX
        return self.game

    cdef int sgf_init_gamestate(self, object sgf_root) except? -1:
        """Helper function to set up a GameState object from the root node
        of an SGF file
        """
        props = sgf_root.properties
        s_size = props.get('SZ', ['19'])[0]
        s_player = props.get('PL', ['B'])[0]

        if int(s_size) != self.bsize:
            raise SizeMismatchError()

        # handle 'add black' property
        if 'AB' in props:
            for stone in props['AB']:
                put_stone(self.game, _parse_sgf_move(stone), S_BLACK)
        # handle 'add white' property
        if 'AW' in props:
            for stone in props['AW']:
                put_stone(self.game, _parse_sgf_move(stone), S_WHITE)
        # setup done; set player according to 'PL' property
        self.game.current_color = S_BLACK if s_player == 'B' else S_WHITE


def save_gamestate_to_sgf(gamestate, path, filename, black_player_name='Unknown',
                          white_player_name='Unknown', size=19, komi=7.5):
    """Creates a simplified sgf for viewing playouts or positions
    """
    str_list = []
    # Game info
    str_list.append('(;GM[1]FF[4]CA[UTF-8]')
    str_list.append('SZ[{}]'.format(size))
    str_list.append('KM[{}]'.format(komi))
    str_list.append('PB[{}]'.format(black_player_name))
    str_list.append('PW[{}]'.format(white_player_name))
    cycle_string = 'BW'
    # Handle handicaps
    if len(gamestate.handicaps) > 0:
        cycle_string = 'WB'
        str_list.append('HA[{}]'.format(len(gamestate.handicaps)))
        str_list.append(';AB')
        for handicap in gamestate.handicaps:
            str_list.append('[{}{}]'.format(LETTERS[handicap[0]].lower(),
                                            LETTERS[handicap[1]].lower()))
    # Move list
    for move, color in zip(gamestate.history, itertools.cycle(cycle_string)):
        # Move color prefix
        str_list.append(';{}'.format(color))
        # Move coordinates
        if move is None:
            str_list.append('[tt]')
        else:
            str_list.append('[{}{}]'.format(LETTERS[move[0]].lower(), LETTERS[move[1]].lower()))
    str_list.append(')')
    with open(os.path.join(path, filename), "w") as f:
        f.write(''.join(str_list))


def confirm(prompt=None, resp=False):
    """prompts for yes or no response from the user. Returns True for yes and
       False for no.
       'resp' should be set to the default value assumed by the caller when
       user simply types ENTER.
       created by:
       http://code.activestate.com/recipes/541096-prompt-the-user-for-confirmation/
    """

    if prompt is None:
        prompt = 'Confirm'

    if resp:
        prompt = '%s [%s]|%s: ' % (prompt, 'y', 'n')
    else:
        prompt = '%s [%s]|%s: ' % (prompt, 'n', 'y')

    while True:
        ans = raw_input(prompt)
        if not ans:
            return resp
        if ans not in ['y', 'Y', 'n', 'N']:
            print 'please enter y or n.'
            continue
        if ans == 'y' or ans == 'Y':
            return True
        if ans == 'n' or ans == 'N':
            return False
