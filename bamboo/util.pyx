import os
import itertools
import numpy as np
import re
import sgf
import sys
import traceback
import warnings

from bamboo.util_error import SizeMismatchError, IllegalMove, TooManyMove, TooFewMove

from bamboo.go.board cimport S_BLACK, S_WHITE, PASS, OB_SIZE, BOARD_MAX
from bamboo.go.board cimport POS, FLIP_COLOR
from bamboo.go.board cimport game_state_t, board_size
from bamboo.go.board cimport allocate_game, set_board_size, initialize_const, clear_const, initialize_board, free_game, put_stone
from bamboo.go.printer cimport print_board

# for board location indexing
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def _parse_sgf_move(node_value):
    """Given a well-formed move string, return either PASS_MOVE or the (x, y) position
    """
    if node_value == '' or node_value == 'tt':
        return PASS
    else:
        x = LETTERS.index(node_value[0].upper())
        y = LETTERS.index(node_value[1].upper())
        return POS(x+OB_SIZE, y+OB_SIZE, board_size)


cpdef min_sgf_extract(sgf_string):
    size = ''.join(re.findall(r'SZ\[.+?\]', sgf_string, flags=re.IGNORECASE))
    player = ''.join(re.findall(r'PL\[.+?\]', sgf_string, flags=re.IGNORECASE))
    kiryoku = ''.join(re.findall(r'[BW]R\[.+?\]', sgf_string, flags=re.IGNORECASE))
    add_stone = ''.join(re.findall(r'A[BW](?:\[[a-z]+\]\s*)+', sgf_string, flags=re.IGNORECASE))
    moves = ''.join(re.findall(r';[WB]\[[a-z]*?\]', sgf_string, flags=re.IGNORECASE))
    return '(;{:s}{:s}{:s}{:s}{:s})'.format(size, player, kiryoku, add_stone, moves)


cdef class SGFMoveIterator:

    def __cinit__(self, int bsize, object sgf_string, bint verbose=False):
        self.bsize = bsize
        self.game = allocate_game()
        self.moves = list()
        self.i = 0
        self.next_move = None
        self.verbose = verbose

        sgf_string = min_sgf_extract(sgf_string)
        try:
            collection = sgf.parse(sgf_string)
        except sgf.ParseException:
            warnings.warn('ParseException\n{:s}\n'.format(sgf_string))
            if self.verbose:
                err, msg, _ = sys.exc_info()
                sys.stderr.write("{:s} {:s}\n{:s}".format(err, msg, sgf_string))
                sys.stderr.write(traceback.format_exc())
            raise

        sgf_game = collection[0]
        if len(sgf_game.nodes) < 50:
            raise TooFewMove(len(sgf_game.nodes))
        if len(sgf_game.nodes) > 500:
            raise TooManyMove(len(sgf_game.nodes))

        self.sgf_init_game(sgf_game.root)

        if sgf_game.rest is not None:
            for node in sgf_game.rest:
                props = node.properties
                if 'W' in props:
                    pos = _parse_sgf_move(props['W'][0])
                    self.moves.append((pos, S_WHITE))
                elif 'B' in props:
                    pos = _parse_sgf_move(props['B'][0])
                    self.moves.append((pos, S_BLACK))

        self.i = 0
        self.next_move = self.moves[self.i]
        if self.next_move[0] != PASS:
            self.game.current_color = self.next_move[1]

    def __dealloc__(self):
        free_game(self.game)

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == 0:
            move = self.moves[0]
        else:
            prev_move = self.moves[self.i-1]
            put_stone(self.game, prev_move[0], prev_move[1])
            if self.i >= len(self.moves):
                raise StopIteration()
            move = self.moves[self.i]

        self.game.current_color = move[1]
        self.i += 1

        if self.i < len(self.moves):
            self.next_move = self.moves[self.i]
        else:
            self.next_move = None 

        return move

    cdef int sgf_init_game(self, object sgf_root) except? -1:
        """Helper function to set up a GameState object from the root node
        of an SGF file
        """
        props = sgf_root.properties
        s_size = props.get('SZ', ['19'])[0]
        s_player = props.get('PL', ['B'])[0]

        if self.bsize != int(s_size):
            raise SizeMismatchError()

        set_board_size(int(s_size))
        initialize_board(self.game, False)

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
