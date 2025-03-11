import os
import itertools
import numpy as np
import re
import sgf
import sys
import traceback
import unicodedata

from tqdm import tqdm

from bamboo.sgf_error import SizeMismatchError, IllegalMove, TooManyMove, TooFewMove, NoResultError

from libcpp.string cimport string as cppstring

from bamboo.board cimport S_BLACK, S_WHITE, PASS, OB_SIZE, BOARD_MAX, BOARD_SIZE, PURE_BOARD_SIZE
from bamboo.board cimport POS, FLIP_COLOR, CORRECT_X, CORRECT_Y
from bamboo.board cimport board_size, pure_board_size, komi
from bamboo.board cimport game_state_t, move_t, board_size
from bamboo.board cimport allocate_game, set_board_size, initialize_const, clear_const, initialize_board, free_game, put_stone
from bamboo.rollout_preprocess cimport initialize_rollout, update_rollout
from bamboo.printer cimport print_board

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
        pos = POS(x+OB_SIZE, y+OB_SIZE, board_size)
        if BOARD_MAX <= pos:
            return PASS
        else:
            return pos


cdef class SGFMoveIterator:

    def __cinit__(self,
                  int bsize,
                  object sgf_string,
                  int too_few_moves_threshold=50,
                  int too_many_moves_threshold=800,
                  bint rollout=False,
                  bint ignore_not_legal=True,
                  bint ignore_no_result=True,
                  bint verbose=False):
        self.bsize = bsize
        self.game = allocate_game()
        self.moves = list()
        self.i = 0
        self.next_move = None
        self.komi = 0.0
        self.winner = 0
        self.resign = False
        self.handicap_game = False
        self.too_few_moves_threshold = too_few_moves_threshold
        self.too_many_moves_threshold = too_many_moves_threshold
        self.ignore_not_legal = ignore_not_legal
        self.ignore_no_result = ignore_no_result
        self.rollout = rollout
        self.verbose = verbose

        try:
            collection = sgf.parse(sgf_string)
        except sgf.ParseException:
            tqdm.write('ParseException\n{:s}\n'.format(sgf_string), file=sys.stderr)
            if self.verbose:
                err, msg, _ = sys.exc_info()
                tqdm.write("{:s} {:s}\n{:s}".format(err, msg, sgf_string), file=sys.stderr)
                tqdm.write(traceback.format_exc(), file=sys.stderr)
            raise

        sgf_game = collection[0]

        self.sgf_init_game(sgf_game.nodes)

        if sgf_game.rest is not None:
            for i, node in enumerate(sgf_game.rest):
                if i > self.too_many_moves_threshold:
                    raise TooManyMove(i)
                props = node.properties
                if 'W' in props:
                    pos = _parse_sgf_move(props['W'][0])
                    self.moves.append((pos, S_WHITE))
                elif 'B' in props:
                    pos = _parse_sgf_move(props['B'][0])
                    self.moves.append((pos, S_BLACK))

        if len(self.moves) < self.too_few_moves_threshold:
            raise TooFewMove(i)

        self.i = 0
        self.next_move = self.moves[self.i]
        if self.next_move[0] != PASS:
            self.game.current_color = self.next_move[1]

    def __dealloc__(self):
        free_game(self.game)

    def __iter__(self):
        return self

    def __next__(self):
        cdef bint is_legal
        if self.i == 0:
            move = self.moves[0]
        else:
            prev_move = self.moves[self.i-1]
            is_legal = put_stone(self.game, prev_move[0], prev_move[1])
            if not (is_legal or self.ignore_not_legal):
                raise IllegalMove(prev_move)
            if self.i >= len(self.moves):
                raise StopIteration()
            move = self.moves[self.i]

        self.game.current_color = move[1]
        self.i += 1

        if self.i < len(self.moves):
            self.next_move = self.moves[self.i]
        else:
            self.next_move = None 

        if self.rollout:
            update_rollout(self.game)

        return move

    cdef int sgf_init_game(self, object sgf_nodes) except? -1:
        """Helper function to set up a GameState object from the root node
        of an SGF file
        """
        sgf_root = sgf_nodes[0]
        props = sgf_root.properties
        s_size = props.get('SZ', ['19'])[0]
        s_player = props.get('PL', ['B'])[0]

        if self.bsize != int(s_size):
            raise SizeMismatchError()

        set_board_size(int(s_size))
        initialize_board(self.game)
        if self.rollout:
            initialize_rollout(self.game)

        # handle 'add black' property
        if 'AB' in props:
            self.handicap_game = True
            for stone in props['AB']:
                put_stone(self.game, _parse_sgf_move(stone), S_BLACK)
        # handle 'add white' property
        if 'AW' in props:
            self.handicap_game = True
            for stone in props['AW']:
                put_stone(self.game, _parse_sgf_move(stone), S_WHITE)
        # setup done; set player according to 'PL' property
        self.game.current_color = S_BLACK if s_player == 'B' else S_WHITE

        # set komi
        s_komi = props.get('KM')
        if s_komi:
            try:
                self.komi = float(s_komi[0])
            except ValueError:
                pass

        # set winner
        self.winner = 0
        self.resign = False
        for node in sgf_nodes:
            props = node.properties
            if "RE" in props:
                s_re = props.get("RE")[0].strip().upper()
                s_re = unicodedata.normalize("NFKC", s_re)
                if any(b in s_re for b in ("B", "黒", "黑")):
                    self.winner = S_BLACK
                elif any(w in s_re for w in ("W", "白")):
                    self.winner = S_WHITE
                self.resign = s_re.endswith('+R')
                break

        if self.winner == 0 and not self.ignore_no_result:
            raise NoResultError


cdef void save_gamestate_to_sgf(game_state_t *game,
                                path,
                                filename,
                                black_player_name,
                                white_player_name):
    """Creates a simplified sgf for viewing playouts or positions
    """
    cdef char *stone = [b'+', b'B', b'W']
    cdef int i
    cdef move_t move
    cdef int pos, x, y, color
    cdef list str_list = []
    # Game info
    str_list.append('(;GM[1]FF[4]CA[UTF-8]')
    str_list.append('SZ[{}]'.format(pure_board_size))
    str_list.append('KM[{}]'.format(komi))
    str_list.append('PB[{}]'.format(black_player_name))
    str_list.append('PW[{}]'.format(white_player_name))
    cycle_string = 'BW'
    # Handle handicaps
    """
    if len(gamestate.handicaps) > 0:
        cycle_string = 'WB'
        str_list.append('HA[{}]'.format(len(gamestate.handicaps)))
        str_list.append(';AB')
        for handicap in gamestate.handicaps:
            str_list.append('[{}{}]'.format(LETTERS[handicap[0]].lower(),
                                            LETTERS[handicap[1]].lower()))
    """
    # Move list
    for i in range(game.moves):
        move = game.record[i]
        # Move color prefix
        str_list.append(';{:s}'.format(cppstring(1, stone[move.color]).decode('utf8')))
        # Move coordinates
        if move.pos == PASS:
            str_list.append('[tt]')
        else:
            x = CORRECT_X(move.pos, board_size, OB_SIZE)
            y = CORRECT_Y(move.pos, board_size, OB_SIZE)
            str_list.append('[{:s}{:s}]'.format(LETTERS[x].lower(), LETTERS[y].lower()))
    str_list.append(')')

    with open(os.path.join(path, filename), "w") as f:
        f.write(''.join(str_list))
