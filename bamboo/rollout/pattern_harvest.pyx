import os
import pandas as pd
import sgf
import sys
import traceback
import warnings

from tqdm import tqdm

from libc.stdio cimport printf

from bamboo.util_error import SizeMismatchError, IllegalMove, TooManyMove

from bamboo.util cimport SGFMoveIterator
from bamboo.go.board cimport PASS, CORRECT_X, CORRECT_Y, OB_SIZE
from bamboo.go.board cimport game_state_t, string_t, liberty_end, board_size 
from bamboo.go.board cimport allocate_game, free_game, onboard_index, set_board_size, initialize_board
from bamboo.go.printer cimport print_board
from bamboo.rollout.pattern cimport x33_bits, print_x33

from collections import Counter, defaultdict

cdef unsigned long long pat3[361]

freq_dict = defaultdict(int)
move_dict = defaultdict(int)

def harvest_pattern(file_name):    
    cdef game_state_t *game
    cdef SGFMoveIterator sgf_iter
    cdef int *updated_string_num
    cdef int *updated_string_id
    cdef int i
    cdef int string_id
    cdef string_t *string
    cdef int center
    cdef unsigned long long pat

    with open(file_name, 'r') as file_object:
        sgf_iter = SGFMoveIterator(19, file_object.read())

    game = sgf_iter.game
    for move in sgf_iter:
        if move[0] != PASS:
            updated_string_num = &game.updated_string_num[<int>game.current_color]
            updated_string_id = game.updated_string_id[<int>game.current_color]
            for i in range(updated_string_num[0]):
                string_id = updated_string_id[i]
                string = &game.string[string_id]
                center = string.lib[0]
                while center != liberty_end:
                    pat = x33_bits(game, center, <int>game.current_color)
                    if pat3[onboard_index[center]] != pat:
                        pat3[onboard_index[center]] = pat
                        freq_dict[pat] += 1
                        if sgf_iter.next_move and sgf_iter.next_move[0] == center:
                            move_dict[pat] += 1
                        else:
                            move_dict[pat] += 0 
                    center = string.lib[center]
        updated_string_num[0] = 0


def main(cmd_line_args=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--outfile", "-o", help="Destination to write data", required=True)  # noqa: E501
    parser.add_argument("--directory", "-d", help="Directory containing SGF files to process. if not present, expects files from stdin", default=None)  # noqa: E501
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    def _is_sgf(fname):
        return fname.strip()[-4:] == ".sgf"

    def _list_sgfs(path):
        files = os.listdir(path)
        return [os.path.join(path, f) for f in files if _is_sgf(f)]

    if args.directory:
        sgf_files = _list_sgfs(args.directory)
    else:
        sgf_files = (f.strip() for f in sys.stdin if _is_sgf(f))

    n_parse_error = 0
    n_runtime_error = 0
    n_not19 = 0
    for sgf_file in tqdm(sgf_files):
        try:
            harvest_pattern(sgf_file)
        except sgf.ParseException:
            n_parse_error += 1
            if args.verbose:
                err, msg, _ = sys.exc_info()
                sys.stderr.write("{} {}\n".format(err, msg))
                sys.stderr.write(traceback.format_exc())
            else:
                warnings.warn('ParseException at {:s}\n'.format(sgf_file))
        except SizeMismatchError:
            n_not19 += 1
        except IllegalMove:
            n_runtime_error += 1
            if args.verbose:
                err, msg, _ = sys.exc_info()
                sys.stderr.write("{} {}\n".format(err, msg))
                sys.stderr.write(traceback.format_exc())
            else:
                warnings.warn('Unexpected error at {:s}\n'.format(sgf_file))
        except TooManyMove:
            warnings.warn('Too many move. more than 500')

    print('Total {:d} Parse Err {:d} Other Err {:d}'.format(len(sgf_files)-n_not19,
        n_parse_error, n_runtime_error))

    df = pd.DataFrame({'freq': freq_dict, 'move_freq': move_dict})
    df['move_ratio'] = df['move_freq']/df['freq']
    sorted_df = df.sort_values(by=['move_ratio', 'freq'], ascending=False)
    print(sorted_df)
    df.to_csv(args.outfile)
