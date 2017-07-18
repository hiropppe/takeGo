import os
import pandas as pd
import sgf
import sys
import traceback
import warnings

from tqdm import tqdm
from collections import defaultdict

from libc.stdio cimport printf

from bamboo.util_error import SizeMismatchError, IllegalMove, TooManyMove, TooFewMove

from bamboo.util cimport SGFMoveIterator
from bamboo.go.board cimport PASS, S_EMPTY, STRING_EMPTY_END
from bamboo.go.board cimport game_state_t, string_t 
from bamboo.go.board cimport onboard_index, get_md12
from bamboo.go.printer cimport print_board
from bamboo.rollout.pattern cimport x33_bits, x33_trans8_min, x33_trans16_min, print_x33
from bamboo.rollout.pattern cimport d12_bits, d12_trans8_min, d12_trans16_min, print_d12


cdef unsigned long long pat3[361]

freq_dict = defaultdict(int)
move_dict = defaultdict(int)

def harvest_3x3_pattern(file_name, verbose=False, quiet=False):    
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
    try:
        for i, move in enumerate(sgf_iter):
            if game.moves > 0 and game.record[game.moves - 1].pos != PASS:
                updated_string_num = &game.updated_string_num[<int>game.current_color]
                updated_string_id = game.updated_string_id[<int>game.current_color]
                for i in range(updated_string_num[0]):
                    string_id = updated_string_id[i]
                    string = &game.string[string_id]
                    center = string.empty[0]
                    while center != STRING_EMPTY_END:
                        pat = x33_bits(game, center, <int>game.current_color)
                        if pat3[onboard_index[center]] != pat:
                            pat3[onboard_index[center]] = pat
                            freq_dict[pat] += 1
                            if move and move[0] == center:
                                move_dict[pat] += 1
                            else:
                                move_dict[pat] += 0 
                        center = string.empty[center]
                updated_string_num[0] = 0
    except IllegalMove:
        if not quiet:
            warnings.warn('IllegalMove {:d}[{:d}] at {:d} in {:s}\n'.format(move[1], move[0], i, file_name))
        if verbose:
            err, msg, _ = sys.exc_info()
            sys.stderr.write("{} {}\n".format(err, msg))
            sys.stderr.write(traceback.format_exc())


def save_3x3_pattern(outfile):
    df = pd.DataFrame({'freq': freq_dict, 'move_freq': move_dict})
    # add move_ratio
    df['move_ratio'] = df['move_freq']/df['freq']
    # add min8, min16 pat
    min8 = []
    min16 = []
    print('Generating minhash ...')
    for i, row in df.iterrows():
        min8.append(x33_trans8_min(row.name))
        min16.append(x33_trans16_min(row.name))
    assert df.shape[0] == len(min8) == len(min16), 'Size mismatch'
    df['min8'] = min8
    df['min16'] = min16
    pd.set_option('display.width', 200)
    print(df)
    df.to_csv(outfile, index_label='pat')


def harvest_12diamond_pattern(file_name, verbose=False, quiet=False):
    cdef game_state_t *game
    cdef SGFMoveIterator sgf_iter
    cdef int empty_ix[12]
    cdef int empty_pos[12]
    cdef int n_empty_val = 0
    cdef int *n_empty = &n_empty_val
    cdef int i, j
    cdef unsigned long long bits, positional_bits

    with open(file_name, 'r') as file_object:
        sgf_iter = SGFMoveIterator(19, file_object.read())

    game = sgf_iter.game
    try:
        for i, move in enumerate(sgf_iter):
            if game.moves > 0 and game.record[game.moves - 1].pos != PASS:
                prev_pos = game.record[game.moves - 1].pos
                prev_color = game.record[game.moves - 1].color
                # generate base bits (color and liberty count)
                bits = d12_bits(game, prev_pos, prev_color, empty_ix, empty_pos, n_empty)
                for j in range(n_empty_val):
                    # add candidate(empty) position bit (no legal check)
                    positional_bits = bits | (1 << empty_ix[j])
                    freq_dict[positional_bits] += 1
                    if move and move[0] == empty_pos[j]:
                        move_dict[positional_bits] += 1
                    else:
                        move_dict[positional_bits] += 0 
    except IllegalMove:
        if not quiet:
            warnings.warn('IllegalMove {:d}[{:d}] at {:d} in {:s}\n'.format(move[1], move[0], i, file_name))
        if verbose:
            err, msg, _ = sys.exc_info()
            sys.stderr.write("{} {}\n".format(err, msg))
            sys.stderr.write(traceback.format_exc())


def save_12diamond_pattern(outfile):
    df = pd.DataFrame({'freq': freq_dict, 'move_freq': move_dict})
    # add move_ratio
    df['move_ratio'] = df['move_freq']/df['freq']
    # add min8, min16 pat
    min8 = []
    min16 = []
    print('Generating minhash ...')
    for i, row in df.iterrows():
        min8.append(d12_trans8_min(row.name))
        min16.append(d12_trans16_min(row.name))
    assert df.shape[0] == len(min8) == len(min16), 'Size mismatch'
    df['min8'] = min8
    df['min16'] = min16
    pd.set_option('display.width', 200)
    print(df)
    df.to_csv(outfile, index_label='pat')


def main(cmd_line_args=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--outfile", "-o", required=True,
                        help="Destination to write data")
    parser.add_argument("--pattern", "-p", type=str, default='x33', choices=['x33', 'd12'],
                        help="Choice pattern to harvest (Default: x33)")
    parser.add_argument("--directory", "-d", default=None,
                        help="Directory containing SGF files to process. if not present, expects files from stdin")
    parser.add_argument("--recurse", "-R", default=False, action="store_true",
                        help="Set to recurse through directories searching for SGF files")
    parser.add_argument("--verbose", "-v", default=False, action="store_true", help="Turn on verbose mode")
    parser.add_argument("--quiet", "-q", default=False, action="store_true", help="Turn on quiet mode")

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    def _is_sgf(fname):
        return fname.strip()[-4:] == ".sgf"

    def _count_all_sgfs(root):
        count = 0
        for (dirpath, dirname, files) in os.walk(root):
            for filename in files:
                if _is_sgf(filename):
                    count += 1
        return count

    def _walk_all_sgfs(root):
        """a helper function/generator to get all SGF files in subdirectories of root
        """
        for (dirpath, dirname, files) in os.walk(root):
            for filename in files:
                if _is_sgf(filename):
                    # yield the full (relative) path to the file
                    yield os.path.join(dirpath, filename)

    def _list_sgfs(path):
        files = os.listdir(path)
        return [os.path.join(path, f) for f in files if _is_sgf(f)]

    if args.directory:
        sgf_total = _count_all_sgfs(args.directory)
        if args.recurse:
            sgf_files = _walk_all_sgfs(args.directory)
        else:
            sgf_files = _list_sgfs(args.directory)
    else:
        sgf_total = 1
        sgf_files = [f.strip() for f in sys.stdin if _is_sgf(f)]

    n_parse_error = 0
    n_not19 = 0
    n_too_few_move = 0
    n_too_many_move = 0
    n_other_error = 0

    if args.pattern == 'x33':
        harvest_func = harvest_3x3_pattern
        save_func = save_3x3_pattern
    else:
        harvest_func = harvest_12diamond_pattern
        save_func = save_12diamond_pattern

    pbar = tqdm(total=sgf_total)
    for i, sgf_file in enumerate(sgf_files):
        pbar.update(1)
        try:
            harvest_func(sgf_file, verbose=args.verbose, quiet=args.quiet)
        except sgf.ParseException:
            n_parse_error += 1
            if not args.quiet:
                warnings.warn('ParseException. {:s}'.format(sgf_file))
            if args.verbose:
                err, msg, _ = sys.exc_info()
                sys.stderr.write("{} {}\n".format(err, msg))
                sys.stderr.write(traceback.format_exc())
        except SizeMismatchError:
            n_not19 += 1
        except TooFewMove as e:
            n_too_few_move += 1
            if not args.quiet:
                warnings.warn('Too few move. {:d} less than 50. {:s}'.format(e.n_moves, sgf_file))
        except TooManyMove as e:
            n_too_many_move += 1
            if not args.quiet:
                warnings.warn('Too many move. {:d} more than 500. {:s}'.format(e.n_moves, sgf_file))
        except KeyboardInterrupt:
            break
        except:
            n_other_error += 1
            if not args.quiet:
                warnings.warn('Unexpected error. {:s}'.format(sgf_file))
            if args.verbose:
                err, msg, _ = sys.exc_info()
                sys.stderr.write("{} {}\n".format(err, msg))
                sys.stderr.write(traceback.format_exc())

    print('Total {:d}/{:d} (Not19 {:d} ParseErr {:d} TooFewMove {:d} TooManyMove {:d} Other {:d})'.format(
        i+1 - n_parse_error - n_not19 - n_too_few_move - n_too_many_move - n_other_error,
        i+1,
        n_parse_error,
        n_not19,
        n_too_few_move,
        n_too_many_move,
        n_other_error))

    save_func(args.outfile)
