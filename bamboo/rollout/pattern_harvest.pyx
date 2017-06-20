import os
import pandas as pd
import sgf
import sys
import traceback
import warnings

from tqdm import tqdm

from libc.stdio cimport printf

from bamboo.util_error import SizeMismatchError, IllegalMove, TooManyMove, TooFewMove

from bamboo.util cimport SGFMoveIterator
from bamboo.go.board cimport PASS, CORRECT_X, CORRECT_Y, OB_SIZE, STRING_EMPTY_END
from bamboo.go.board cimport game_state_t, string_t, board_size 
from bamboo.go.board cimport allocate_game, free_game, onboard_index, set_board_size, initialize_board
from bamboo.go.printer cimport print_board
from bamboo.rollout.pattern cimport x33_bits, x33_trans8_min, x33_trans16_min, print_x33

from collections import Counter, defaultdict

cdef unsigned long long pat3[361]

freq_dict = defaultdict(int)
move_dict = defaultdict(int)

def harvest_pattern(file_name, verbose=False, quiet=False):    
    cdef game_state_t *game
    cdef SGFMoveIterator sgf_iter
    cdef int *updated_string_num
    cdef int *updated_string_id
    cdef int i
    cdef int string_id
    cdef string_t *string
    cdef int center
    cdef unsigned long long pat, pat8_min, pat16_min

    with open(file_name, 'r') as file_object:
        sgf_iter = SGFMoveIterator(19, file_object.read())

    game = sgf_iter.game
    try:
        for i, move in enumerate(sgf_iter):
            if move[0] != PASS:
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
                            if sgf_iter.next_move and sgf_iter.next_move[0] == center:
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


def main(cmd_line_args=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--outfile", "-o", help="Destination to write data", required=True)  # noqa: E501
    parser.add_argument("--directory", "-d", help="Directory containing SGF files to process. if not present, expects files from stdin", default=None)  # noqa: E501
    parser.add_argument("--recurse", "-R", help="Set to recurse through directories searching for SGF files", default=False, action="store_true")  # noqa: E501
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501
    parser.add_argument("--quiet", "-q", help="Turn on quiet mode", default=False, action="store_true")  # noqa: E501

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
    pbar = tqdm(total=sgf_total)
    for i, sgf_file in enumerate(sgf_files):
        pbar.update(1)
        try:
            harvest_pattern(sgf_file, verbose=args.verbose, quiet=args.quiet)
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

    df = pd.DataFrame({'freq': freq_dict, 'move_freq': move_dict})
    # add move_ratio
    df['move_ratio'] = df['move_freq']/df['freq']
    # add min8, min16 pat
    min8 = []
    min16 = []
    for i, row in df.iterrows():
        min8.append(x33_trans8_min(row.name))
        min16.append(x33_trans16_min(row.name))
    assert df.shape[0] == len(min8) == len(min16), 'Size mismatch'
    df['min8'] = min8
    df['min16'] = min16
    df.to_csv(args.outfile)
    print(df)
