#!/usr/bin/env python

import argparse
import os
import shutil
import sys

from tqdm import tqdm

from bamboo.sgf_util import SGFMoveIterator
from bamboo.sgf_error import SizeMismatchError, IllegalMove, TooManyMove, TooFewMove

def extract_decent_sgf():
    """Run conversions. command-line args may be passed in as a list
    """

    parser = argparse.ArgumentParser(
        description='Prepare SGF Go game files for training the rollout model.')
    parser.add_argument("--output-directory", "-o", required=True,
                        help="Destination to copy decent SGF files")
    parser.add_argument("--input-directory", "-i", required=True,
                        help="Directory containing SGF files to process.")
    parser.add_argument("--size", "-s", type=int, default=19,
                        help="Size of the game board. SGFs not matching this are discarded")
    parser.add_argument("--min-move", "-min", type=int, default=50,
                        help="Threshold of min moves ignoring. SGFs which has moves less than this are discarded")
    parser.add_argument("--max-move", "-max", type=int, default=500,
                        help="Threshold of max moves ignoring. SGFs which has moves less than this are discarded")
    parser.add_argument("--recurse", "-R", default=False, action="store_true",
                        help="Set to recurse through directories searching for SGF files")
    parser.add_argument("--verbose", "-v", default=False, action="store_true",
                        help="Turn on verbose mode")
    parser.add_argument("--quiet", "-q", default=False, action="store_true",
                        help="Turn on quiet mode")

    args = parser.parse_args()

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
        """helper function to get all SGF files in a directory (does not recurse)
        """
        files = os.listdir(path)
        return (os.path.join(path, f) for f in files if _is_sgf(f))

    # get an iterator of SGF files according to command line args
    sgf_count = _count_all_sgfs(args.input_directory)
    if args.recurse:
        sgf_files = _walk_all_sgfs(args.input_directory)
    else:
        sgf_files = _list_sgfs(args.input_directory)

    if not os.path.exists(args.output_directory):
        os.mkdir(args.output_directory)

    n_parse_error = 0
    n_not19 = 0
    n_too_few_move = 0
    n_too_many_move = 0
    n_illegal_move = 0
    n_other_error = 0

    pbar = tqdm(total=sgf_count)
    for i, sgf_file in enumerate(sgf_files):
        try:
            with open(sgf_file, 'r') as file_object:
                sgf_iter = SGFMoveIterator(19, file_object.read(), args.min_move, args.max_move, False)

            for move in sgf_iter:
                pass

            shutil.copy2(sgf_file, args.output_directory)
        except KeyboardInterrupt:
            break
        except SizeMismatchError:
            n_not19 += 1
            if not args.quiet:
                sys.stderr.write('SizeMismatch. {:s}\n'.format(sgf_file))
        except TooFewMove as e:
            n_too_few_move += 1
            if not args.quiet:
                sys.stderr.write('Too few move. {:d} less than {:d}. {:s}\n'.format(e.n_moves, args.min_move, sgf_file))
        except TooManyMove as e:
            n_too_many_move += 1
            if not args.quiet:
                sys.stderr.write('Too many move. {:d} more than {:d}. {:s}\n'.format(e.n_moves, args.max_move, sgf_file))
        except IllegalMove as e:
            n_illegal_move += 1
            if not args.quiet:
                sys.stderr.write('IllegalMove {:d}[{:d}]. {:s}\n'.format(e.color, e.pos, sgf_file))
        except sgf.ParseException:
            n_parse_error += 1
            if not args.quiet:
                sys.stderr.write('ParseException. {:s}\n'.format(sgf_file))
                if args.verbose:
                    err, msg, _ = sys.exc_info()
                    sys.stderr.write("{} {}\n".format(err, msg))
                    sys.stderr.write(traceback.format_exc())
        except:
            n_other_error += 1
            if not args.quiet:
                sys.stderr.write('Unexpected error. {:s}'.format(sgf_file))
                if args.verbose:
                    err, msg, _ = sys.exc_info()
                    sys.stderr.write("{} {}\n".format(err, msg))
                    sys.stderr.write(traceback.format_exc())
        finally:
            pbar.update(1)
        
    print('Finished. {:d}/{:d} (Not19 {:d} TooFewMove {:d} TooManyMove {:d} IllegalMove {:d} ParseErr {:d} Other {:d})'.format(
        i + 1 - n_not19 - n_too_few_move - n_too_many_move - n_illegal_move - n_parse_error - n_other_error,
        sgf_count,
        n_not19,
        n_too_few_move,
        n_too_many_move,
        n_illegal_move,
        n_parse_error,
        n_other_error))

if __name__ == '__main__':
    extract_decent_sgf()
