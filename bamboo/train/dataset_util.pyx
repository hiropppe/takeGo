#!/usr/bin/env python

import argparse
import glob
import os
import re
import sgf
import shutil
import sys
import traceback

from tqdm import tqdm

from libc.stdio cimport printf 

from bamboo.board cimport game_state_t
from bamboo.zobrist_hash cimport initialize_hash 
from bamboo.sgf_util cimport SGFMoveIterator
from bamboo.sgf_error import SizeMismatchError, IllegalMove, TooManyMove, TooFewMove, NoResultError


def merge_dataset(cmd_line_args=None):
    cdef game_state_t *game
    cdef SGFMoveIterator sgf_iter
    cdef dict final_hash_dict = dict()

    """Run conversions. command-line args may be passed in as a list
    """

    parser = argparse.ArgumentParser(
        description='Prepare SGF Go game files for training the rollout model.')
    parser.add_argument("--output-directory", "-o", required=True,
                        help="Destination to copy decent SGF files")
    parser.add_argument("--input-directory", "-i", required=True,
                        help="Comma separated directories containing SGF files to process.")
    parser.add_argument("--size", "-s", type=int, default=19,
                        help="Size of the game board. SGFs not matching this are discarded")
    parser.add_argument("--min-move", "-min", type=int, default=50,
                        help="Threshold of min moves ignoring. SGFs which has moves less than this are discarded")
    parser.add_argument("--max-move", "-max", type=int, default=500,
                        help="Threshold of max moves ignoring. SGFs which has moves less than this are discarded")
    parser.add_argument("--recurse", "-R", default=False, action="store_true",
                        help="Set to recurse through directories searching for SGF files")
    parser.add_argument("--ignore_no_result", default=False, action="store_true",
                        help="Ignoring sgf without [RE] propery")
    parser.add_argument("--verbose", "-v", default=False, action="store_true",
                        help="Turn on verbose mode")
    parser.add_argument("--quiet", "-q", default=False, action="store_true",
                        help="Turn on quiet mode")

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    def _is_sgf(fname):
        return fname.strip()[-4:] == ".sgf"

    def _count_all_sgfs(dirs):
        count = 0
        for d in dirs:
            for (dirpath, dirname, files) in os.walk(d):
                for filename in files:
                    if _is_sgf(filename):
                        count += 1
        return count

    def _walk_all_sgfs(dirs):
        """a helper function/generator to get all SGF files in subdirectories of root
        """
        for d in dirs:
            print('Enter {:s}'.format(d))
            for (dirpath, dirname, files) in os.walk(d):
                for filename in files:
                    if _is_sgf(filename):
                        # yield the full (relative) path to the file
                        yield os.path.join(dirpath, filename)

    def _list_sgfs(dirs):
        """helper function to get all SGF files in a directory (does not recurse)
        """
        for d in dirs:
            print('Enter {:s}'.format(d))
            for f in os.listdir(d):
                if _is_sgf(f):
                    yield os.path.join(d, f)

    # get an iterator of SGF files according to command line args
    input_dirs = [d.strip() for d in re.split(r'[\s,]+', args.input_directory)]
    sgf_count = _count_all_sgfs(input_dirs)
    if args.recurse:
        sgf_files = _walk_all_sgfs(input_dirs)
    else:
        sgf_files = _list_sgfs(input_dirs)

    if not os.path.exists(args.output_directory):
        os.mkdir(args.output_directory)

    initialize_hash()

    n_parse_error = 0
    n_not19 = 0
    n_too_few_move = 0
    n_too_many_move = 0
    n_illegal_move = 0
    n_no_result = 0
    n_hash_corrision = 0
    n_other_error = 0

    pbar = tqdm(total=sgf_count)
    for sgf_file in sgf_files:
        if args.verbose:
            print(sgf_file)
        try:
            with open(sgf_file, 'r') as file_object:
                sgf_iter = SGFMoveIterator(19,
                                           file_object.read(),
                                           args.min_move,
                                           args.max_move,
                                           ignore_not_legal=False,
                                           ignore_no_result=args.ignore_no_result)
            game = sgf_iter.game

            for j, move in enumerate(sgf_iter):
                pass

            final_hash = game.current_hash
            if final_hash not in final_hash_dict:
                final_hash_dict[final_hash] = sgf_file
                shutil.copy2(sgf_file, args.output_directory)
            else:
                n_hash_corrision += 1
                if not args.quiet:
                    print('Hash corrision !! "{:s}" equals to "{:s}". hash={:s}\n'.format(
                        sgf_file,
                        str(final_hash_dict[final_hash]),
                        str(final_hash)))
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
        except NoResultError as e:
            n_no_result += 1
            if not args.quiet:
                sys.stderr.write('NoResult {:s}\n'.format(sgf_file))
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
        
    print('Finished. {:d}/{:d} (Not19 {:d} TooFewMove {:d} TooManyMove {:d} NoResult {:d} IllegalMove {:d} SameHash {:d} ParseErr {:d} Other {:d})'.format(
        sgf_count - n_not19 - n_too_few_move - n_too_many_move - n_no_result - n_illegal_move - n_hash_corrision - n_parse_error - n_other_error,
        sgf_count,
        n_not19,
        n_too_few_move,
        n_too_many_move,
        n_no_result,
        n_illegal_move,
        n_hash_corrision,
        n_parse_error,
        n_other_error))


def split_dataset_by_komi(cmd_line_args=None):
    cdef SGFMoveIterator sgf_iter

    parser = argparse.ArgumentParser(
        description='Split sgf dataset by komi for training the value network.')
    parser.add_argument("--output_directory", "-o", required=True,
                        help="Destination to copy decent SGF files")
    parser.add_argument("--input_directory", "-i", required=True,
                        help="Directory containing SGF files to process.")
    parser.add_argument("--size", "-s", type=int, default=19,
                        help="Size of the game board. SGFs not matching this are discarded")
    parser.add_argument("--min_move", "-min", type=int, default=50,
                        help="Threshold of min moves ignoring. SGFs which has moves less than this are discarded")
    parser.add_argument("--max_move", "-max", type=int, default=500,
                        help="Threshold of max moves ignoring. SGFs which has moves less than this are discarded")
    parser.add_argument("--recurse", "-R", default=False, action="store_true",
                        help="Set to recurse through directories searching for SGF files")
    parser.add_argument("--ignore_no_result", default=False, action="store_true",
                        help="Ignoring sgf without [RE] propery")
    parser.add_argument("--verbose", "-v", default=False, action="store_true",
                        help="Turn on verbose mode")
    parser.add_argument("--quiet", "-q", default=False, action="store_true",
                        help="Turn on quiet mode")

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
    n_no_result = 0
    n_illegal_move = 0
    n_other_error = 0

    pbar = tqdm(total=sgf_count)
    for sgf_file in sgf_files:
        try:
            with open(sgf_file, 'r') as file_object:
                sgf_iter = SGFMoveIterator(19,
                                           file_object.read(),
                                           args.min_move,
                                           args.max_move,
                                           ignore_not_legal=False,
                                           ignore_no_result=args.ignore_no_result)

                for move in sgf_iter:
                    pass

                output_path = os.path.join(args.output_directory, str(sgf_iter.komi))
                if not os.path.exists(output_path):
                    os.mkdir(output_path)
                shutil.copy2(sgf_file, output_path)
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
        except NoResultError as e:
            n_no_result += 1
            if not args.quiet:
                sys.stderr.write('NoResult {:s}\n'.format(sgf_file))
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

    print('Finished. {:d}/{:d} (Not19 {:d} TooFewMove {:d} TooManyMove {:d} NoResult {:d} IllegalMove {:d} ParseErr {:d} Other {:d})'.format(
        sgf_count - n_not19 - n_too_few_move - n_too_many_move - n_no_result - n_illegal_move - n_parse_error - n_other_error,
        sgf_count,
        n_not19,
        n_too_few_move,
        n_too_many_move,
        n_no_result,
        n_illegal_move,
        n_parse_error,
        n_other_error))

    for d in glob.glob(os.path.join(args.output_directory, '*')):
        print('{:s}: {:d}'.format(os.path.basename(d), len(os.listdir(d))))
