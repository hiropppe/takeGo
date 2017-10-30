#!/usr/bin/env python

import argparse
import glob
import os
import re
import sgf
import shutil
import sys
import traceback

from collections import defaultdict
from tqdm import tqdm

from libc.stdio cimport printf 

from bamboo.board cimport PASS, S_BLACK, S_WHITE, MAX_MOVES
from bamboo.board cimport FLIP_COLOR
from bamboo.board cimport game_state_t, komi
from bamboo.board cimport allocate_game, copy_game, set_komi, is_legal_not_eye, put_stone, calculate_score
from bamboo.zobrist_hash cimport initialize_hash
from bamboo.local_pattern cimport read_rands
from bamboo.sgf_util cimport SGFMoveIterator
from bamboo.sgf_error import SizeMismatchError, IllegalMove, TooManyMove, TooFewMove, NoResultError
from bamboo.zobrist_hash import set_hash_size, initialize_hash
from bamboo.nakade import initialize_nakade_hash
from bamboo.local_pattern import read_rands, init_d12_rsp_hash, init_x33_hash, init_d12_hash
from bamboo.rollout_preprocess cimport initialize_rollout_const, update_rollout, choice_rollout_move, set_illegal


def is_sgf(fname):
    return fname.strip()[-4:] == ".sgf"


def count_all_sgfs(root):
    count = 0
    for (dirpath, dirname, files) in os.walk(root):
        for filename in files:
            if is_sgf(filename):
                count += 1
    return count


def walk_all_sgfs(root):
    """a helper function/generator to get all SGF files in subdirectories of root
    """
    for (dirpath, dirname, files) in os.walk(root):
        for filename in files:
            if is_sgf(filename):
                # yield the full (relative) path to the file
                yield os.path.join(dirpath, filename)


def list_sgfs(path):
    """helper function to get all SGF files in a directory (does not recurse)
    """
    files = os.listdir(path)
    return (os.path.join(path, f) for f in files if is_sgf(f))


cdef int rollout(game_state_t *game) nogil:
    cdef int winner
    cdef double score
    cdef int color, other_color
    cdef int pos
    cdef int pass_count = 0
    cdef int moves_remain = MAX_MOVES - game.moves

    color = game.current_color
    other_color = FLIP_COLOR(game.current_color)
    while moves_remain and pass_count < 2:
        pos = PASS

        if pos == PASS or is_legal_not_eye(game, pos, color) == False:
            while True:
                pos = choice_rollout_move(game)
                if is_legal_not_eye(game, pos, color):
                    break
                else:
                    set_illegal(game, pos)

        put_stone(game, pos, color)
        game.current_color = other_color

        update_rollout(game)

        other_color = color
        color = game.current_color

        pass_count = pass_count + 1 if pos == PASS else 0
        moves_remain -= 1

    score = <double>calculate_score(game)

    if score - komi > 0:
        winner = <int>S_BLACK
    else:
        winner = <int>S_WHITE

    return winner


def generate_value_dataset(cmd_line_args=None):
    cdef SGFMoveIterator sgf_iter
    cdef game_state_t *rollout_game = allocate_game()
    cdef int black_wins, white_wins
    cdef int i

    parser = argparse.ArgumentParser(
        description='Extract dataset for training the value network.')
    parser.add_argument("--output_directory", "-o", required=True,
                        help="Destination to copy decent SGF files")
    parser.add_argument("--input_directory", "-i", required=True,
                        help="Directory containing SGF files to process.")
    parser.add_argument("--size", "-s", type=int, default=19,
                        help="Size of the game board. SGFs not matching this are discarded")
    parser.add_argument("--komi", "-k", type=float, default=7.5,
                        help="Size of komi")
    parser.add_argument("--min_move", "-min", type=int, default=50,
                        help="Threshold of min moves ignoring. SGFs which has moves less than this are discarded")
    parser.add_argument("--max_move", "-max", type=int, default=500,
                        help="Threshold of max moves ignoring. SGFs which has moves less than this are discarded")
    parser.add_argument("--recurse", "-R", default=False, action="store_true",
                        help="Set to recurse through directories searching for SGF files")
    parser.add_argument("--ignore_no_result", default=False, action="store_true",
                        help="Ignoring sgf without [RE] propery")
    parser.add_argument("--rollout_path", "-ro", type=str, required=True,
                        help="Rollout policy network weights (hdf5)")
    parser.add_argument("--mt_rands_file", "-mt", type=str, required=True,
                        help="Mersenne twister random number file")
    parser.add_argument("--x33_csv", "-x33", type=str, required=True,
                        help="Non-response 3x3 pattern file")
    parser.add_argument("--d12_rsp_csv", "-rd12", type=str, required=True,
                        help="Response 12 point diamond(MD2) pattern file")
    parser.add_argument("--d12_csv", "-d12", type=str, required=True,
                        help="Non-response 12 point diamond(MD2) pattern file")
    parser.add_argument("--threads", "-t", type=int, default=1,
                        help="Number of search threads (Default: 1)")
    parser.add_argument("--pro", "-pro", default=False, action="store_true",
                        help="True if pro dataset")
    parser.add_argument("--verbose", "-v", default=False, action="store_true",
                        help="Turn on verbose mode")
    parser.add_argument("--quiet", "-q", default=False, action="store_true",
                        help="Turn on quiet mode")

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    # get an iterator of SGF files according to command line args
    sgf_count = count_all_sgfs(args.input_directory)
    if args.recurse:
        sgf_files = walk_all_sgfs(args.input_directory)
    else:
        sgf_files = list_sgfs(args.input_directory)

    if not os.path.exists(args.output_directory):
        os.mkdir(args.output_directory)

    n_trust_pro_game = 0
    n_trust_other_game = 0

    n_parse_error = 0
    n_not19 = 0
    n_too_few_move = 0
    n_too_many_move = 0
    n_no_result = 0
    n_illegal_move = 0
    n_other_error = 0

    set_komi(args.komi)
    read_rands(args.mt_rands_file)
    initialize_rollout_const(8,
        init_x33_hash(args.x33_csv),
        init_d12_rsp_hash(args.d12_rsp_csv),
        init_d12_hash(args.d12_csv),
        pos_aware_d12=False)

    pbar = tqdm(total=sgf_count)
    for sgf_file in sgf_files:
        trust_game = False
        try:
            with open(sgf_file, 'r') as file_object:
                sgf_iter = SGFMoveIterator(19,
                                           file_object.read(),
                                           args.min_move,
                                           args.max_move,
                                           rollout=True,
                                           ignore_not_legal=False,
                                           ignore_no_result=args.ignore_no_result)

                # play moves
                for move in sgf_iter:
                    pass

                if args.pro and sgf_iter.komi == args.komi and sgf_iter.resign:
                    trust_game = True
                    n_trust_pro_game += 1
                else:
                    trust_game = False

                if not trust_game:
                    # evaluate end state by rollout
                    copy_game(rollout_game, sgf_iter.game)
                    for i in range(500):
                        winner = rollout(rollout_game)
                        if winner == S_BLACK:
                            black_wins += 1
                        elif winner == S_WHITE:
                            white_wins += 1

                    if ((sgf_iter.winner == S_BLACK and black_wins > white_wins) or
                        (sgf_iter.winner == S_WHITE and black_wins < white_wins)):
                        trust_game = True
                        n_trust_other_game += 1

            if trust_game:
                output_path = args.output_directory
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

    # get an iterator of SGF files according to command line args
    sgf_count = count_all_sgfs(args.input_directory)
    if args.recurse:
        sgf_files = walk_all_sgfs(args.input_directory)
    else:
        sgf_files = list_sgfs(args.input_directory)

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
