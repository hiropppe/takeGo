# cython: boundscheck = False
# cython: cdivision = True

import dill
import itertools
import numpy as np
cimport numpy as np
import os
import warnings
import re
import sgf
import sys
import tensorflow as tf
import time
import traceback

from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor

from libc.stdio cimport printf

from bamboo.sgf_error import SizeMismatchError, IllegalMove, TooManyMove, TooFewMove, NoResultError

from bamboo.sgf_util cimport SGFMoveIterator
from bamboo.board cimport PASS, S_BLACK, S_WHITE
from bamboo.board cimport game_state_t, pure_board_max, onboard_index
from bamboo.policy_feature cimport MAX_VALUE_PLANES
from bamboo.policy_feature cimport policy_feature_t, allocate_feature, initialize_feature, free_feature, update
from bamboo.printer cimport print_board


BOARD_TRANSFORMATIONS = {
    'noop': lambda feature: feature,
    'rot90': lambda feature: np.rot90(feature, 1),
    'rot180': lambda feature: np.rot90(feature, 2),
    'rot270': lambda feature: np.rot90(feature, 3),
    'fliplr': lambda feature: np.fliplr(feature),
    'flipud': lambda feature: np.flipud(feature),
    'diag1': lambda feature: np.transpose(feature),
    'diag2': lambda feature: np.fliplr(np.rot90(feature, 1))
}


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
    for (dirpath, dirname, files) in os.walk(root):
        for filename in files:
            if is_sgf(filename):
                yield os.path.join(dirpath, filename)


def walk_worker_sgfs(root, n_workers, worker_seq):
    i = 0
    for (dirpath, dirname, files) in os.walk(root):
        for filename in files:
            if is_sgf(filename) and (i % n_workers == worker_seq):
                yield os.path.join(dirpath, filename)
            i += 1


def sgfs_to_tfrecord(data_directory,
                     out_name,
                     n_workers,
                     split_by,
                     samples_per_game,
                     verbose=False,
                     quiet=False):
    if n_workers > 1:
        print('Run {:d} workers (output {:d} files).'.format(n_workers, n_workers))
        if split_by == 'transformation':
            print('Each worker process all sgf file but partial transformations.')
        else:
            print('Each worker process partial sgf files but all transformation.')

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for worker_seq in range(n_workers):
                executor.submit(sgfs_to_tfrecord_by_worker,
                                data_directory,
                                out_name,
                                n_workers,
                                worker_seq,
                                split_by,
                                samples_per_game,
                                verbose=verbose,
                                quiet=quiet)
    else:
        write_tfrecord_all(data_directory,
                           out_name,
                           samples_per_game,
                           verbose,
                           quiet)


def sgfs_to_tfrecord_by_worker(directory, 
                               out_name,
                               n_workers,
                               worker_seq,
                               split_by,
                               samples_per_game,
                               verbose,
                               quiet):
    if split_by == 'transformation':
        write_tfrecord_by_transformation(directory,
                out_name,
                n_workers,
                worker_seq,
                samples_per_game,
                verbose,
                quiet)
    else:
        write_tfrecord_by_data(directory,
                out_name,
                n_workers,
                worker_seq,
                samples_per_game,
                verbose,
                quiet)


def write_tfrecord_all(data_directory, out_name, samples_per_game, verbose, quiet):
    n_sgfs = count_all_sgfs(data_directory)
    sgf_generator = walk_all_sgfs(data_directory)
    converter = GameConverter()
    converter.sgfs_to_tfrecord(sgf_generator,
        n_sgfs,
        out_name,
        None,
        BOARD_TRANSFORMATIONS,
        samples_per_game,
        verbose,
        quiet)


def write_tfrecord_by_transformation(data_directory, out_name, n_workers, worker_seq, samples_per_game, verbose, quiet):
    n_sgfs = count_all_sgfs(data_directory)
    sgf_generator = walk_all_sgfs(data_directory)
    apply_transformations = {}
    for i, (name, op) in enumerate(BOARD_TRANSFORMATIONS.items()):
        if i % n_workers == worker_seq:
            apply_transformations[name] = op

    converter = GameConverter()
    converter.sgfs_to_tfrecord(sgf_generator,
        n_sgfs,
        out_name,
        worker_seq,
        apply_transformations,
        samples_per_game,
        verbose,
        quiet)


def write_tfrecord_by_data(data_directory, out_name, n_workers, worker_seq, samples_per_game, verbose, quiet):
    n_sgfs = count_all_sgfs(data_directory)
    if worker_seq < n_sgfs % n_workers:
        n_sgfs = int(n_sgfs/n_workers) + 1
    else:
        n_sgfs = int(n_sgfs/n_workers)

    sgf_generator = walk_worker_sgfs(data_directory, n_workers, worker_seq)

    converter = GameConverter()
    converter.sgfs_to_tfrecord(sgf_generator,
        n_sgfs,
        out_name,
        worker_seq,
        BOARD_TRANSFORMATIONS,
        samples_per_game,
        verbose,
        quiet)


def onboard_index_to_np_move(ix, size):
    y, x = divmod(ix, size)
    return (x, y)


cdef class GameConverter(object):

    cdef:
        int bsize
        policy_feature_t *feature
        int n_features
        list update_speeds

    def __cinit__(self, bsize=19):
        self.bsize = bsize
        self.feature = allocate_feature(MAX_VALUE_PLANES)
        self.n_features = self.feature.n_planes
        self.update_speeds = list()

    def __dealloc__(self):
        free_feature(self.feature)

    def sgfs_to_tfrecord(self,
                         sgf_generator,
                         n_sgfs,
                         out_name,
                         worker_seq,
                         apply_transformations,
                         samples_per_game,
                         ignore_errors=True,
                         verbose=False,
                         quiet=False):
        try:
            opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
            if worker_seq is None:
                writer = tf.python_io.TFRecordWriter('{:s}.tfrecord'.format(out_name), options=opt)
            else:
                writer = tf.python_io.TFRecordWriter('{:s}-{:d}.tfrecord'.format(out_name, worker_seq), options=opt)

            n_parse_error = 0
            n_not19 = 0
            n_too_few_move = 0
            n_too_many_move = 0
            n_no_result = 0

            pbar = tqdm(total=n_sgfs)
            if worker_seq is not None:
                pbar.set_description('Worker {:d}'.format(worker_seq))
            for file_name in sgf_generator:
                pbar.update(1)
                if verbose:
                    print(file_name)
                n_pairs = 0
                try:
                    self.write_tfrecords(file_name, writer, apply_transformations, samples_per_game)
                    n_pairs += 1
                except sgf.ParseException:
                    n_parse_error += 1
                    warnings.warn('ParseException. {:s}'.format(file_name))
                    if verbose:
                        err, msg, _ = sys.exc_info()
                        sys.stderr.write("{} {}\n".format(err, msg))
                        sys.stderr.write(traceback.format_exc())
                except SizeMismatchError:
                    n_not19 += 1
                except TooFewMove as e:
                    n_too_few_move += 1
                    warnings.warn('Too few move. {:d} less than 50. {:s}'.format(e.n_moves, file_name))
                except TooManyMove as e:
                    n_too_many_move += 1
                    warnings.warn('Too many move. {:d} more than 500. {:s}'.format(e.n_moves, file_name))
                except NoResultError as e:
                    n_no_result += 1
                    warnings.warn('NoResultError. {:s}'.format(file_name))
                except KeyboardInterrupt:
                    break
                finally:
                    if n_pairs > 0:
                        if verbose:
                            print("\t%d state/action pairs extracted" % n_pairs)
                    elif verbose:
                        print("\t-no usable data-")
        except Exception as e:
            print("sgfs_to_tfrecord failed")
            err, msg, _ = sys.exc_info()
            sys.stderr.write("{} {}\n".format(err, msg))
            sys.stderr.write(traceback.format_exc())
            raise e
        finally:
            if writer:
                writer.close()

        print('{:s}Total {:d}/{:d} (Not19 {:d} ParseErr {:d} TooFewMove {:d} TooManyMove {:d} NoResult {:d})'.format(
            'Wrorker:{:d} '.format(worker_seq) if worker_seq is not None else '',
            n_sgfs - n_parse_error - n_not19 - n_too_few_move - n_too_many_move,
            n_sgfs,
            n_parse_error,
            n_not19,
            n_too_few_move,
            n_too_many_move,
            n_no_result))
        print('Update Speed: Avg. {:3f} us'.format(np.mean(self.update_speeds)*1000*1000))

    def write_tfrecords(self, sgf_file, writer, apply_transformations, samples_per_game):
        """Converts a dataset to tfrecords."""
        with open(sgf_file) as f:
            n_moves = len(re.findall(r';[WB]\[[a-z]*?\]', f.read(), flags=re.IGNORECASE))

        if samples_per_game:
            sample_idx = np.random.randint(1, n_moves, samples_per_game)

        for i, (state, z) in enumerate(self.convert_game(sgf_file)):
            if samples_per_game and (i not in sample_idx):
                continue
            noop = state
            for name, op in apply_transformations.items():
                transformed_state = op(state)
                if name == 'noop' or (not np.all(transformed_state == state)):
                    d_feature = {}
                    d_feature['state'] = tf.train.Feature(float_list=tf.train.FloatList(value=state.flatten()))
                    d_feature['z'] = tf.train.Feature(float_list=tf.train.FloatList(value=[z]))

                    features = tf.train.Features(feature=d_feature)
                    example = tf.train.Example(features=features)
                    serialized = example.SerializeToString()
                    writer.write(serialized)

    def convert_game(self, file_name, verbose=False):
        cdef game_state_t *game
        cdef SGFMoveIterator sgf_iter

        initialize_feature(self.feature)

        with open(file_name, 'r') as file_object:
            sgf_iter = SGFMoveIterator(self.bsize, file_object.read())

        game = sgf_iter.game
        for i, move in enumerate(sgf_iter):
            if move[0] != PASS:
                s = time.time()
                update(self.feature, game)
                self.update_speeds.append(time.time()-s)
                if onboard_index[move[0]] >= pure_board_max:
                    continue
                else:
                    planes = np.asarray(self.feature.planes)
                    planes = planes.reshape(self.n_features, self.bsize, self.bsize)
                    # transpose to TF input shape (samples, rows, cols, input_depth)
                    planes = planes.transpose(1, 2, 0)
                    planes = planes.astype(np.float32)

                    if sgf_iter.winner == S_BLACK:
                        z = 1.0
                    elif sgf_iter.winner == S_WHITE:
                        z = -1.0
                    else:
                        z = 0.0
                    yield (planes, z)
