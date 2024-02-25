# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

import numpy as np
cimport numpy as np
import os
import warnings
import sgf
import sys
import traceback
import time
import h5py

from tqdm import tqdm

from libc.stdlib cimport malloc, free
from libc.stdio cimport printf

from bamboo.sgf_error import SizeMismatchError, IllegalMove, TooManyMove, TooFewMove

from bamboo.sgf_util cimport SGFMoveIterator
from bamboo.board cimport PASS
from bamboo.board cimport game_state_t, pure_board_max, onboard_index
from bamboo.policy_feature cimport MAX_POLICY_PLANES
from bamboo.policy_feature cimport PolicyFeature, allocate_feature, initialize_feature, free_feature, update, free_feature_games
from bamboo.tree_search cimport tree_node_t
from bamboo.printer cimport print_board


def onboard_index_to_np_move(ix, size):
    y, x = divmod(ix, size)
    return (x, y)


cdef class GameConverter(object):

    cdef:
        int bsize
        PolicyFeature feature
        int n_features
        list update_speeds

    def __cinit__(self, bsize=19):
        self.bsize = bsize
        self.feature = allocate_feature(MAX_POLICY_PLANES)
        self.n_features = self.feature.n_planes
        self.update_speeds = list()

    def __dealloc__(self):
        free_feature(self.feature)

    def convert_game(self, file_name, verbose=False):
        cdef tree_node_t *node
        cdef game_state_t *game
        cdef SGFMoveIterator sgf_iter

        node = <tree_node_t *>malloc(sizeof(tree_node_t))
        for i in range(self.bsize**2):
            node.children[i] = <tree_node_t *>malloc(sizeof(tree_node_t))
        initialize_feature(self.feature)
        try:
            with open(file_name, 'r') as file_object:
                sgf_iter = SGFMoveIterator(self.bsize, file_object.read())

            game = sgf_iter.game
            node.game = game
            for i, move in enumerate(sgf_iter):
                if move[0] != PASS:
                    s = time.time()
                    update(self.feature, node)
                    self.update_speeds.append(time.time()-s)
                    if onboard_index[move[0]] >= pure_board_max:
                        continue
                    else:
                        planes = np.asarray(self.feature.planes)
                        planes = planes.reshape(1, self.n_features, self.bsize, self.bsize)
                        planes = planes.transpose(0, 1, 3, 2)  # required?
                        yield (planes, onboard_index_to_np_move(onboard_index[move[0]], self.bsize))
        finally:
            free_feature_games(self.feature)
            for i in range(self.bsize**2):
                free(node.children[i])
            free(node)

    def sgfs_to_hdf5(self,
                     sgf_files,
                     sgf_total,
                     hdf5_file,
                     ignore_errors=True, verbose=False, quiet=False):
        """ Save each feature onehot index in board shape matrix.
        """
        tmp_file = os.path.join(os.path.dirname(hdf5_file), ".tmp." + os.path.basename(hdf5_file))
        h5f = h5py.File(tmp_file, 'w')

        try:
            states = h5f.require_dataset(
                'states',
                dtype=np.int32,
                shape=(1, self.n_features, self.bsize, self.bsize),
                maxshape=(None, self.n_features, self.bsize, self.bsize),  # 'None' == arbitrary size
                exact=False, 
                chunks=(64, self.n_features, self.bsize, self.bsize),
                compression="lzf")
            actions = h5f.require_dataset(
                'actions',
                dtype=np.int32,
                shape=(1, 2),
                maxshape=(None, 2),
                exact=False,
                chunks=(1024, 2),
                compression="lzf")

            # 'file_offsets' is an HDF5 group so that 'file_name in file_offsets' is fast
            file_offsets = h5f.require_group('file_offsets')

            if verbose:
                print("created HDF5 dataset in {}".format(tmp_file))

            next_idx = 0
            n_parse_error = 0
            n_not19 = 0
            n_too_few_move = 0
            n_too_many_move = 0

            pbar = tqdm(total=sgf_total)
            for file_name in sgf_files:
                pbar.update(1)
                if verbose:
                    print(file_name)
                n_pairs = 0
                file_start_idx = next_idx
                try:
                    for state, move in self.convert_game(file_name):
                        if next_idx >= len(states):
                            states.resize((next_idx + 1, self.n_features, self.bsize, self.bsize))
                            actions.resize((next_idx + 1, 2))
                        states[next_idx] = state
                        actions[next_idx] = move
                        n_pairs += 1
                        next_idx += 1
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
                except KeyboardInterrupt:
                    break
                finally:
                    if n_pairs > 0:
                        # '/' has special meaning in HDF5 key names, so they
                        # are replaced with ':' here
                        file_name_key = file_name.replace('/', ':')
                        file_offsets[file_name_key] = [file_start_idx, n_pairs]
                        if verbose:
                            print("\t%d state/action pairs extracted" % n_pairs)
                    elif verbose:
                        print("\t-no usable data-")
        except Exception as e:
            print("sgfs_to_hdf5 failed")
            err, msg, _ = sys.exc_info()
            sys.stderr.write("{} {}\n".format(err, msg))
            sys.stderr.write(traceback.format_exc())
            os.remove(tmp_file)
            raise e

        h5f['n_features'] = self.n_features

        if verbose:
            print("finished. renaming %s to %s" % (tmp_file, hdf5_file))

        print('Total {:d}/{:d} (Not19 {:d} ParseErr {:d} TooFewMove {:d} TooManyMove {:d})'.format(
            len(sgf_files) - n_parse_error - n_not19 - n_too_few_move - n_too_many_move,
            len(sgf_files),
            n_parse_error,
            n_not19,
            n_too_few_move,
            n_too_many_move))
        print('Update Speed: Avg. {:3f} us'.format(np.mean(self.update_speeds)*1000*1000))

        # processing complete; rename tmp_file to hdf5_file
        h5f.close()
        os.rename(tmp_file, hdf5_file)
