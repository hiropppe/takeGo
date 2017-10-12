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

from bamboo.sgf_error import SizeMismatchError, IllegalMove, TooManyMove, TooFewMove

from bamboo.sgf_util cimport SGFMoveIterator
from bamboo.board cimport PURE_BOARD_MAX, S_BLACK, S_WHITE, PASS, POS, CORRECT_X, CORRECT_Y
from bamboo.board cimport game_state_t, rollout_feature_t, pure_board_size, pure_board_max, onboard_index
from bamboo.printer cimport print_board
from bamboo.zobrist_hash cimport initialize_hash
from bamboo.nakade cimport initialize_nakade_hash
from bamboo.local_pattern cimport read_rands, init_x33_hash, init_d12_hash, init_nonres_d12_hash
from bamboo.local_pattern cimport x33_hash, x33_hashmap
from bamboo.rollout_preprocess cimport tree_feature_size
from bamboo.rollout_preprocess cimport initialize_const, initialize_planes, update_planes, update_tree_planes_all 


cdef class GameConverter(object):

    cdef:
        int bsize
        list update_speeds

    def __cinit__(self, bsize=19, rands_file=None, x33_file=None, d12_file=None, nonres_d12_file=None):
        cdef int nakade_size, x33_size, d12_size

        self.bsize = bsize

        read_rands(rands_file)

        initialize_hash()

        nakade_size = initialize_nakade_hash()
        x33_size = init_x33_hash(x33_file)
        d12_size = init_d12_hash(d12_file)
        nonres_d12_size = init_nonres_d12_hash(nonres_d12_file)

        self.update_speeds = list()

        initialize_const(nakade_size, x33_size, d12_size, nonres_d12_size)

    def __dealloc__(self):
        pass

    def convert_game(self, file_name, verbose=False):
        cdef game_state_t *game
        cdef rollout_feature_t *feature
        cdef SGFMoveIterator sgf_iter
        cdef int i, j

        with open(file_name, 'r') as file_object:
            sgf_iter = SGFMoveIterator(self.bsize,
                                       file_object.read(),
                                       ignore_not_legal=True,
                                       ignore_no_result=False)
        game = sgf_iter.game

        initialize_planes(game)
        for i, move in enumerate(sgf_iter):
            if move[0] != PASS:
                s = time.time()

                update_planes(game)
                update_tree_planes_all(game)

                self.update_speeds.append(time.time()-s)

                feature = &game.rollout_feature_planes[<int>game.current_color]
                onehot_index_array = np.asarray(feature.tensor)

                if onboard_index[move[0]] >= pure_board_max:
                    continue
                else:
                    yield (onehot_index_array, onboard_index[move[0]])

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
                shape=(1, 9, PURE_BOARD_MAX),
                maxshape=(None, 9, PURE_BOARD_MAX),  # 'None' == arbitrary size
                exact=False, 
                chunks=(64, 9, PURE_BOARD_MAX),
                compression="lzf")
            actions = h5f.require_dataset(
                'actions',
                dtype=np.int32,
                shape=(1, 1),
                maxshape=(None, 1),
                exact=False,
                chunks=(1024, 1),
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
            for file_name in tqdm(sgf_files):
                pbar.update(1)
                if verbose:
                    print(file_name)
                n_pairs = 0
                file_start_idx = next_idx
                try:
                    for state, move in self.convert_game(file_name):
                        if next_idx >= len(states):
                            states.resize((next_idx + 1, 9, PURE_BOARD_MAX))
                            actions.resize((next_idx + 1, 1))
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
            print("sgfs_to_onehot_index_array failed")
            err, msg, _ = sys.exc_info()
            sys.stderr.write("{} {}\n".format(err, msg))
            sys.stderr.write(traceback.format_exc())
            os.remove(tmp_file)
            raise e

        h5f['n_features'] = tree_feature_size 

        if verbose:
            print("finished. renaming %s to %s" % (tmp_file, hdf5_file))

        print('Total {:d}/{:d} (Not19 {:d} ParseErr {:d} TooFewMove {:d} TooManyMove {:d})'.format(
            sgf_total - n_parse_error - n_not19 - n_too_few_move - n_too_many_move,
            sgf_total,
            n_parse_error,
            n_not19,
            n_too_few_move,
            n_too_many_move))
        print('Update Speed: Avg. {:3f} us'.format(np.mean(self.update_speeds)*1000*1000))

        # processing complete; rename tmp_file to hdf5_file
        h5f.close()
        os.rename(tmp_file, hdf5_file)
