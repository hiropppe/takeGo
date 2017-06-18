#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
import os
import warnings
import sgf
import sys
import traceback
import tables
import h5py

from scipy.sparse import lil_matrix, csr_matrix
from tqdm import tqdm

from bamboo.util_error import SizeMismatchError, IllegalMove, TooManyMove, TooFewMove

from bamboo.util cimport SGFMoveIterator
from bamboo.go.board cimport PURE_BOARD_MAX, S_BLACK, S_WHITE, PASS, POS, CORRECT_X, CORRECT_Y
from bamboo.go.board cimport game_state_t, pure_board_size, set_board_size, pure_board_max, free_game
from bamboo.rollout.pattern cimport initialize_hash, init_nakade_hash, init_x33_hash, init_d12_hash
from bamboo.rollout.preprocess cimport RolloutFeature, rollout_feature_t
from bamboo.go.printer cimport print_board
from bamboo.rollout.pattern cimport x33_hash, x33_hashmap


cdef class GameConverter(object):

    cdef:
        RolloutFeature preprocessor
        int bsize
        int nakade_size, x33_size, d12_size
        int n_features

    def __cinit__(self, bsize=19, nakade_file=None, x33_file=None, d12_file=None):
        self.bsize = bsize

        initialize_hash()
        self.nakade_size = init_nakade_hash(nakade_file)
        self.x33_size = init_x33_hash(x33_file)
        self.d12_size = init_d12_hash(d12_file)

    def __dealloc__(self):
        pass

    def convert_game_as_csr_matrix(self, file_name, verbose=False):
        cdef game_state_t *game
        cdef SGFMoveIterator sgf_iter

        self.preprocessor = RolloutFeature(self.nakade_size,
                                           self.x33_size,
                                           self.d12_size)
        self.n_features = self.preprocessor.feature_size

        with open(file_name, 'r') as file_object:
            sgf_iter = SGFMoveIterator(self.bsize, file_object.read())

        game = sgf_iter.game
        tensors = {S_BLACK: lil_matrix((pure_board_max, self.n_features)),
                   S_WHITE: lil_matrix((pure_board_max, self.n_features))}
        # h5 0 dim error workaround. constant planes filled with 1
        tensors[S_BLACK][:, self.n_features-1] = 1
        tensors[S_WHITE][:, self.n_features-1] = 1
        try:
            for i, move in enumerate(sgf_iter):
                if move[0] != PASS and sgf_iter.next_move:
                    next_move = sgf_iter.next_move
                    current_tensor = tensors[next_move[1]]
                    self.preprocessor.update_lil(game, current_tensor)
                    self.preprocessor.update(game)
                    csr_tensor = current_tensor.tocsr()
                    yield (csr_tensor, next_move[0])
        except IllegalMove:
            warnings.warn('IllegalMove {:d}[{:d}] at {:d} in {:s}\n'.format(move[1], move[0], i, file_name))
            if verbose:
                err, msg, _ = sys.exc_info()
                sys.stderr.write("{} {}\n".format(err, msg))
                sys.stderr.write(traceback.format_exc())

    def convert_game_as_onehot_index_array(self, file_name, verbose=False):
        cdef game_state_t *game
        cdef SGFMoveIterator sgf_iter
        cdef rollout_feature_t *feature

        self.preprocessor = RolloutFeature(self.nakade_size,
                                           self.x33_size,
                                           self.d12_size)

        with open(file_name, 'r') as file_object:
            sgf_iter = SGFMoveIterator(self.bsize, file_object.read())

        game = sgf_iter.game
        try:
            for i, move in enumerate(sgf_iter):
                if move[0] != PASS and sgf_iter.next_move:
                    next_move = sgf_iter.next_move
                    self.preprocessor.update(game)
                    feature = &self.preprocessor.feature_planes[<int>game.current_color]
                    onehot_index_array = np.asarray(feature.tensor)
                    yield (onehot_index_array, next_move[0])
        except IllegalMove:
            warnings.warn('IllegalMove {:d}[{:d}] at {:d} in {:s}\n'.format(move[1], move[0], i, file_name))
            if verbose:
                err, msg, _ = sys.exc_info()
                sys.stderr.write("{} {}\n".format(err, msg))
                sys.stderr.write(traceback.format_exc())

    def sgfs_to_onehot_index_array(self,
                     sgf_files,
                     hdf5_file,
                     ignore_errors=True, verbose=False):
        tmp_file = os.path.join(os.path.dirname(hdf5_file), ".tmp." + os.path.basename(hdf5_file))
        h5f = h5py.File(tmp_file, 'w')

        try:
            states = h5f.require_dataset(
                'states',
                dtype=np.uint8,
                shape=(1, PURE_BOARD_MAX, 6),
                maxshape=(None, PURE_BOARD_MAX, 6),  # 'None' == arbitrary size
                exact=False,  # allow non-uint8 datasets to be loaded, coerced to uint8
                chunks=(64, PURE_BOARD_MAX, 6),  # approximately 1MB chunks
                compression="lzf")
            actions = h5f.require_dataset(
                'actions',
                dtype=np.uint8,
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
            for file_name in tqdm(sgf_files):
                if verbose:
                    print(file_name)
                n_pairs = 0
                file_start_idx = next_idx
                try:
                    for state, move in self.convert_game_as_onehot_index_array(file_name):
                        if next_idx >= len(states):
                            states.resize((next_idx + 1, PURE_BOARD_MAX, 6))
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

        if verbose:
            print("finished. renaming %s to %s" % (tmp_file, hdf5_file))

        # processing complete; rename tmp_file to hdf5_file
        h5f.close()
        os.rename(tmp_file, hdf5_file)

    def sgfs_to_csr_matrix(self,
                     sgf_files,
                     hdf5_file,
                     ignore_errors=False, verbose=True):
        h5 = tables.open_file(hdf5_file, mode="w")

        state_root = h5.create_group(h5.root, 'state')
        data_root = h5.create_group(state_root, 'data')
        indices_root = h5.create_group(state_root, 'indices')
        indptr_root = h5.create_group(state_root, 'indptr')

        action_root = h5.create_group(h5.root, 'action')

        filters = tables.Filters(complevel=5, complib="zlib")

        group_size = 100
        h5.set_node_attr(h5.root, 'group_size', group_size)
        try:
            if verbose:
                print("created HDF5 dataset in {}".format(hdf5_file))

            next_idx = 0
            n_parse_error = 0
            n_not19 = 0
            n_too_few_move = 0
            n_too_many_move = 0
            for file_name in tqdm(sgf_files):
                try:
                    n_pairs = 0
                    for state, move in self.convert_game_as_csr_matrix(file_name):
                        if next_idx % group_size == 0:
                            group_id = 'g' + str(next_idx/group_size).rjust(5, '0')
                            current_data = h5.create_group(data_root, group_id,
                                                           filters=filters)
                            current_indices = h5.create_group(indices_root, group_id,
                                                              filters=filters)
                            current_indptr = h5.create_group(indptr_root, group_id,
                                                             filters=filters)
                            current_action = h5.create_group(action_root, group_id,
                                                             filters=filters)

                        name = "s" + str(next_idx).rjust(8, '0')
                        data = state.data
                        data_atom = tables.Atom.from_dtype(data.dtype)
                        data_store = h5.create_carray(current_data, name, data_atom, data.shape)
                        data_store[:] = data

                        indices = state.indices
                        indices_atom = tables.Atom.from_dtype(indices.dtype)
                        indices_store = h5.create_carray(current_indices, name, indices_atom, indices.shape)
                        indices_store[:] = indices

                        indptr = state.indptr
                        indptr_atom = tables.Atom.from_dtype(indptr.dtype)
                        indptr_store = h5.create_carray(current_indptr, name, indptr_atom, indptr.shape)
                        indptr_store[:] = indptr

                        action = np.array((move,))
                        action_atom = tables.Atom.from_dtype(action.dtype)
                        action_store = h5.create_carray(current_action, name, action_atom, action.shape)
                        action_store[:] = move

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
                        if verbose:
                            print("\t%d state/action pairs extracted" % n_pairs)
                    elif verbose:
                        print("\t-no usable data-")
            h5.set_node_attr(h5.root, 'size', next_idx-1)
        except Exception as e:
            print("sgfs_to_csr_matrix failed")
            err, msg, _ = sys.exc_info()
            sys.stderr.write("{} {}\n".format(err, msg))
            sys.stderr.write(traceback.format_exc())
            os.remove(hdf5_file)
            raise e

        if verbose:
            print("finished.")

        print('Total {:d}/{:d} (Not19 {:d} ParseErr {:d} TooFewMove {:d} TooManyMove {:d})'.format(
            len(sgf_files) - n_parse_error - n_not19 - n_too_few_move - n_too_many_move,
            len(sgf_files)+1,
            n_parse_error,
            n_not19,
            n_too_few_move,
            n_too_many_move))

        h5.close()

