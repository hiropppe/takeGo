#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import warnings
import sgf
import sys
import traceback
import tables

from scipy.sparse import lil_matrix, csr_matrix
from tqdm import tqdm

from bamboo.util import SizeMismatchError
from bamboo.util cimport SGFMoveIterator
from bamboo.go.board cimport PASS, POS, CORRECT_X, CORRECT_Y, game_state_t, pure_board_size, set_board_size, pure_board_max, free_game
from bamboo.rollout.pattern cimport init_nakade, init_x33, init_d12
from bamboo.rollout.preprocess cimport RolloutFeature
from bamboo.go.printer cimport print_board


cdef class GameConverter(object):

    cdef:
        RolloutFeature preprocessor
        int bsize
        int nakade_size, x33_size, d12_size
        int n_features

    def __cinit__(self, bsize=19, nakade_file=None, x33_file=None, d12_file=None):
        self.bsize = bsize

        self.nakade_size = init_nakade(nakade_file)
        self.x33_size = init_x33(x33_file)
        self.d12_size = init_d12(d12_file)

        self.preprocessor = RolloutFeature(self.nakade_size,
                                           self.x33_size,
                                           self.d12_size)
        self.n_features = self.preprocessor.feature_size

    def __dealloc__(self):
        pass

    def convert_game(self, file_name):
        """Read the given SGF file into an iterable of (input,output) pairs
        for neural network training

        Each input is an one-hot neural net features
        Each output is an action as an (x,y) pair (passes are skipped)

        If this game's size does not match bsize, a SizeMismatchError is raised
        """
        cdef game_state_t *game
        cdef SGFMoveIterator sgf_iter

        with open(file_name, 'r') as file_object:
            sgf_iter = SGFMoveIterator(file_object.read())

        game = sgf_iter.game
        #import time
        lil_tensor = lil_matrix((pure_board_max, self.n_features))
        for move in sgf_iter:
            #print_board(game)
            if sgf_iter.next_move != PASS:
                #s = time.time()
                self.preprocessor.update_lil(game, lil_tensor)
                #print('{:.3f} us'.format((time.time()-s)*1000*1000))
                csr_tensor = lil_tensor.tocsr()
                move = sgf_iter.next_move
                yield (csr_tensor, move)

    def sgfs_to_hdf5(self,
                     sgf_files,
                     hdf5_file,
                     ignore_errors=False, verbose=True):
        """Convert all files in the iterable sgf_files into an hdf5 group to be stored in hdf5_file
        """
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
            for file_name in tqdm(sgf_files):
                try:
                    n_pairs = 0
                    for state, move in self.convert_game(file_name):
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
                        #action = move
                        action_atom = tables.Atom.from_dtype(action.dtype)
                        action_store = h5.create_carray(current_action, name, action_atom, action.shape)
                        action_store[:] = move

                        n_pairs += 1
                        next_idx += 1
                except sgf.ParseException:
                    warnings.warn("Could not parse %s\n\tdropping game" % file_name)
                except SizeMismatchError:
                    warnings.warn("Skipping %s; wrong board size" % file_name)
                except Exception as e:
                    # catch everything else
                    if ignore_errors:
                        warnings.warn("Unkown exception with file %s\n\t%s" % (file_name, e),
                                      stacklevel=2)
                    else:
                        raise e
                finally:
                    if n_pairs > 0:
                        if verbose:
                            print("\t%d state/action pairs extracted" % n_pairs)
                    elif verbose:
                        print("\t-no usable data-")
            h5.set_node_attr(h5.root, 'size', next_idx-1)
        except Exception as e:
            print("sgfs_to_hdf5 failed")
            os.remove(hdf5_file)
            raise e

        if verbose:
            print("finished.")

        h5.close()

