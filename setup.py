#!/usr/bin/env python

import numpy

from setuptools import setup, find_packages

from distutils import core
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize


extensions = [Extension('bamboo.go.board', sources=['bamboo/go/board.pyx'], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.go.zobrist_hash', sources=["bamboo/go/zobrist_hash.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.go.pattern', sources=["bamboo/go/pattern.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.go.printer', sources=["bamboo/go/printer.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.go.policy_feature', sources=["bamboo/go/policy_feature.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.go.sgf2hdf5', sources=["bamboo/go/sgf2hdf5.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.go.parseboard', sources=["bamboo/go/parseboard.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.util', sources=["bamboo/util.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.merge_dataset', sources=["bamboo/merge_dataset.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.mcts.tree_search', sources=["bamboo/mcts/tree_search.pyx"], language="c++", extra_compile_args=["-std=c++11", "-fopenmp"], extra_link_args=['-lgomp']),
              Extension('bamboo.rollout.sgf2hdf5', sources=["bamboo/rollout/sgf2hdf5.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.rollout.sgf2hdf5_tree', sources=["bamboo/rollout/sgf2hdf5_tree.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.rollout.pattern', sources=["bamboo/rollout/pattern.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.rollout.preprocess', sources=["bamboo/rollout/preprocess.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.rollout.pattern_harvest', sources=["bamboo/rollout/pattern_harvest.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.gtp.gtp_connector', sources=["bamboo/gtp/gtp_connector.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.go.test_board', sources=["bamboo/go/test_board.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.go.test_ladder', sources=["bamboo/go/test_ladder.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.go.test_policy_feature', sources=["bamboo/go/test_policy_feature.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.rollout.test_x33_pattern', sources=["bamboo/rollout/test_x33_pattern.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.rollout.test_d12_pattern', sources=["bamboo/rollout/test_d12_pattern.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.rollout.test_rollout_preprocess', sources=["bamboo/rollout/test_rollout_preprocess.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.mcts.self_play', sources=["bamboo/mcts/self_play.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.mcts.test_mcts', sources=["bamboo/mcts/test_mcts.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.test_cyboard_eq_pyboard', sources=["bamboo/test_cyboard_eq_pyboard.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.test_util', sources=["bamboo/test_util.pyx"], language="c++", extra_compile_args=["-std=c++11"])]

core.setup(
  ext_modules=cythonize(extensions),
  include_dirs=[numpy.get_include(), 'bamboo/include']
)

requires = [
]

setup(
  name='bambooStone',
  version='0.0.1',
  author='take',
  url='',
  packages=find_packages(),
  scripts=[
    'bbrpc',
    'bbs',
    'bbc'
  ],
  install_requires=requires,
  license='MIT',
  test_suite='test',
  classifiers=[
    'Operating System :: OS Independent',
    'Environment :: Console',
    'Programming Language :: Python',
    'License :: OSI Approved :: MIT License',
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Information Technology',
    'Intended Audience :: Science/Research',
    'Topic :: Utilities',
  ],
  data_files=[
  ]
)
