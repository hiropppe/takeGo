#!/usr/bin/env python

import numpy

from setuptools import setup, find_packages

from distutils import core
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

# Remove the "-Wstrict-prototypes" compiler option, which isn't valid for C++.
import distutils.sysconfig
cfg_vars = distutils.sysconfig.get_config_vars()
for key, value in cfg_vars.items():
    if type(value) == str:
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "")
# ==================================

extensions = [Extension('bamboo.board', sources=['bamboo/board.pyx'], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.zobrist_hash', sources=["bamboo/zobrist_hash.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.pattern', sources=["bamboo/pattern.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.printer', sources=["bamboo/printer.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.parseboard', sources=["bamboo/parseboard.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.tree_search', sources=["bamboo/tree_search.pyx"], language="c++", extra_compile_args=["-std=c++11", "-fopenmp"], extra_link_args=['-lgomp']),
              Extension('bamboo.seki', sources=["bamboo/seki.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.nakade', sources=["bamboo/nakade.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.local_pattern', sources=["bamboo/local_pattern.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.policy_feature', sources=["bamboo/policy_feature.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.rollout_preprocess', sources=["bamboo/rollout_preprocess.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.sgf_util', sources=["bamboo/sgf_util.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.train.dataset_util', sources=["bamboo/train/dataset_util.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.train.policy.sgf2tfrecord', sources=["bamboo/train/policy/v1/sgf2tfrecord.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.train.policy.sgf2hdf5', sources=["bamboo/train/policy/sgf2hdf5.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.train.value.sgf2tfrecord', sources=["bamboo/train/value/sgf2tfrecord.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.train.value.sgf_debug', sources=["bamboo/train/value/sgf_debug.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.train.rollout.pattern_harvest', sources=["bamboo/train/rollout/pattern_harvest.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.train.rollout.sgf2hdf5', sources=["bamboo/train/rollout/sgf2hdf5.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.train.rollout.sgf2hdf5_tree', sources=["bamboo/train/rollout/sgf2hdf5_tree.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.train.rollout.print_acc_for_sgf', sources=["bamboo/train/rollout/print_acc_for_sgf.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.train.rollout.print_tree_acc_for_sgf', sources=["bamboo/train/rollout/print_tree_acc_for_sgf.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.gtp.gtp_connector', sources=["bamboo/gtp/gtp_connector.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.self_play', sources=["bamboo/self_play.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.debug_for_sgf', sources=["bamboo/debug_for_sgf.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.test_board', sources=["bamboo/test_board.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.test_ladder', sources=["bamboo/test_ladder.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.test_policy_feature', sources=["bamboo/test_policy_feature.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.test_seki', sources=["bamboo/test_seki.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.test_nakade', sources=["bamboo/test_nakade.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.test_x33_pattern', sources=["bamboo/test_x33_pattern.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.test_d12_pattern', sources=["bamboo/test_d12_pattern.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.test_rollout_preprocess', sources=["bamboo/test_rollout_preprocess.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.test_mcts', sources=["bamboo/test_mcts.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.test_sgf_util', sources=["bamboo/test_sgf_util.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.player', sources=["bamboo/player.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.self_play_game', sources=["bamboo/self_play_game.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              Extension('bamboo.test_cython', sources=["bamboo/test_cython.pyx"], language="c++", extra_compile_args=["-std=c++11"]),
              ]

core.setup(
  ext_modules=cythonize(extensions, compiler_directives={'language_level' : "3"}),
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
