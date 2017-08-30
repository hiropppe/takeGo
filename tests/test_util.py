from bamboo import test_util as ctest

import unittest


class TestUtil(unittest.TestCase):
    def test_save_gamestate_to_sgf(self):
        ctest.test_save_gamestate_to_sgf()
