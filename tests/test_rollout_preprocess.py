from bamboo.rollout import test_rollout_preprocess as ctest

import unittest


class TestRolloutPreprocess(unittest.TestCase):
    @classmethod
    def setUpClass(clazz):
        ctest.setup()

    @classmethod
    def tearDownClass(clazz):
        ctest.teardown()

    def test_update_save_atari(self):
        ctest.test_update_save_atari()

    def test_update_save_atari_connect_string(self):
        ctest.test_update_save_atari_connect_string()

    def test_update_save_atari_not_escape(self):
        ctest.test_update_save_atari_not_escape()

    def test_update_save_atari_not_escape_on_edge(self):
        ctest.test_update_save_atari_not_escape_on_edge()

    def test_update_neighbor_0(self):
        ctest.test_update_neighbor_0()

    def test_update_12diamond_0(self):
        ctest.test_update_12diamond_0()

    def test_update_12diamond_after_pass_0(self):
        ctest.test_update_12diamond_after_pass_0()

    def test_update_3x3_0(self):
        ctest.test_update_3x3_0()
