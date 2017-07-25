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

    def test_update_all_save_atari(self):
        ctest.test_update_all_save_atari()

    def test_update_all_neighbor(self):
        ctest.test_update_all_neighbor()

    def test_update_all_12diamond(self):
        ctest.test_update_all_12diamond()

    def test_update_all_3x3(self):
        ctest.test_update_all_3x3()

    def test_memorize_updated(self):
        ctest.test_memorize_updated()

    def test_choice_rollout_move(self):
        ctest.test_choice_rollout_move()
