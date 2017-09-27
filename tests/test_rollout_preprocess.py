from bamboo import test_rollout_preprocess as ctest

import unittest


class TestRolloutPreprocess(unittest.TestCase):
    @classmethod
    def setUpClass(clazz):
        ctest.setup()

    @classmethod
    def tearDownClass(clazz):
        ctest.teardown()

    def test_update_self_atari(self):
        ctest.test_update_self_atari()

    def test_update_last_move_distance(self):
        ctest.test_update_last_move_distance()

    def test_update_nonres_12diamond(self):
        ctest.test_update_nonres_12diamond()

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

    def test_memorize_updated(self):
        ctest.test_memorize_updated()

    def test_choice_rollout_move(self):
        ctest.test_choice_rollout_move()

    def test_set_illegal(self):
        ctest.test_set_illegal()

    def test_copy_game(self):
        ctest.test_copy_game()
