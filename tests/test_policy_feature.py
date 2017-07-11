from bamboo.go import test_policy_feature as ctest

import unittest


class TestPolicyFeature(unittest.TestCase):
    def test_stone_color(self):
        ctest.test_stone_color()

    def test_turns_since(self):
        ctest.test_turns_since()

    def test_liberties(self):
        ctest.test_liberties()

    def test_capture_size(self):
        ctest.test_capture_size()

    def test_self_atari_size(self):
        ctest.test_self_atari_size()

    def test_liberties_after_move(self):
        ctest.test_liberties_after_move()

    def test_liberties_after_move_1(self):
        ctest.test_liberties_after_move_1()

    def test_liberties_after_move_dupe_empty(self):
        ctest.test_liberties_after_move_dupe_empty()

    def test_liberties_after_move_captured(self):
        ctest.test_liberties_after_move_captured()

    def test_liberties_after_move_captured_1(self):
        ctest.test_liberties_after_move_captured_1()

    def test_liberties_after_move_captured_2(self):
        ctest.test_liberties_after_move_captured_2()

    def test_sensibleness_not_suicide(self):
        ctest.test_sensibleness_not_suicide()

    def test_sensibleness_true_eye(self):
        ctest.test_sensibleness_true_eye()

    def test_sensibleness_not_true_eye(self):
        ctest.test_sensibleness_not_true_eye()

    def test_sensibleness_true_eye_remove_stone(self):
        ctest.test_sensibleness_true_eye_remove_stone()
