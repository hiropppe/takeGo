from bamboo.go import test_ladder

import unittest


class TestLadder(unittest.TestCase):
    def test_captured_1(self):
        test_ladder.test_captured_1()

    def test_breaker_1(self):
        test_ladder.test_breaker_1()

    def test_missing_ladder_breaker_1(self):
        test_ladder.test_missing_ladder_breaker_1()

    def test_missing_ladder_breaker_2(self):
        # BBS(MCTS-CNN) put 'b' at after 'a' in actual game
        test_ladder.test_missing_ladder_breaker_2()

    def test_capture_to_escape_1(self):
        test_ladder.test_capture_to_escape_1()

    def test_throw_in_1(self):
        test_ladder.test_throw_in_1()

    def test_snapback_1(self):
        test_ladder.test_snapback_1()

    def test_two_captures(self):
        test_ladder.test_two_captures()

    def test_two_escapes(self):
        test_ladder.test_two_escapes()

    def test_escapes_1(self):
        test_ladder.test_escapes_1()

    def test_escapes_require_many_moves(self):
        test_ladder.test_escapes_require_many_moves()

    def test_captured_require_many_moves(self):
        test_ladder.test_captured_require_many_moves()

    def test_captured_2(self):
        test_ladder.test_captured_2()

    def test_escape_segmentation_fault_1(self):
        test_ladder.test_escape_segmentation_fault_1()

    def test_capture_segmentation_fault_1(self):
        test_ladder.test_capture_segmentation_fault_1()
