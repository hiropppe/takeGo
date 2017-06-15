from bamboo.rollout import test_x33_pattern as ctest

import unittest


class TestX33Pattern(unittest.TestCase):
    def test_x33_bit(self):
        ctest.test_x33_bit()

    def test_x33_hash_from_bits_0(self):
        ctest.test_x33_hash_from_bits_0()

    def test_x33_hash_from_bits_1(self):
        ctest.test_x33_hash_from_bits_1()
