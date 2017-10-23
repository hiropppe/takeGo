from bamboo import test_d12_pattern as ctest

import unittest


class Test12DiamondPattern(unittest.TestCase):
    def test_d12_rsp_bits_0(self):
        ctest.test_d12_rsp_bits_0()

    def test_d12_rsp_hash_0(self):
        ctest.test_d12_rsp_hash_0()

    def test_d12_rspos_bits_0(self):
        ctest.test_d12_rspos_bits_0()

    def test_d12_rspos_hash_0(self):
        ctest.test_d12_rspos_hash_0()
