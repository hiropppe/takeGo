from bamboo import test_seki as ctest

import unittest


class TestSeki(unittest.TestCase):
    @classmethod
    def setUpClass(clazz):
        ctest.setup()

    def test_seki_0(self):
        ctest.test_seki_0()

    def test_seki_1(self):
        ctest.test_seki_1()

    def test_seki_2(self):
        ctest.test_seki_2()

    def test_seki_3(self):
        ctest.test_seki_3()

    def test_seki_4(self):
        ctest.test_seki_4()

    def test_seki_5(self):
        ctest.test_seki_5()
