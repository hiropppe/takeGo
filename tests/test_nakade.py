from bamboo.rollout import test_nakade as ctest

import unittest


class TestNakade(unittest.TestCase):
    @classmethod
    def setUpClass(clazz):
        ctest.setup()

    def test_nakade3(self):
        ctest.test_nakade3()

    def test_nakade4(self):
        ctest.test_nakade4()

    def test_nakade5(self):
        ctest.test_nakade5()

    def test_nakade6(self):
        ctest.test_nakade6()
