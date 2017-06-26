from bamboo.rollout import test_rollout_preprocess as ctest

import unittest


class TestRolloutPreprocess(unittest.TestCase):
    @classmethod
    def setUpClass(clazz):
        ctest.setup()

    @classmethod
    def tearDownClass(clazz):
        ctest.teardown()

    def test_update_12diamond_0(self):
        ctest.test_update_12diamond_0()

    def test_update_12diamond_after_pass_0(self):
        ctest.test_update_12diamond_after_pass_0()

    def test_update_3x3_0(self):
        ctest.test_update_3x3_0()
