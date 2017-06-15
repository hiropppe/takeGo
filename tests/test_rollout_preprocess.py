from bamboo.rollout import test_rollout_preprocess as ctest

import unittest


class TestRolloutPreprocess(unittest.TestCase):
    @classmethod
    def setUpClass(clazz):
        ctest.setup()

    @classmethod
    def tearDownClass(clazz):
        ctest.teardown() 

    def test_update_0(self):
        ctest.test_update_0()
