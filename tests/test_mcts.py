from bamboo.mcts import test_mcts as ctest

import os
import unittest


class TestMCTS(unittest.TestCase):
    @classmethod
    def setUpClass(clazz):
        d = os.path.dirname(os.path.abspath(__file__))
        model = os.path.join(d, '../params/policy/policy.json')
        weights = os.path.join(d, '../params/policy/weights.00088.hdf5')

        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto(device_count={"GPU": 0})
        #config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.2
        set_session(tf.Session(config=config))

        ctest.setup_class(model, weights)

    @classmethod
    def tearDownClass(clazz):
        ctest.teardown_class()

    def setUp(self):
        ctest.setup()

    def tearDown(self):
        ctest.teardown()

    def test_seek_root(self):
        ctest.test_seek_root()

    def test_expand(self):
        ctest.test_expand()

    def test_eval_leafs_by_policy_network(self):
        ctest.test_eval_leafs_by_policy_network()
