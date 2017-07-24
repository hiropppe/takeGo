from bamboo.mcts import test_mcts as ctest

import os
import unittest


class TestMCTS(unittest.TestCase):
    @classmethod
    def setUpClass(clazz):
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto(device_count={"GPU": 0})
        #config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.2
        set_session(tf.Session(config=config))

        d = os.path.dirname(os.path.abspath(__file__))
        # setup supervised policy
        model = os.path.join(d, '../params/policy/policy.json')
        weights = os.path.join(d, '../params/policy/weights.00088.hdf5')
        ctest.setup_supervised_policy(model, weights)
        # setup rollout policy
        rollout_weights = os.path.join(d, '../params/rollout/sample.hdf5')
        ctest.setup_rollout_policy(rollout_weights)


    @classmethod
    def tearDownClass(clazz):
        pass

    def setUp(self):
        ctest.setup()

    def tearDown(self):
        ctest.teardown()

    def test_seek_root(self):
        ctest.test_seek_root()

    def test_expand(self):
        ctest.test_expand()

    def test_select(self):
        ctest.test_select()

    def test_eval_leafs_by_policy_network(self):
        ctest.test_eval_leafs_by_policy_network()
