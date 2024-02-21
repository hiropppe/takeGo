import os

from bamboo import test_mcts as ctest


class TestMCTS():

    @classmethod
    def setup_class(clazz):
        import tensorflow as tf
        from tensorflow.python.keras import backend as K

        # config = tf.compat.v1.ConfigProto(device_count={"GPU": 0})
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.2
        K.set_session(tf.compat.v1.Session(config=config))

        d = os.path.dirname(os.path.abspath(__file__))
        # setup supervised policy
        pn_path = os.path.join(d, '../params/policy/weights.hdf5')
        ctest.setup_supervised_policy(pn_path, nogpu=True)
        # setup rollout policy
        rollout_weights = os.path.join(d, '../params/rollout/rollout.hdf5')
        ctest.setup_rollout_policy(rollout_weights)
        # setup pattern hash for rollout
        rands_file = os.path.join(d, '../params/rollout/mt_rands.txt')
        d12_file = os.path.join(d, '../params/rollout/d12.csv')
        x33_file = os.path.join(d, '../params/rollout/x33.csv')
        ctest.setup_pattern(rands_file, d12_file, x33_file)

    @classmethod
    def teardown_class(clazz):
        pass

    def setup_method(self, method):
        ctest.setup()

    def teardown_method(self, method):
        ctest.teardown()

    def test_seek_root(self):
        ctest.test_seek_root()

    def test_expand(self):
        ctest.test_expand()

    def test_select(self):
        #ctest.test_select()
        pass

    def test_eval_leaf_by_policy_network(self):
        ctest.test_eval_leaf_by_policy_network()

    def test_rollout(self):
        ctest.test_rollout()
