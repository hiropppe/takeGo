import numpy as np
import tensorflow as tf

from bamboo.models.dcnn_resnet_value import inference_agz

cimport numpy as np

from libc.stdlib cimport malloc, free

from bamboo.policy_feature cimport PolicyFeature
from bamboo.policy_feature cimport allocate_feature, initialize_feature, update
from bamboo.tree_search cimport tree_node_t
from bamboo.sgf_util cimport SGFMoveIterator
from bamboo.printer cimport print_board


def print_value(vn_path, sgf, moves=[4, 5, 50, 51, 100, 101, 150, 151, 250, 251, 300, 351]):
    cdef tree_node_t *node
    cdef PolicyFeature feature = allocate_feature(49)
    cdef SGFMoveIterator sgf_iterator
    cdef object sess

    with tf.Graph().as_default() as graph:
        inputs = tf.placeholder(tf.float32, [None, 49, 19, 19])
        op = inference_agz(inputs, is_training=True)

        sess = tf.Session(graph=graph)
        # init_op = tf.global_variables_initializer()
        # sess.run(init_op)
        saver = tf.train.Saver()
        saver.restore(sess, vn_path)

        # print tf.global_variables()
        # eval = tf.global_variables()[0].eval

    # print eval(session=sess)

    #sgf = '../self_play/aya/203_19_0114_2k_r16_add300_1/20170319_0400_17784.sgf'

    node = <tree_node_t *>malloc(sizeof(tree_node_t))
    initialize_feature(feature)

    with open(sgf, 'r') as fo:
        sgf_iterator =  SGFMoveIterator(19, fo.read(), ignore_no_result=False)

    node.game = sgf_iterator.game
    for i, move in enumerate(sgf_iterator):
        if i in moves:
            update(feature, node)

            tensor = np.asarray(feature.planes).reshape(1, 49, 19, 19)

            # print(tensor)
            print_board(sgf_iterator.game)

            print('color:{:d} move:{:d} >> VN: {:.3f}'.format(sgf_iterator.game.current_color, i, sess.run(op, feed_dict={inputs: tensor})[0][0]))

