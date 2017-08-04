# coding: utf-8

import pyjsonrpc

#from bamboo.mcts.tree_search import PyMCTS

mcts = None


def set_mcts(new_mcts):
    global mcts
    mcts = new_mcts


class MCTSRequestHandler(pyjsonrpc.HttpRequestHandler):

    """
    def __init__(self,
                  policy,
                  temperature=0.67,
                  playout_limit=8000,
                  n_threads=1):
        self.mcts = PyMCTS(policy, temperature, playout_limit, n_threads)
    """

    @pyjsonrpc.rpcmethod
    def clear(self):
        mcts.clear()

    @pyjsonrpc.rpcmethod
    def start_pondering(self):
        mcts.start_pondering()

    @pyjsonrpc.rpcmethod
    def stop_pondering(self):
        mcts.stop_pondering()

    @pyjsonrpc.rpcmethod
    def genmove(self, color):
        return mcts.genmove(color)

    @pyjsonrpc.rpcmethod
    def play(self, pos, color):
        return mcts.play(pos, color)

    @pyjsonrpc.rpcmethod
    def start_policy_network_queue(self):
        mcts.start_policy_network_queue()

    @pyjsonrpc.rpcmethod
    def stop_policy_network_queue(self):
        mcts.stop_policy_network_queue()

    @pyjsonrpc.rpcmethod
    def eval_all_leafs_by_policy_network(self):
        mcts.eval_all_leafs_by_policy_network()

    @pyjsonrpc.rpcmethod
    def set_size(self, bsize):
        mcts.set_size(bsize)

    @pyjsonrpc.rpcmethod
    def set_komi(self, new_komi):
        mcts.set_komi(new_komi)

    @pyjsonrpc.rpcmethod
    def set_time(self, m, b, stone):
        pass

    @pyjsonrpc.rpcmethod
    def set_time_left(self, color, time, stone):
        pass

    @pyjsonrpc.rpcmethod
    def set_playout_limit(self, limit):
        mcts.set_playout_limit(limit)

    @pyjsonrpc.rpcmethod
    def showboard(self):
        mcts.showboard()

    @pyjsonrpc.rpcmethod
    def save_sgf(self, black_name, white_name):
        return mcts.save_sgf(black_name, white_name)

    @pyjsonrpc.rpcmethod
    def quit(self):
        mcts.quit()
