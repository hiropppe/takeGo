# coding: utf-8

import pyjsonrpc

mcts = None


def set_mcts(new_mcts):
    global mcts
    mcts = new_mcts


class MCTSRequestHandler(pyjsonrpc.HttpRequestHandler):

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
    def set_time_settings(self, main_time, byoyomi_time, byoyomi_stones):
        mcts.set_time_settings(main_time, byoyomi_time, byoyomi_stones)

    @pyjsonrpc.rpcmethod
    def set_time_left(self, color, time, stones):
        mcts.set_time_left(color, time, stones)

    @pyjsonrpc.rpcmethod
    def set_const_time(self, limit):
        mcts.set_const_time(limit)

    @pyjsonrpc.rpcmethod
    def set_const_playout(self, limit):
        mcts.set_const_playout(limit)

    @pyjsonrpc.rpcmethod
    def showboard(self):
        mcts.showboard()

    @pyjsonrpc.rpcmethod
    def save_sgf(self, black_name, white_name):
        return mcts.save_sgf(black_name, white_name)

    @pyjsonrpc.rpcmethod
    def quit(self):
        mcts.quit()
