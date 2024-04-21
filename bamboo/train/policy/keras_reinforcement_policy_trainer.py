import json
import numpy as np
import os
import time

import tensorflow as tf

from shutil import copyfile

from tensorflow import keras
from tensorflow.keras import backend as K

from bamboo.models.keras_dcnn_policy import KerasPolicy, cnn_policy

from bamboo.self_play_game import run_n_games


def log_loss(y_true, y_pred):
    '''Keras 'loss' function for the REINFORCE algorithm, where y_true is the action that was
    taken, and updates with the negative gradient will make that action more likely. We use the
    negative gradient because keras expects training data to minimize a loss function.
    '''
    return -y_true * K.log(K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon()))


def start_training(args):

    ZEROTH_FILE = "weights.00000.hdf5"

    if args.resume:
        if not os.path.exists(os.path.join(args.out_directory, "metadata.json")):
            raise ValueError("Cannot resume without existing output directory")

    if not os.path.exists(args.out_directory):
        if args.verbose:
            print("creating output directory {}".format(args.out_directory))
        os.makedirs(args.out_directory)

    if not args.resume:
        # make a copy of weights file, "weights.00000.hdf5" in the output directory
        copyfile(args.initial_weights, os.path.join(args.out_directory, ZEROTH_FILE))
        if args.verbose:
            print("copied {} to {}".format(args.initial_weights,
                                           os.path.join(args.out_directory, ZEROTH_FILE)))
        learner_weights = ZEROTH_FILE
    else:
        # if resuming, we expect initial_weights to be just a
        # "weights.#####.hdf5" file, not a full path
        args.initial_weights = os.path.join(args.out_directory,
                                            os.path.basename(args.initial_weights))
        if not os.path.exists(args.initial_weights):
            raise ValueError("Cannot resume; weights {} do not exist".format(args.initial_weights))
        elif args.verbose:
            print("Resuming with weights {}".format(args.initial_weights))
        learner_weights = os.path.basename(args.initial_weights)
    
    d = os.path.dirname(os.path.abspath(__file__))
    learner_model = cnn_policy()
    learner_model.load_weights(args.initial_weights)
    learner_policy = KerasPolicy(learner_model)
 
    opponent_model = cnn_policy()
    opponent_policy = KerasPolicy(opponent_model)

    if args.verbose:
        print("created player and opponent with temperature {}".format(args.policy_temp))
    
    if not args.resume:
        metadata = {
            "init_weights": args.initial_weights,
            "learning_rate": args.learning_rate,
            "temperature": args.policy_temp,
            "game_batch": args.game_batch,
            "opponents": [ZEROTH_FILE],  # which weights from which to sample an opponent each batch
            "win_ratio": {}  # map from player to tuple of (opponent, win ratio) Useful for
                             # validating in lieu of 'accuracy/loss'
        }
    else:
        with open(os.path.join(args.out_directory, "metadata.json"), "r") as f:
            metadata = json.load(f)

    # Append args of current run to history of full command args.
    metadata["cmd_line_args"] = metadata.get("cmd_line_args", [])
    metadata["cmd_line_args"].append(vars(args))

    def save_metadata():
        with open(os.path.join(args.out_directory, "metadata.json"), "w") as f:
            json.dump(metadata, f, sort_keys=True, indent=2)

    optimizer = keras.optimizers.SGD(learning_rate=args.learning_rate)
    learner_model.compile(loss=log_loss, optimizer=optimizer)

    for i_iter in range(1, args.iterations + 1):
        # Randomly choose opponent from pool (possibly self), and playing
        # game_batch games against them.
        opp_weights = np.random.choice(metadata["opponents"])
        opp_path = os.path.join(args.out_directory, opp_weights)

        # Load new weights into opponent's network, but keep the same opponent object.
        opponent_model.load_weights(opp_path)
        if args.verbose:
            print("Batch {}\tsampled opponent is {}".format(i_iter, opp_weights))

        state_tensors, move_tensors, learner_won, win_ratio = run_n_games(learner_policy, opponent_policy)
        print(f'winning ratio: {win_ratio:.2f}')
        # Train on each game's results, setting the learning rate negative to 'unlearn' positions from
        # games where the learner lost.
        for (st_tensor, mv_tensor, won) in zip(state_tensors, move_tensors, learner_won):
            print(len(st_tensor), len(mv_tensor), st_tensor[0].shape, mv_tensor[0].shape, won)
            # optimizer.lr = K.abs(optimizer.lr) * (+1 if won else -1)
            K.set_value(optimizer.lr, abs(args.learning_rate) * (+1 if won else -1))
            learner_model.train_on_batch(np.concatenate(st_tensor, axis=0),
                                 np.concatenate(mv_tensor, axis=0))

        metadata["win_ratio"][learner_weights] = (opp_weights, win_ratio)

        # Save intermediate models.
        if i_iter % args.record_every == 0:
            learner_weights = "weights.{:05d}.hdf5".format(i_iter)
            learner_model.save_weights(os.path.join(args.out_directory, learner_weights))

        # Add player to batch of oppenents once in a while.
        if i_iter % args.save_every == 0:
            metadata["opponents"].append(learner_weights)
        
        save_metadata()


def main(cmd_line_args=None):
    """Run training. command-line args may be passed in as a list
    """
    import argparse
    parser = argparse.ArgumentParser(description='Perform reinforcement learning to improve given policy network. Second phase of pipeline.')  # noqa: E501
    subparsers = parser.add_subparsers(help='sub-command help')

    train = subparsers.add_parser('train', help='Start or resume supervised training on a policy network.')  # noqa: E501
    train.add_argument("initial_weights", help="Path to HDF5 file with inital weights (i.e. result of supervised training).")  # noqa: E501
    train.add_argument("out_directory", help="Path to folder where the model params and metadata will be saved after each epoch.")  # noqa: E501
    train.add_argument("--learning-rate", help="Keras learning rate (Default: 0.001)", type=float, default=0.001)  # noqa: E501
    train.add_argument("--policy-temp", help="Distribution temperature of players using policies (Default: 0.67)", type=float, default=0.67)  # noqa: E501
    train.add_argument("--save-every", help="Save policy as a new opponent every n batches (Default: 500)", type=int, default=500)  # noqa: E501
    train.add_argument("--record-every", help="Save learner's weights every n batches (Default: 1)", type=int, default=1)  # noqa: E501
    train.add_argument("--game-batch", help="Number of games per mini-batch (Default: 20)", type=int, default=20)  # noqa: E501
    train.add_argument("--move-limit", help="Maximum number of moves per game", type=int, default=500)  # noqa: E501
    train.add_argument("--iterations", help="Number of training batches/iterations (Default: 10000)", type=int, default=1)  # noqa: E501
    train.add_argument("--resume", help="Load latest weights in out_directory and resume", default=False, action="store_true")  # noqa: E501
    train.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501
    train.set_defaults(func=start_training)

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    args = {
        'initial_weights': './test_data/rl_policy/kihuu.hdf5',
        'out_directory': './test_training/rl_policy/',
        'learning_rate': 0.001,
        'policy_temp': 0.67,
        'save_every': 1,
        'record_every': 1, 
        'game_batch': 2,
        'move_limit': 500,
        'iterations': 2,
        'resume': False, 
        'verbose': True,  # Turn on verbose mode
    }

    from types import SimpleNamespace
    start_args = SimpleNamespace(**args)
    start_training(start_args)

    # execute function (train or resume)
    #args.func(args)


if __name__ == '__main__':
    main()