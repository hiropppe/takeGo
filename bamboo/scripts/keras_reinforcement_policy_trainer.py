import json
import numpy as np
import os
import time

import tensorflow as tf
import tensorflow_probability as tfp

from shutil import copyfile

from tensorflow import keras
from tensorflow.keras import backend as K

from bamboo.models.keras_dcnn_policy import KerasPolicy, cnn_policy

from bamboo.self_play_game import run_n_games

np.set_printoptions(suppress=True, linewidth=200, precision=3)


def start_training(args):

    if args.resume:
        if not os.path.exists(os.path.join(args.out_directory, "metadata.json")):
            raise ValueError("Cannot resume without existing output directory")

    if not os.path.exists(args.out_directory):
        if args.verbose:
            print("creating output directory {}".format(args.out_directory))
        os.makedirs(args.out_directory)

    if not args.resume:
        # make a copy of weights file, "weights.00000.hdf5" in the output directory
        if args.initial_weights.endswith(".weights.h5"):
            ZEROTH_FILE = "sl_policy.weights.h5"
        else:
            ZEROTH_FILE = "sl_policy.h5"

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
    
    learner_model = cnn_policy()
    learner_model.load_weights(os.path.join(args.out_directory, learner_weights))
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

    for i_iter in range(1, args.iterations + 1):
        # Randomly choose opponent from pool (possibly self), and playing
        # game_batch games against them.
        opp_weights = np.random.choice(metadata["opponents"])
        opp_path = os.path.join(args.out_directory, opp_weights)

        # Load new weights into opponent's network, but keep the same opponent object.
        opponent_model.load_weights(opp_path)
        if args.verbose:
            print("Batch {}\tsampled opponent is {}".format(i_iter, opp_weights))
        
        state_tensors, move_tensors, learner_won, win_ratio = run_n_games(learner_policy,
                                                                          opponent_policy,
                                                                          n_games=args.game_batch,
                                                                          move_limit=args.move_limit,
                                                                          temperature=args.policy_temp,
                                                                          greedy=args.greedy,
                                                                          verbose=args.verbose)
        print(f'iter.{i_iter} winning ratio: {win_ratio*100:.3f}')
        # Train on each game's results, setting the learning rate negative to 'unlearn' positions from
        # games where the learner lost.
        #learna(state_tensors, move_tensors, learner_won, learner_model, tf.keras.losses.categorical_crossentropy, optimizer)
        learnb(state_tensors, move_tensors, learner_won, learner_model, optimizer)

        metadata["win_ratio"][learner_weights] = (opp_weights, win_ratio)

        # Save intermediate models.
        if i_iter % args.record_every == 0:
            learner_weights = "{:05d}.weights.h5".format(i_iter)
            learner_model.save_weights(os.path.join(args.out_directory, learner_weights))

        # Add player to batch of oppenents once in a while.
        if i_iter % args.save_every == 0:
            metadata["opponents"].append(learner_weights)
        
        save_metadata()


def learna(state_tensors,
           move_tensors,
           learner_won,
           model: tf.keras.Model,
           loss_fn: tf.keras.losses.Loss,
           optimizer: tf.keras.optimizers.Optimizer):
    game_batch = len(state_tensors)
    grads = None
    for (st_tensor, mv_tensor, won) in zip(state_tensors, move_tensors, learner_won):
        try:
            st_tensor = tf.cast(tf.constant(np.concatenate(st_tensor, axis=0)), tf.float32)
            mv_tensor = tf.cast(tf.constant(np.concatenate(mv_tensor, axis=0)), tf.float32)
        except ValueError as e:
            print(st_tensor, mv_tensor)
            continue
        z = +1 if won else -1
            
        game_grads = train_step(st_tensor, mv_tensor, model, loss_fn, optimizer) 

        if grads:
            for i, g in enumerate(game_grads):
                grads[i] += z*g/game_batch
        else:
            grads = [z*g/game_batch for g in game_grads]
        
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


@tf.function
def train_step(states: tf.Tensor,
               moves: tf.Tensor,
               model: tf.keras.Model,
               loss_fn: tf.keras.losses.Loss,
               optimizer: tf.keras.optimizers.Optimizer):
    with tf.GradientTape() as tape:
        tape.watch(states)
        output = model(states)
        loss = tf.reduce_mean(loss_fn(moves, output))
    grads = tape.gradient(loss, model.trainable_variables)
    return grads


def learnb(state_tensors,
           move_tensors,
           learner_won,
           model: tf.keras.Model,
           optimizer: tf.keras.optimizers.Optimizer):
    game_batch = len(state_tensors)
    grads = None
    loss = 0
    for (st_tensor, mv_tensor, won) in zip(state_tensors, move_tensors, learner_won):
        st_tensor = tf.cast(tf.constant(np.concatenate(st_tensor, axis=0)), tf.float32)
        mv_tensor = tf.cast(tf.constant(np.concatenate(mv_tensor, axis=0)), tf.float32)
        z = tf.constant(+1.0 if won else -1.0)
            
        game_loss, game_grads = compute_game_grads(st_tensor, mv_tensor, z, model, optimizer)
        loss += game_loss

        if grads:
            for i, g in enumerate(game_grads):
                grads[i] += g/game_batch
        else:
            grads = [g/game_batch for g in game_grads]

    global_norm = tf.linalg.global_norm(grads)
    print(f"Loss: {loss/game_batch} Grads norm: {global_norm}")
    ##grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


@tf.function
def compute_game_grads(states: tf.Tensor,
                       moves: tf.Tensor,
                       z: tf.Tensor,
                       model: tf.keras.Model,
                       optimizer: tf.keras.optimizers.Optimizer):
    with tf.GradientTape() as tape:
        tape.watch(states)
        probs = model(states)
        probs = probs * moves

        # We set 1 in the position of probability 0 so that the log probability is 0 instead of nan
        safe_probs = tf.where(tf.equal(probs, 0.), tf.ones_like(probs), probs)
        log_probs = tf.math.log(safe_probs)

        # We take the average of step loss (-log(p(a|s))) since the number of moves varies from game to game
        loss = -tf.reduce_mean(tf.reduce_sum(log_probs, axis=1)) * z
        #loss = -tf.reduce_mean(log_probs) * z

        grads = tape.gradient(loss, model.trainable_variables)

    return loss, grads


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
    train.add_argument("--greedy", help="Greedy play", default=False, action="store_true")  # noqa: E501
    train.add_argument("--resume", help="Load latest weights in out_directory and resume", default=False, action="store_true")  # noqa: E501
    train.add_argument("--verbose", "-v", help="Turn on verbose mode", type=int, default=0)  # noqa: E501
    train.set_defaults(func=start_training)

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    args = {
        'initial_weights': './params/policy/kihuu.hdf5',
        'out_directory': './train/rl_policy/',
        'learning_rate': 0.001,
        'policy_temp': 0.67,
        'save_every': 500,
        'record_every': 1,
        'game_batch': 128,
        'move_limit': 500,
        'iterations': 10000,
        'greedy': False,
        'resume': False,
        'verbose': 1,  # Turn on verbose mode
    }

    from types import SimpleNamespace
    start_args = SimpleNamespace(**args)
    start_training(start_args)

    # execute function (train or resume)
    #args.func(args)


if __name__ == '__main__':
    main()
