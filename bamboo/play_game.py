from bamboo.models.keras_dcnn_policy import KerasPolicy, cnn_policy
from bamboo.self_play_game import run_n_games


def main(weights_a, weights_b, policy_temp, n_games, verbose=False):
    model_a = cnn_policy()
    model_a.load_weights(weights_a)
    policy_a = KerasPolicy(model_a)

    model_b = cnn_policy()
    model_b.load_weights(weights_b)
    policy_b = KerasPolicy(model_b)

    _, _, _, win_ratio = run_n_games(policy_a, policy_b, n_games, temperature=policy_temp, verbose=verbose)
    print(f'winning ratio: {win_ratio:.2f}')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Play games with specified weights.')  # noqa: E501
    parser.add_argument("weights_a", help="Player weights A.")
    parser.add_argument("weights_b", help="Player weights B.")
    parser.add_argument("--policy_temp", help="Distribution temperature of players using policies (Default: 0.67)", type=float, default=0.67)  # noqa: E501
    parser.add_argument("--n_games", help="Number of games to play (Default: 10)", type=int, default=10)  # noqa: E501
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501

    args = parser.parse_args()

    main(args.weights_a, args.weights_b, args.policy_temp, args.n_games, args.verbose)