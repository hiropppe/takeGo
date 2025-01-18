from bamboo.models.keras_dcnn_policy import KerasPolicy, cnn_policy
from bamboo.self_play_game import run_n_games


def main(ckpt_a, ckpt_b, policy_temp, n_games, verbose=False):
    policy_a = load_policy(ckpt_a)
    policy_b = load_policy(ckpt_b)

    _, _, _, win_ratio = run_n_games(policy_a, policy_b, n_games, temperature=policy_temp, verbose=verbose)

    print(f'winning ratio (A) : {win_ratio:.2f}')


def load_policy(ckpt):
    if ckpt.endswith(".keras"):
        policy = KerasPolicy.load(ckpt)
    else:
        model = cnn_policy()
        model.load_weights(ckpt)
        policy = KerasPolicy(model)
    return policy


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Play games with specified weights.')  # noqa: E501
    parser.add_argument("ckpt_a", help="SL Policy weights or model for Player A.")
    parser.add_argument("ckpt_b", help="SL Policy weights or model for Player B.")
    parser.add_argument("--policy_temp", help="Distribution temperature of players using policies (Default: 0.67)", type=float, default=0.67)  # noqa: E501
    parser.add_argument("--n_games", help="Number of games to play (Default: 10)", type=int, default=10)  # noqa: E501
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501

    args = parser.parse_args()

    main(args.ckpt_a, args.ckpt_b, args.policy_temp, args.n_games, args.verbose)
