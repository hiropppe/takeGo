import os

from bamboo.rollout.sgf2hdf5 import GameConverter


def run_game_converter(cmd_line_args=None):
    """Run conversions. command-line args may be passed in as a list
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Prepare SGF Go game files for training the neural network model.',
        epilog="Available features are: board, ones, turns_since, liberties,\
        capture_size, self_atari_size, liberties_after, sensibleness, and zeros.\
        Ladder features are not currently implemented")
    parser.add_argument("--outfile", "-o", help="Destination to write data (hdf5 file)", required=True)  # noqa: E501
    parser.add_argument("--directory", "-d", help="Directory containing SGF files to process. if not present, expects files from stdin", default=None)  # noqa: E501
    parser.add_argument("--size", "-s", help="Size of the game board. SGFs not matching this are discarded with a warning", type=int, default=19)  # noqa: E501
    parser.add_argument("--nakade_file", "-nakade", help="nakade pattern file", default=None)
    parser.add_argument("--x33_file", "-x33", help="3x3 pattern file", default=None)
    parser.add_argument("--d12_file", "-d12", help="12 point diamond pattern file", default=None)
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    converter = GameConverter(args.size,
                              args.nakade_file,
                              args.x33_file,
                              args.d12_file)

    def _is_sgf(fname):
        return fname.strip()[-4:] == ".sgf"

    def _list_sgfs(path):
        """helper function to get all SGF files in a directory (does not recurse)
        """
        files = os.listdir(path)
        return [os.path.join(path, f) for f in files if _is_sgf(f)]

    # get an iterator of SGF files according to command line args
    if args.directory:
        files = _list_sgfs(args.directory)
    else:
        files = (f.strip() for f in sys.stdin if _is_sgf(f))

    converter.sgfs_to_hdf5(files, args.outfile, verbose=args.verbose)


if __name__ == '__main__':
    run_game_converter()
