import os

from bamboo.rollout.sgf2hdf5 import GameConverter as RolloutGameConverter
from bamboo.rollout.sgf2hdf5_tree import GameConverter as TreeGameConverter


def run_game_converter(cmd_line_args=None):
    """Run conversions. command-line args may be passed in as a list
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Prepare SGF Go game files for training the rollout model.')
    parser.add_argument("--outfile", "-o", required=True,
                        help="Destination to write data (hdf5 file)")
    parser.add_argument("--directory", "-d", default=None,
                        help="Directory containing SGF files to process. if not present, expects files from stdin")
    parser.add_argument("--policy", "-p", type=str, default='rollout', choices=['rollout', 'tree'],
                        help="Choice policy to generate feature (Default: rollout)")
    parser.add_argument("--size", "-s", type=int, default=19,
                        help="Size of the game board. SGFs not matching this are discarded with a warning")
    parser.add_argument("--mt-rands-file", "-mt", required=True, type=str, default=None,
                        help="Mersenne twister random number file. Default: None")
    parser.add_argument("--x33-file", "-x33", default=None,
                        help="3x3 pattern file. Default: None")
    parser.add_argument("--d12-file", "-d12", default=None,
                        help="12 point diamond pattern file. Default:None")
    parser.add_argument("--nonres-d12-file", "-nd12", default=None,
                        help="Non-response 12 point diamond pattern file. Default:None")
    parser.add_argument("--verbose", "-v", default=False, action="store_true",
                        help="Turn on verbose mode")
    parser.add_argument("--quiet", "-q", default=False, action="store_true",
                        help="Turn on quiet mode")

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    if args.policy == 'rollout':
        converter = RolloutGameConverter(args.size,
                                         args.mt_rands_file,
                                         args.x33_file,
                                         args.d12_file)
    else:
        converter = TreeGameConverter(args.size,
                                      args.mt_rands_file,
                                      args.x33_file,
                                      args.d12_file,
                                      args.nonres_d12_file)

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

    converter.sgfs_to_hdf5(files, args.outfile, verbose=args.verbose, quiet=args.quiet)


if __name__ == '__main__':
    run_game_converter()
