import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from bamboo.train.policy.sgf2tfrecord import sgfs_to_tfrecord


def run_game_converter(cmd_line_args=None):
    """Run conversions. command-line args may be passed in as a list
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Prepare SGF Go game files for training the rollout model.')
    parser.add_argument("--outfile", "-o", required=True,
                        help="Destination to write data (hdf5 file)")
    parser.add_argument("--directory", "-d", default=None,
                        help="Directory containing SGF files to process. if not present, expects files from stdin")
    parser.add_argument("--split_by", type=str, default='transformation', choices=['transformation', 'sgf'],
                        help="Split output file by specified method. (Default: transformation)")
    parser.add_argument("--size", "-s", type=int, default=19,
                        help="Size of the game board. SGFs not matching this are discarded with a warning")
    parser.add_argument("--workers", "-w", type=int, default=1,
                        help="")
    parser.add_argument("--symmetry", default=False, action="store_true",
                        help="Write symmetric pattern or not (Default: False)")
    parser.add_argument("--verbose", "-v", default=False, action="store_true",
                        help="Turn on verbose mode (Default: False)")
    parser.add_argument("--quiet", "-q", default=False, action="store_true",
                        help="Turn on quiet mode (Default: False)")

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    sgfs_to_tfrecord(args.directory,
                     args.outfile,
                     args.workers,
                     split_by=args.split_by,
                     symmetry=args.symmetry,
                     verbose=args.verbose,
                     quiet=args.quiet)


if __name__ == '__main__':
    run_game_converter()
