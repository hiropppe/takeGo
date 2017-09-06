import os

from bamboo.go.sgf2hdf5 import GameConverter


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
    parser.add_argument("--size", "-s", type=int, default=19,
                        help="Size of the game board. SGFs not matching this are discarded with a warning")
    parser.add_argument("--recurse", "-R", default=False, action="store_true",
                        help="Set to recurse through directories searching for SGF files")
    parser.add_argument("--verbose", "-v", default=False, action="store_true",
                        help="Turn on verbose mode")
    parser.add_argument("--quiet", "-q", default=False, action="store_true",
                        help="Turn on quiet mode")

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    converter = GameConverter(args.size)

    def _is_sgf(fname):
        return fname.strip()[-4:] == ".sgf"

    def _count_all_sgfs(root):
        """a helper function/generator to count all SGF files in subdirectories of root
        """
        count = 0
        for (dirpath, dirname, files) in os.walk(root):
            for filename in files:
                if _is_sgf(filename):
                    count += 1
        return count

    def _walk_all_sgfs(root):
        """a helper function/generator to get all SGF files in subdirectories of root
        """
        for (dirpath, dirname, files) in os.walk(root):
            for filename in files:
                if _is_sgf(filename):
                    # yield the full (relative) path to the file
                    yield os.path.join(dirpath, filename)

    def _list_sgfs(path):
        """helper function to get all SGF files in a directory (does not recurse)
        """
        files = os.listdir(path)
        return [os.path.join(path, f) for f in files if _is_sgf(f)]

    # get an iterator of SGF files according to command line args
    if args.directory:
        sgf_total = _count_all_sgfs(args.directory)
        if args.recurse:
            sgf_files = _walk_all_sgfs(args.directory)
        else:
            sgf_files = _list_sgfs(args.directory)
    else:
        sgf_total = 1
        sgf_files = [f.strip() for f in sys.stdin if _is_sgf(f)]

    converter.sgfs_to_hdf5(sgf_files, sgf_total, args.outfile, verbose=args.verbose, quiet=args.quiet)


if __name__ == '__main__':
    run_game_converter()
