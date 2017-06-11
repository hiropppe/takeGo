import os
import sys

from libc.stdio cimport printf

from bamboo.util cimport SGFMoveIterator
from bamboo.go.board cimport PASS, CORRECT_X, CORRECT_Y, OB_SIZE
from bamboo.go.board cimport game_state_t, string_t, liberty_end, board_size 
from bamboo.go.board cimport allocate_game, free_game, onboard_index, set_board_size, initialize_board
from bamboo.go.printer cimport print_board
from bamboo.rollout.pattern cimport calculate_x33_bit, print_x33

from collections import Counter, defaultdict

cdef unsigned int pat3[361]

pat_freq = Counter()
move_freq = Counter()
move_ratio = defaultdict(float)

def harvest_pattern(file_name):    
    cdef game_state_t *game
    cdef SGFMoveIterator sgf_iter
    cdef int *updated_string_num
    cdef int *updated_string_id
    cdef int i
    cdef int string_id
    cdef string_t *string
    cdef int center
    cdef int pat

    with open(file_name, 'r') as file_object:
        sgf_iter = SGFMoveIterator(file_object.read())

    game = sgf_iter.game
    for move in sgf_iter:
        if move[0] != PASS:
            updated_string_num = &game.updated_string_num[<int>game.current_color]
            updated_string_id = game.updated_string_id[<int>game.current_color]
            for i in range(updated_string_num[0]):
                string_id = updated_string_id[i]
                string = &game.string[string_id]
                center = string.lib[0]
                while center != liberty_end:
                    pat = calculate_x33_bit(game, center, <int>game.current_color)
                    if pat3[onboard_index[center]] != pat:
                        pat3[onboard_index[center]] = pat
                        pat_freq[pat] += 1
                        if sgf_iter.next_move and sgf_iter.next_move[0] == center:
                            move_freq[pat] += 1
                            move_ratio[pat] = move_freq[pat]/float(pat_freq[pat])
                    center = string.lib[center]
        updated_string_num[0] = 0

    from operator import itemgetter
    for pat, freq in sorted(move_ratio.items(), key=itemgetter(1), reverse=True):
        print('Put {:s} {:.3f}'.format(bin(pat), freq))
        print_x33(pat)
	

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--outfile", "-o", help="Destination to write data", required=True)  # noqa: E501
    parser.add_argument("--directory", "-d", help="Directory containing SGF files to process. if not present, expects files from stdin", default=None)  # noqa: E501
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501
    args = parser.parse_args()

    def _is_sgf(fname):
        return fname.strip()[-4:] == ".sgf"

    def _list_sgfs(path):
        files = os.listdir(path)
        return [os.path.join(path, f) for f in files if _is_sgf(f)]

    if args.directory:
        sgf_files = _list_sgfs(args.directory)
    else:
        sgf_files = (f.strip() for f in sys.stdin if _is_sgf(f))

    for sgf in sgf_files:
        harvest_pattern(sgf, args.outfile)
