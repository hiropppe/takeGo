import os
import sys

from libc.stdio cimport printf

from bamboo.util cimport SGFIterator
from bamboo.go.board cimport PASS, CORRECT_X, CORRECT_Y, OB_SIZE
from bamboo.go.board cimport game_state_t, string_t, liberty_end, board_size 
from bamboo.go.board cimport allocate_game, free_game, onboard_index, set_board_size, initialize_board
from bamboo.go.printer cimport print_board
from bamboo.rollout.pattern cimport calculate_x33_bit, print_x33

from collections import Counter

cdef unsigned int pat3[361]

freq = Counter()
move_freq = Counter()

def harvest_pattern(file_name):    
    cdef game_state_t *game
    cdef SGFIterator sgf_iter = SGFIterator(19)
    cdef int *updated_string_num
    cdef int *updated_string_id
    cdef int i
    cdef int string_id
    cdef string_t *string
    cdef int pos
    cdef int pat

    with open(file_name, 'r') as file_object:
        sgf_iter.read(file_object.read())

    while sgf_iter.has_next():
        game = sgf_iter.move_next()
        if sgf_iter.next_move != PASS:
            updated_string_num = &game.updated_string_num[<int>game.current_color]
            updated_string_id = game.updated_string_id[<int>game.current_color]
            for i in range(updated_string_num[0]):
                string_id = updated_string_id[i]
                string = &game.string[string_id]
                pos = string.lib[0]
                while pos != liberty_end:
                    pat = calculate_x33_bit(game, pos, <int>game.current_color)
                    if pat3[onboard_index[pos]] != pat:
                        pat3[onboard_index[pos]] = pat
                        freq[pat] += 1
                        if pos == sgf_iter.next_move:
                            move_freq[pat] += 1
                    pos = string.lib[pos]
        updated_string_num[0] = 0

    for freq_pat in freq.most_common(10):
        print 'Freq', freq_pat[1]
        print_x33(freq_pat[0])

    for freq_pat in move_freq.most_common(10):
        print 'Move', freq_pat[1]
        print_x33(freq_pat[0])


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
