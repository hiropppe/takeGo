cimport board 

from bamboo.rollout_preprocess cimport initialize_rollout 


cdef tuple parse(board.game_state_t *game, boardstr):
    '''Parses a board into a gamestate, and returns the location of any moves
    marked with anything other than 'B', 'X', '#', 'W', 'O', or '.'

    Rows are separated by '|', spaces are ignored.

    '''
    cdef int row, col, pos
    cdef dict moves = {}
    cdef dict pure_moves = {}

    boardstr = boardstr.replace(' ', '')
    board_size = max(boardstr.index('|'), boardstr.count('|'))

    # assert board_size in (7, 9, 13, 19), "Illegal board size"
    board.set_board_size(board_size)

    board.initialize_board(game)
    initialize_rollout(game)

    for row, rowstr in enumerate(boardstr.split('|')):
        for col, c in enumerate(rowstr):
            pos = board.POS(col + board.OB_SIZE, row + board.OB_SIZE, board.board_size)
            if c == '.':
                continue  # ignore empty spaces
            elif c in 'BX#':
                board.put_stone(game, pos, color=board.S_BLACK)
            elif c in 'WO':
                board.put_stone(game, pos, color=board.S_WHITE)
            else:
                # move reference
                assert c not in moves, "{} already used as a move marker".format(c)
                moves[c] = board.POS(col + board.OB_SIZE, row + board.OB_SIZE, board.board_size)
                pure_moves[c] = board.POS(col, row, board_size)

    return moves, pure_moves
