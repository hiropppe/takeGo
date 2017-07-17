class SizeMismatchError(Exception):
    pass


class IllegalMove(Exception):
    def __init__(self, move=None):
        if move is not None:
            self.pos = move[0]
            self.color = move[1]


class TooManyMove(Exception):
    def __init__(self, n_moves):
        self.n_moves = n_moves


class TooFewMove(Exception):
    def __init__(self, n_moves):
        self.n_moves = n_moves
