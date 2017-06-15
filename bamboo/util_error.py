class SizeMismatchError(Exception):
    pass


class IllegalMove(Exception):
    pass


class TooManyMove(Exception):

    def __init__(self, n_moves):
        self.n_moves = n_moves


class TooFewMove(Exception):

    def __init__(self, n_moves):
        self.n_moves = n_moves
