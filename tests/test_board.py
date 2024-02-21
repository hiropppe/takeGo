from bamboo import test_board


class TestBoard():
    
    def test_set_board_size_9(self):
        test_board.test_set_board_size_9()

    def test_set_board_size_19(self):
        test_board.test_set_board_size_19()

    def test_add_liberty_to_isolated_one(self):
        test_board.test_add_liberty_to_isolated_one()

    def test_remove_liberty_of_isolated_one(self):
        test_board.test_remove_liberty_of_isolated_one()

    def test_make_string_of_isolated_one(self):
        test_board.test_make_string_of_isolated_one()

    def test_add_neighbor_to_isolated_one(self):
        test_board.test_add_neighbor_to_isolated_one()

    def test_add_stone_to_string_less_position(self):
        test_board.test_add_stone_to_string_less_position()

    def test_add_stone_to_string_larger_position_from_zero_head(self):
        test_board.test_add_stone_to_string_larger_position_from_zero_head()

    def test_add_stone_to_string_larger_position_from_nonzero_head(self):
        test_board.test_add_stone_to_string_larger_position_from_nonzero_head()

    def test_add_stone_to_isolated_stone(self):
        test_board.test_add_stone_to_isolated_stone()

    def test_merge_string_two(self):
        test_board.test_merge_string_two()

    def test_merge_string_three(self):
        test_board.test_merge_string_three()

    def test_is_legal_stone_exists(self):
        test_board.test_is_legal_stone_exists()

    def test_is_legal_nb4_empty_is_zero(self):
        test_board.test_is_legal_nb4_empty_is_zero()

    def test_is_legal_nb4_empty_is_zero_edge(self):
        test_board.test_is_legal_nb4_empty_is_zero_edge()

    def test_string_add_and_remove_empty(self):
        test_board.test_string_add_and_remove_empty()

    def test_string_merge_empty(self):
        test_board.test_string_merge_empty()

    def test_ko(self):
        test_board.test_ko()
