from bamboo import test_board


def test_set_board_size_9():
    test_board.test_set_board_size_9()

def test_set_board_size_19():
    test_board.test_set_board_size_19()

def test_add_liberty_to_isolated_one():
    test_board.test_add_liberty_to_isolated_one()

def test_remove_liberty_of_isolated_one():
    test_board.test_remove_liberty_of_isolated_one()

def test_make_string_of_isolated_one():
    test_board.test_make_string_of_isolated_one()

def test_add_neighbor_to_isolated_one():
    test_board.test_add_neighbor_to_isolated_one()

def test_add_stone_to_string_less_position():
    test_board.test_add_stone_to_string_less_position()

def test_add_stone_to_string_larger_position_from_zero_head():
    test_board.test_add_stone_to_string_larger_position_from_zero_head()

def test_add_stone_to_string_larger_position_from_nonzero_head():
    test_board.test_add_stone_to_string_larger_position_from_nonzero_head()

def test_add_stone_to_isolated_stone():
    test_board.test_add_stone_to_isolated_stone()

def test_merge_string_two():
    test_board.test_merge_string_two()

def test_merge_string_three():
    test_board.test_merge_string_three()

def test_is_legal_stone_exists():
    test_board.test_is_legal_stone_exists()

def test_is_legal_nb4_empty_is_zero():
    test_board.test_is_legal_nb4_empty_is_zero()

def test_is_legal_nb4_empty_is_zero_edge():
    test_board.test_is_legal_nb4_empty_is_zero_edge()

def test_string_add_and_remove_empty():
    test_board.test_string_add_and_remove_empty()

def test_string_merge_empty():
    test_board.test_string_merge_empty()

def test_ko():
    test_board.test_ko()
