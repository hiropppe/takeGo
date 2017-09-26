from libcpp.string cimport string as cppstring
from bamboo.board cimport game_state_t
from bamboo.tree_search cimport tree_node_t


cdef extern from "ray.h":
    char *gogui_x


cdef void print_board(game_state_t *game) nogil

cdef void print_rollout_count(tree_node_t *root) nogil

cdef void print_winning_ratio(tree_node_t *root) nogil

cdef void print_prior_probability(tree_node_t *root) nogil

cdef void print_action_value(tree_node_t *root) nogil

cdef void print_bonus(tree_node_t *root) nogil

cdef void print_selection_value(tree_node_t *root) nogil

cdef cppstring *rjust(char *buf, unsigned int width, char *fill_char) nogil
