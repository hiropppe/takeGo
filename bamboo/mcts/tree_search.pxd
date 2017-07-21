from libcpp.queue cimport queue as cppqueue

from bamboo.go.board cimport game_state_t
from bamboo.go.policy_feature cimport policy_feature_t

cimport openmp


cdef extern from "ray.h":
    int THREAD_MAX
    int MAX_NODES
    int UCT_CHILD_MAX
    int NOT_EXPANDED
    int PASS_INDEX

    int EXPANSION_THRESHOLD
    int EXPLORATION_CONSTANT
    int VIRTUAL_LOSS
    int MIXING_PARAMETER


ctypedef struct tree_node_t:
    int node_i       # node index
    int time_step
    int pos
    int color
    float P     # prior probability
    int Nv    # evaluation count
    int Nr    # rollout count(visit count)
    float Wv    # evaluation value
    int Wr    # rollout value
    float Q     # action-value for edge
    float u     # PUCT algorithm
    bint is_root
    bint is_edge

    tree_node_t *parent
    tree_node_t *children[361]

    int children_pos[361]   # PURE_BOARD_MAX
    int num_child

    game_state_t *game
    bint has_game

    openmp.omp_lock_t node_lock


cdef class MCTS:
    cdef tree_node_t *nodes
    cdef unsigned int current_root
    cdef object policy
    cdef policy_feature_t *policy_feature
    cdef cppqueue[tree_node_t *] policy_network_queue
    cdef cppqueue[tree_node_t *] value_network_queue
    cdef bint pondering
    cdef int n_threads
    cdef openmp.omp_lock_t tree_lock
    
    cdef void start_search_thread(self, game_state_t *game, int player_color)

    cdef void stop_search_thread(self)

    cdef void run_search(self, game_state_t *game, int player_color) nogil

    cdef bint seek_root(self, game_state_t *game) nogil

    cdef void search(self, tree_node_t *node, game_state_t *game, int player_color) nogil
    
    cdef tree_node_t *select(self, tree_node_t *node, game_state_t *game) nogil

    cdef int expand(self, tree_node_t *node, game_state_t *game) nogil

    cdef void evaluate_and_backup(self, tree_node_t *node, game_state_t *game, int player_color) nogil

    cdef void rollout(self, game_state_t *game) nogil

    cdef void backup(self, tree_node_t *node, int Nr, int Wr) nogil

    cdef void eval_leafs_by_policy_network(self, tree_node_t *node)
