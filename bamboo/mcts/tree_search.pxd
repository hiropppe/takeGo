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
    unsigned int node_i       # node index
    int time_step
    int pos
    int color
    int player_color
    double P     # prior probability
    double Nv    # evaluation count
    double Nr    # rollout count(visit count)
    double Wv    # evaluation value
    double Wr    # rollout value
    double Q     # action-value for edge
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
    cdef bint policy_queue_running
    cdef int playout_limit
    cdef int n_playout
    cdef int n_threads
    cdef double beta
    cdef openmp.omp_lock_t tree_lock
    cdef int max_queue_size_P
    cdef bint debug

    cdef int genmove(self, game_state_t *game) nogil

    cdef void start_search_thread(self, game_state_t *game)

    cdef void stop_search_thread(self)

    cdef void run_search(self, game_state_t *game) nogil

    cdef bint seek_root(self, game_state_t *game) nogil

    cdef void search(self, tree_node_t *node, game_state_t *game) nogil
    
    cdef tree_node_t *select(self, tree_node_t *node, game_state_t *game) nogil

    cdef bint expand(self, tree_node_t *node, game_state_t *game) nogil

    cdef void evaluate_and_backup(self, tree_node_t *node, game_state_t *game) nogil

    cdef void rollout(self, game_state_t *game) nogil

    cdef void backup(self, tree_node_t *node, int winner) nogil

    cdef void eval_leafs_by_policy_network(self, tree_node_t *node)
