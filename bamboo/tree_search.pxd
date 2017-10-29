from libcpp.queue cimport queue as cppqueue
from posix.time cimport timeval

from bamboo.board cimport game_state_t
from bamboo.policy_feature cimport policy_feature_t

cimport openmp


cdef extern from "ray.h":
    int THREAD_MAX
    int MAX_NODES
    int UCT_CHILD_MAX
    int NOT_EXPANDED
    int PASS_INDEX

    int RESIGN_THRESHOLD
    int EXPANSION_THRESHOLD
    int EXPLORATION_CONSTANT
    int VIRTUAL_LOSS
    int MIXING_PARAMETER

    int PLAYOUT_LIMIT
    double THINKING_TIME_LIMIT


ctypedef struct tree_node_t:
    unsigned int node_i       # node index
    int time_step
    int pos
    int color
    int player_color
    int depth
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

    openmp.omp_lock_t lock


ctypedef struct lgr2_seed_t:
    int prev_pos
    int prev2_pos
    int pos


cdef class MCTS:
    cdef game_state_t *game
    cdef char player_color
    cdef tree_node_t *nodes
    cdef unsigned int current_root
    cdef object pn
    cdef object vn_session
    cdef object vn_op
    cdef object vn_inputs
    cdef bint intuition
    cdef bint use_pn
    cdef bint use_vn
    cdef bint use_rollout
    cdef bint use_tree
    cdef policy_feature_t *policy_feature
    cdef policy_feature_t *value_feature
    cdef cppqueue[tree_node_t *] policy_network_queue
    cdef cppqueue[tree_node_t *] value_network_queue
    # cdef cppqueue[game_state_t *] game_queue
    cdef bint pondered
    cdef bint pondering
    cdef bint pondering_stopped
    cdef bint pondering_suspending
    cdef bint pondering_suspended
    cdef bint policy_queue_running
    cdef bint value_queue_running
    cdef double winning_ratio
    cdef double main_time
    cdef double byoyomi_time
    cdef double time_left
    cdef double const_time
    cdef bint can_extend
    cdef int root_depth
    cdef int leaf_depth
    cdef int const_playout
    cdef int n_playout
    cdef int n_threads
    cdef double beta
    cdef int max_queue_size_P
    cdef int max_queue_size_V
    cdef timeval search_start_time
    cdef openmp.omp_lock_t tree_lock
    cdef openmp.omp_lock_t expand_lock
    cdef openmp.omp_lock_t policy_queue_lock
    cdef openmp.omp_lock_t value_queue_lock
    # cdef openmp.omp_lock_t game_queue_lock
    cdef int n_threads_playout[100]
    cdef int lgr2[3][529][529]
    cdef bint self_play
    cdef bint debug

    cdef int genmove(self, game_state_t *game) nogil

    cdef void start_pondering(self) nogil

    cdef void stop_pondering(self) nogil

    cdef void suspend_pondering(self) nogil

    cdef void resume_pondering(self) nogil

    cdef void ponder(self, game_state_t *game, bint extend) nogil

    cdef void run_search(self, int thread_id, game_state_t *game, double thinking_time, int playout_limit) nogil

    cdef bint seek_root(self, game_state_t *game) nogil

    cdef void search(self, tree_node_t *node, game_state_t *game) nogil
    
    cdef tree_node_t *select(self, tree_node_t *node, game_state_t *game) nogil

    cdef bint expand(self, tree_node_t *node, game_state_t *game) nogil

    cdef int rollout(self, game_state_t *game) nogil

    cdef void backup(self, tree_node_t *node, int winner) nogil

    cdef void start_policy_network_queue(self) nogil

    cdef void stop_policy_network_queue(self) nogil

    cdef void clear_policy_network_queue(self) nogil

    cdef void eval_all_leaf_by_policy_network(self) nogil

    cdef void eval_leaf_by_policy_network(self, tree_node_t *node)

    cdef void start_value_network_queue(self) nogil

    cdef void stop_value_network_queue(self) nogil

    cdef void clear_value_network_queue(self) nogil

    cdef void eval_leaf_by_value_network(self, tree_node_t *node)


cdef class PyMCTS:
    cdef:
        MCTS mcts
        game_state_t *game
        double const_time
        int const_playout
        bint read_ahead
