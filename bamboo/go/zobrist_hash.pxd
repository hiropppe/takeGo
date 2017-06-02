from bamboo.go.board cimport game_state_t


cdef extern from "ray.h":
    unsigned int UCT_HASH_SIZE

    ctypedef enum hash:
        HASH_PASS
        HASH_BLACK
        HASH_WHITE
        HASH_KO
        HASH_MAX

    ctypedef struct node_hash_t:
        unsigned long long hash
        int color
        int moves
        bint flag

cdef unsigned long long hash_bit[529][4]    # BOARD_MAX, HASH_MAX
cdef unsigned long long shape_bit[529]      # BOARD_MAX

cdef unsigned int uct_hash_size
cdef unsigned int uct_hash_limit


cdef unsigned long long mt() nogil

cdef void set_hash_size(unsigned int hash_size)

cdef void initialize_hash()

cdef void initialize_uct_hash()

cdef void clear_uct_hash()

cdef void delete_old_hash(game_state_t *game) nogil

cdef unsigned int search_empty_index(unsigned long long hash, int color, int moves) nogil

cdef unsigned int find_same_hash_index(unsigned long long hash, int color, int moves) nogil

cdef bint check_remaining_hash_size() nogil
