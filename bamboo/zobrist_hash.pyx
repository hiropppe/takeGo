import sys

from libc.stdlib cimport malloc, free
from libc.stdio cimport printf

from bamboo.board cimport BOARD_MAX
from bamboo.board cimport game_state_t

from cpprand cimport uniform_int_distribution, mt19937_64, random_device


ctypedef uniform_int_distribution[long long] int_dist

cdef int_dist *randint = new int_dist(0, sys.maxsize)
cdef random_device *rand_gen = new random_device()
cdef int seed = rand_gen[0]()
cdef mt19937_64 *engine = new mt19937_64(seed)

cdef node_hash_t *node_hash
cdef int oldest_move = 1

cdef bint enough_size = True

uct_hash_size = UCT_HASH_SIZE
uct_hash_limit = UCT_HASH_SIZE * 9 // 10
used = 0


cdef unsigned long long mt() nogil:
    return randint[0](engine[0])


cdef unsigned int trans_hash(unsigned long long hash) nogil:
    return ((hash & <unsigned long long>0xffffffff) ^ ((hash >> 32) & <unsigned long long>0xffffffff)) & (uct_hash_size - 1)


cpdef void set_hash_size(unsigned int new_size):
    cdef int i

    global uct_hash_size, uct_hash_limit

    if not new_size & (new_size-1):
        uct_hash_size = new_size
        uct_hash_limit = new_size * 9 // 10
    else:
        printf("Hash size must be 2 ^ n\n")
        for i in range(1, 21):
            printf("2^%d:%d\n", i, 1 << i)


cpdef void initialize_hash():
    cdef int i

    global node_hash

    for i in range(BOARD_MAX):
        hash_bit[i][<int>HASH_PASS] = mt()
        hash_bit[i][<int>HASH_BLACK] = mt()
        hash_bit[i][<int>HASH_WHITE] = mt()
        hash_bit[i][<int>HASH_KO] = mt()
        shape_bit[i] = mt()

    node_hash = <node_hash_t *>malloc(uct_hash_size * sizeof(node_hash_t))


cpdef void initialize_uct_hash():
    cdef unsigned int i

    global oldest_move, used

    oldest_move = 1
    used = 0

    for i in range(uct_hash_size):
        node_hash[i].flag = False
        node_hash[i].hash = 0
        node_hash[i].color = 0
        node_hash[i].moves = 0


cpdef void clear_uct_hash():
    cdef unsigned int i

    global used
    global enough_size

    used = 0

    for i in range(uct_hash_size):
        node_hash[i].flag = False
        node_hash[i].hash = 0
        node_hash[i].color = 0
        node_hash[i].moves = 0


cdef void delete_old_hash(game_state_t *game) nogil:
    cdef unsigned int i

    global used
    global enough_size
    global oldest_move

    while oldest_move < game.moves:
        for i in range(uct_hash_size):
            if node_hash[i].flag and node_hash[i].moves == oldest_move:
                node_hash[i].flag = False
                node_hash[i].hash = 0
                node_hash[i].color = 0
                node_hash[i].moves = 0
                used -= 1

        oldest_move += 1

    enough_size = True


cdef unsigned int search_empty_index(unsigned long long hash, int color, int moves) nogil:
    cdef unsigned int key = trans_hash(hash)
    cdef unsigned int i = key

    global used, enough_size

    while True:
        if not node_hash[i].flag:
            node_hash[i].flag = True
            node_hash[i].hash = hash
            node_hash[i].moves = moves
            node_hash[i].color = color
            used+=1
            if used > uct_hash_limit:
                enough_size = False
            return i
        i+=1
        if i >= uct_hash_size:
            i = 0
        if i == key:
            break

    return uct_hash_size


cdef unsigned int find_same_hash_index(unsigned long long hash, int color, int moves) nogil:
    cdef unsigned int key = trans_hash(hash)
    cdef int i = key

    while True:
        if not node_hash[i].flag:
            return uct_hash_size
        elif (node_hash[i].hash == hash and
              node_hash[i].color == color and
              node_hash[i].moves == moves):
            return i
        i+=1
        if i >= uct_hash_size:
            i = 0
        if i == key:
            break

    return uct_hash_size


cdef bint check_remaining_hash_size() nogil:
    return enough_size


def test():
    print(uct_hash_size)
    print(uct_hash_limit)
    print(mt())
    initialize_hash()
    for i in range(BOARD_MAX):
        print(hash_bit[i][<int>HASH_PASS])
        print(hash_bit[i][<int>HASH_BLACK])
        print(hash_bit[i][<int>HASH_WHITE])
        print(hash_bit[i][<int>HASH_KO])
        print(shape_bit[i])
