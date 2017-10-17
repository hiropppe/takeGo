from libcpp.unordered_map cimport unordered_map


cdef extern from "common.h":
    int MAX(int x, int y) nogil
    int MIN(int x, int y) nogil

cdef extern from "ray.h":
    int PURE_BOARD_SIZE
    int PURE_BOARD_MAX
    int OB_SIZE
    int BOARD_SIZE
    int BOARD_MAX
    int MAX_STRING
    int MAX_NEIGHBOR 
    int BOARD_START
    int BOARD_END
    int STRING_LIB_MAX
    int STRING_POS_MAX
    int STRING_EMPTY_MAX
    int STRING_END
    int NEIGHBOR_END
    int LIBERTY_END
    int STRING_EMPTY_END
    int MAX_RECORDS
    int MAX_MOVES
    int PASS
    int RESIGN
    int KOMI

    int POS(int x, int y, int board_size) nogil
    int X(int pos, int board_size) nogil
    int Y(int pos, int board_size) nogil
    int CORRECT_X(int pos, int board_size, int ob_size) nogil
    int CORRECT_Y(int pos, int board_size, int ob_size) nogil
    int NORTH_WEST(int pos, int board_size) nogil
    int NORTH(int pos, int board_size) nogil
    int NORTH_EAST(int pos, int board_size) nogil
    int WEST(int pos) nogil
    int EAST(int pos) nogil
    int SOUTH_WEST(int pos, int board_size) nogil
    int SOUTH(int pos, int board_size) nogil
    int SOUTH_EAST(int pos, int board_size) nogil

    char FLIP_COLOR(char color) nogil

    int DX(int pos1, int pos2, int *board_x) nogil
    int DY(int pos1, int pos2, int *board_y) nogil
    int DIS(int pos1, int pos2, int *board_x, int *board_y, int move_dis[19][19]) nogil

    ctypedef enum stone:
        S_EMPTY
        S_BLACK
        S_WHITE
        S_OB
        S_MAX

    ctypedef enum eye_condition_t:
        E_NOT_EYE
        E_COMPLETE_HALF_EYE
        E_HALF_3_EYE
        E_HALF_2_EYE
        E_HALF_1_EYE
        E_COMPLETE_ONE_EYE
        E_MAX

    ctypedef struct move_t:
        int color
        int pos
        unsigned long long hash

    ctypedef struct string_t:
        char color
        int libs
        short lib[483]
        int neighbors
        short neighbor[288]
        int empties
        short empty[483]
        int origin
        int size
        bint flag

    ctypedef struct rollout_feature_t:
        int color
        int tensor[9][361]
        int prev_neighbor8[8]
        int prev_neighbor8_num
        int prev_d12[12]
        int prev_d12_num
        int prev_nakade
        int updated[529]
        int updated_num

    ctypedef struct game_state_t:
        char current_color
        move_t record[1083]     # MAX_RECORDS
        int moves
        int prisoner[4]         # S_MAX
        int ko_pos
        int ko_move

        unsigned long long current_hash
        unsigned long long positional_hash

        char board[529]         # BOARD_MAX
        int birth_move[529]     # BOARD_MAX

        int pass_count

        unsigned int pat[529]   # BOARD_MAX

        string_t string[288]    # MAX_STRING
        int string_id[483]      # STRING_POS_MAX
        int string_next[483]    # STRING_POS_MAX

        int candidates[529]     # BOARD_MAX

        int seki[529]          # BOARD_MAX

        int capture_num[3]      # S_OB
        int capture_pos[3][361] # S_OB, PURE_BOARD_MAX

        int updated_string_num[3]       # S_OB
        int updated_string_id[3][1083]  # S_OB, MAX_RECORDS

        rollout_feature_t rollout_feature_planes[3] # S_OB
        double rollout_probs[3][361]    # S_OB, PURE_BOARD_MAX
        double rollout_row_probs[3][19] # S_OB, PURE_BOARD_SIZE
        double rollout_logits[3][361]   # S_OB, PURE_BOARD_MAX
        double rollout_logits_sum[3]    # S_OB


cdef int pure_board_size
cdef int pure_board_max

cdef int board_size
cdef int board_max

cdef int max_string
cdef int max_neighbor

cdef int board_start
cdef int board_end

cdef int string_lib_max
cdef int string_pos_max

cdef int string_end
cdef int liberty_end

cdef int max_records
cdef int max_moves

cdef double komi

cdef int *board_pos_id

cdef int *board_x
cdef int *board_y

cdef int *board_dis_x
cdef int *board_dis_y

cdef int move_dis[19][19]   # PURE_BOARD_SIZE

cdef int *onboard_pos
cdef int *onboard_index

cdef int *corner
cdef int[:, ::1] corner_neighbor

cdef unsigned char territory[65536]     # PAT3_MAX
cdef unsigned char nb4_empty[65536]     # PAT3_MAX
cdef unsigned char eye_condition[65536] # PAT3_MAX

cdef bint check_superko
cdef bint japanese_rule

cdef int diagonals[529][4]
cdef int neighbor4[529][4]
cdef int neighbor8[529][8]
cdef int neighbor8_in_order[529][8]

cdef void fill_n_char (char *arr, int size, char v) nogil
cdef void fill_n_unsigned_char (unsigned char *arr, int size, unsigned char v) nogil
cdef void fill_n_short (short *arr, int size, short v) nogil
cdef void fill_n_int (int *arr, int size, int v) nogil
cdef void fill_n_bint (bint *arr, int size, bint v) nogil

cdef void initialize_const() 
cdef void clear_const()
cdef void set_board_size(int size)
cdef void set_komi(double new_komi)
cdef void set_superko(bint check)

cdef game_state_t *allocate_game() nogil
cdef void free_game(game_state_t *game) nogil
cdef void copy_game(game_state_t *dst, game_state_t *src) nogil
cdef void initialize_board(game_state_t *game)

cdef bint do_move(game_state_t *game, int pos) nogil

cdef bint put_stone(game_state_t *game, int pos, char color) nogil
cdef void connect_string(game_state_t *game, int pos, char color, int connection, int string_id[4]) nogil
cdef void merge_string(game_state_t *game, string_t *dst, string_t *src[3], int n) nogil
cdef void add_stone(game_state_t *game, int pos, char color, int string_id) nogil
cdef void add_stone_to_string(game_state_t *game, string_t *string, int pos, int head) nogil
cdef void make_string(game_state_t *game, int pos, char color) nogil
cdef int remove_string(game_state_t *game, string_t *string) nogil


cdef int add_liberty(string_t *string, int pos, int head) nogil
cdef void remove_liberty(string_t *string, int pos) nogil
cdef void add_neighbor(string_t *string, int id, int head) nogil
cdef void remove_neighbor_string(string_t *string, int id) nogil

cdef int add_empty(string_t *string, int pos, int head) nogil
cdef void remove_empty(string_t *string, int pos) nogil

cdef void get_diagonals(int diagonals[4], int pos) nogil
cdef void get_neighbor4(int neighbor4[4], int pos) nogil
cdef void get_neighbor8(int neighbor8[8], int pos) nogil
cdef void get_neighbor8_in_order(int neighbor8[8], int pos) nogil

cdef void get_md12(int md12[12], int pos) nogil

cdef void init_board_position()
cdef void init_line_number()
cdef void init_move_distance()
cdef void init_corner()
cdef void initialize_neighbor()
cdef void initialize_territory()
cdef void initialize_eye()
cdef int get_neighbor4_empty(game_state_t *game, int pos) nogil
cdef bint is_legal(game_state_t *game, int pos, char color) nogil
cdef bint is_legal_not_eye(game_state_t *game, int pos, char color) nogil
cdef bint is_suicide(game_state_t *game, int pos, char color) nogil
cdef bint is_true_eye(game_state_t *game, int pos, char color, char other_color, int empty_diagonal_stack[200], int empty_diagonal_top) nogil
cdef bint is_superko(game_state_t *game, int pos, char color) nogil
cdef int calculate_score(game_state_t *game) nogil
cdef void check_bent_four_in_the_corner(game_state_t *game) nogil
