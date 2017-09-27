# -*- coding: utf-8 -*-

cdef extern from "ray.h":
    unsigned int REV(unsigned int p)
    unsigned int REV2(unsigned int p)
    unsigned int REV3(unsigned int p)
    unsigned int REV4(unsigned int p)
    unsigned int REV6(unsigned int p)
    unsigned int REV8(unsigned int p)
    unsigned int REV10(unsigned int p)
    unsigned int REV12(unsigned int p)
    unsigned int REV14(unsigned int p)
    unsigned int REV16(unsigned int p)
    unsigned int REV18(unsigned int p)

    int PAT3_MAX


cdef int N
cdef int S
cdef int W
cdef int E
cdef int NN
cdef int NW
cdef int NE
cdef int SS
cdef int SW
cdef int SE
cdef int WW
cdef int EE


# init
cdef void init_const()
cdef void clear_pattern(unsigned int *pat)

# update
cdef void update_md2_stone(unsigned int *pat, int pos, char color) nogil
cdef void update_md2_empty(unsigned int *pat, int pos) nogil

# search pattern
cdef unsigned int pat3(unsigned int *pat, int pos) nogil

# symmetric 
cdef void pat3_transpose8(unsigned int pat3, unsigned int *transp)
cdef void pat3_transpose16(unsigned int pat3, unsigned int *transp)

# flip color
cdef unsigned int pat3_reverse(unsigned int pat3)

# vertical mirror
cdef unsigned int pat3_vertical_mirror(unsigned int pat3)

# horizontal mirror
cdef unsigned int pat3_horizontal_mirror(unsigned int pat3)

# rotate
cdef unsigned int pat3_rotate90(unsigned int pat3)

# print pattern
cdef void print_input_pat3(unsigned int pat3)
