# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

import numpy as np

cimport numpy as np

from libcpp.string cimport string as cppstring
from libcpp.string cimport to_string
from libcpp.vector cimport vector as cppvector
from libc.math cimport roundf, powf, ceilf
from libc.string cimport memset
from libc.stdio cimport printf, snprintf, stdout, fflush
from bamboo.board cimport board_start, board_end, board_size, pure_board_size

ctypedef np.float32_t FLOAT_t
ctypedef np.float64_t DOUBLE_t


cdef cppstring *rjust(char *buf, unsigned int width, char *fill_char) nogil:
    cdef unsigned int i
    cdef cppstring val
    val = cppstring(buf)
    s = &val
    if width > s.size():
        for i in range(width - s.size()):
            s.insert(0, fill_char)
    return s


cpdef void test_pad():
    cdef cppstring str = to_string(123)
    cdef int num = 10
    cdef char fill_char = b' '
    cdef float p = 15.456
    #pad(str, num, fill_char)
    str.insert(0, num - str.size(), fill_char);
    printf('%s\n', str.c_str())
    fflush(stdout)

    fpad(p)


cdef void fpad(float p) nogil:
    cdef cppstring str
    cdef int num = 10
    cdef char fill_char = b' '
    cdef float pp
    pp = ceilf(p*powf(10, 2))/powf(10, 2)
    str = to_string(pp).substr(0, 5)
    str.insert(0, num - str.size(), fill_char);
    printf('%s', str.c_str())
    fflush(stdout)


cdef void pad(cppstring &str, unsigned int num, char fill_char) nogil:
    #cdef char fill_char = b' '
    str.insert(0, num - str.size(), fill_char);


cdef void pp():
    cdef char buf[10]
    cdef cppstring *s
    cdef cppstring a
    snprintf(buf, sizeof(buf), "%d:|", pure_board_size)
    s = rjust(buf, 4, " ")
    a = f'{pure_board_size}:|'.encode('utf8')
    printf("%s", a.c_str())
    #print(<bytes>buf)
    #s = cppstring(b'a')
    #s = <bytes>buf
    #s = b"1:|".rjust(4, b" ")
    #s = b"aiueo"
    printf("%s", s.c_str())
    fflush(stdout)


cpdef void vector_to_memoryview():
    cdef int i
    cdef cppvector[int] move_vector
    cdef int[::1] move_array
    cdef int[:] move_ndarray

    for i in range(361):
        if i % 2 == 0:
            move_vector.push_back(i)

    move_array = <int [:move_vector.size()]>move_vector.data()
    print(move_array, move_array.shape)
    # arr is of type Memoryview. To cast into Numpy:
    move_ndarray = np.asarray(move_array)
    print(move_ndarray, move_ndarray.shape)


cpdef void test_mask_by_memoryview():
    cdef np.ndarray[FLOAT_t, ndim=2] probs
    cdef np.ndarray[DOUBLE_t, ndim=2] masked_probs
    cdef int[:] mask = np.zeros(3, dtype=np.int32)
    #cdef np.ndarray[np.npy_bool, ndim=1] mask = np.zeros(3, dtype=np.bool_)

    mask[0] = 1
    mask[1] = 0
    mask[2] = 1

    probs = np.array([
        [0.3, 0.2, 0.5], [0.1, 0.5, 0.4]
    ], dtype=np.float32)

    masked_probs = probs * mask

    print(mask)
    print(probs)
    print(masked_probs)


cpdef void test_mask():
    cdef np.ndarray[FLOAT_t, ndim=2] probs, masked_probs
    cdef np.ndarray[np.npy_bool, ndim=1] mask = np.zeros(3, dtype=np.bool_)

    mask[0] = True
    mask[1] = False
    mask[2] = True

    probs = np.array([
        [0.3, 0.2, 0.5], [0.1, 0.5, 0.4]
    ], dtype=np.float32)

    masked_probs = probs * mask

    print(mask)
    print(probs)
    print(masked_probs)
