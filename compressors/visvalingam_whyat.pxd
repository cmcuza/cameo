import numpy as np
cimport numpy as np

cpdef np.ndarray[np.uint8_t, ndim=1] simplify(long [:] x, double[:] y, int nlags, double acf_threshold)

cdef double[:] triangle_areas_from_array(long[:] x, double[:] y)

cpdef np.ndarray[np.uint8_t, ndim=1] simplify_by_acf_agg(long [:] x, double[:] y,
                                                                   int nlags, int kappa, double acf_threshold)