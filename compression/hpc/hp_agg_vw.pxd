import numpy as np
cimport numpy as np


cpdef np.ndarray[np.uint8_t, ndim=1] simplify_by_agg_vw(long [:] x, double[:] y, Py_ssize_t nlags, Py_ssize_t kappa, double acf_threshold)