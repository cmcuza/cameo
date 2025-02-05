from compressors.heap cimport Node
from compressors.inc_acf cimport AcfAgg
import numpy as np
cimport numpy as np

cdef double compute_importance(double[:] points, Node &node)

cdef bint is_line(double[:] points, int i)

cdef bint is_concave(double[:] points, int i)

cdef bint is_convex(double[:] points, int i)

cdef bint is_downtrend(double[:] points, int i)

cdef bint is_uptrend(double[:] points, int i)

cdef bint is_same_trend(double[:] points, int i)

cdef int extract_1st_tps_importance(AcfAgg *model, double[:] x, double * raw_acf,
                                     double acf_error, int * selected_tp,
                                     double * importance_tp, np.ndarray[np.uint8_t, ndim=1] extract_1st_tps_importance)

cpdef np.ndarray[np.uint8_t, ndim=1] simplify(double[:] y, int nlags, double acf_threshold)