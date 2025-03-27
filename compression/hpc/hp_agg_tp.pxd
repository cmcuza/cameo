from compression.lpc.heap cimport Node
from compression.hpc.hp_acf_agg_model cimport HPAcfAgg
import numpy as np
cimport numpy as np

cdef Py_ssize_t extract_1st_tps_importance(HPAcfAgg *model, double[:] x, 
                                    long double[:] aggregates, long double * raw_acf,
                                    double acf_error, Py_ssize_t * selected_tp,
                                    double * importance_tp, Py_ssize_t kappa, 
                                    np.ndarray[np.uint8_t, ndim=1] no_removed_indices)

cpdef np.ndarray[np.uint8_t, ndim=1] simplify_by_agg_tp(double[:] y, Py_ssize_t nlags, Py_ssize_t kappa, double acf_threshold)