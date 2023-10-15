from cython_modules.heap cimport Heap, Node
from libcpp.unordered_map cimport unordered_map
from cython_modules.inc_acf cimport AcfAgg
import numpy as np
cimport numpy as np


cdef void look_ahead_reheap(AcfAgg *acf_agg, Heap *acf_errors, unordered_map[int, int] &map_node_to_heap,
                            double [:]y, double[:] aggregates, double *raw_acf, const Node &removed_node, int &hops, int kappa)

cpdef np.ndarray[np.uint8_t, ndim=1] simplify_single_thread(double[:] y, int hops, int kappa,
                                                            int nlags, double acf_threshold)

cpdef np.ndarray[np.uint8_t, ndim=1] parallel_simplify_by_blocking(double[:] y, int hops,
                                                            int nlags, int kappa, double acf_threshold, int num_threads)
