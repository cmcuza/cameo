from compression.cython.heap cimport Heap, Node
from libcpp.unordered_map cimport unordered_map
from compression.cython.inc_acf cimport AcfAgg
import numpy as np
cimport numpy as np


cdef void parallel_look_ahead_reheap(AcfAgg *acf_agg, Heap *acf_errors, unordered_map[int, int] &map_node_to_heap,
                                     double [:]y, double *raw_acf, const Node &removed_node, int &hops, int &num_threads)


cdef void look_ahead_reheap(AcfAgg *acf_agg, Heap *acf_errors, unordered_map[int, int] &map_node_to_heap,
                            double [:]y, double *raw_acf, const Node &removed_node, int &hops)

cpdef np.ndarray[np.uint8_t, ndim=1] simplify_by_blocking(long [:] x, double[:] y, int hops, int nlags, double acf_threshold)

cpdef np.ndarray[np.uint8_t, ndim=1] parallel_simplify_by_blocking(long [:] x, double[:] y,
                                                                   int hops, int nlags,
                                                                   double acf_threshold, int num_threads)

cpdef np.ndarray[np.uint8_t, ndim=1] parallel_simplify_by_blocking_compression_centric(long [:] x, double[:] y,
                                                                                       int hops, int nlags,
                                                                                       double cr,
                                                                                       int num_threads)

cpdef np.ndarray[np.float, ndim=1] get_initial_distribution(double[:] y, int nlags)