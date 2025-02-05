from compressors.heap cimport Heap, Node
from libcpp.unordered_map cimport unordered_map
from compressors.inc_acf cimport AcfAgg
import numpy as np
cimport numpy as np


cdef void look_ahead_reheap_mean(AcfAgg *acf_agg, Heap *acf_errors, unordered_map[int, int] &map_node_to_heap,
                                 double [:]y, double[:] aggregates, double *raw_acf, const Node &removed_node, int &hops, int kappa)

cdef void look_ahead_reheap_sum(AcfAgg *acf_agg, Heap *acf_errors, unordered_map[int, int] &map_node_to_heap,
                                double [:]y, double[:] aggregates, double *raw_acf, const Node &removed_node, int &hops, int kappa)

cdef void parallel_look_ahead_reheap_sum(AcfAgg *acf_model, Heap *acf_errors, unordered_map[int, int] &map_node_to_heap,
                                         double [:]y, double[:] aggregates, double *raw_acf,
                                         const Node &removed_node, int &hops, int kappa, int num_threads)

cdef void parallel_look_ahead_reheap_sum_mse(AcfAgg *acf_model, Heap *acf_errors, unordered_map[int, int] &map_node_to_heap,
                                             double [:]y, double[:] aggregates, double *raw_acf,
                                             const Node &removed_node, int &hops, int kappa, int num_threads)

cdef void parallel_look_ahead_reheap_mean(AcfAgg *acf_model, Heap *acf_errors, unordered_map[int, int] &map_node_to_heap,
                                          double [:]y, double[:] aggregates, double *raw_acf,
                                          const Node &removed_node, int &hops, int kappa, int num_threads)

cpdef np.ndarray[np.uint8_t, ndim=1] simplify_single_thread_mean(double[:] y, int hops, int kappa,
                                                                 int nlags, double acf_threshold)

cpdef np.ndarray[np.uint8_t, ndim=1] simplify_single_thread_sum(double[:] y, int hops, int kappa,
                                                                int nlags, double acf_threshold)

cpdef np.ndarray[np.uint8_t, ndim=1] parallel_simplify_by_blocking_mean(double[:] y, int hops, int nlags,
                                                                        int kappa, double acf_threshold, int num_threads)

cpdef np.ndarray[np.uint8_t, ndim=1] parallel_simplify_by_blocking_sum(double[:] y, int hops, int nlags,
                                                                       int kappa, double acf_threshold, int num_threads)

cpdef np.ndarray[np.uint8_t, ndim=1] parallel_simplify_by_blocking_sum_mse(double[:] y, int hops, int nlags,
                                                                           int kappa, double acf_threshold, int num_threads)

cpdef np.ndarray[np.uint8_t, ndim=1] parallel_simplify_by_blocking_max(double[:] y, int hops,
                                                                       int nlags, int kappa, double acf_threshold, int num_threads)
