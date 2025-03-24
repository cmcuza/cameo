from compression.lpc.heap_swab cimport Heap, Segment
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libcpp.utility cimport pair
cimport numpy as np


cdef double sse_of_segment(int seg_start, int seg_end, int[:] xs, double[:] ys)

cdef double merge_cost(Segment seg_a, Segment seg_b, int[:] xs, double[:] ys)

cdef void build_segments_pairwise(Heap * heap, unordered_map[int, int] &map_node_to_heap, int [:] xs, double [:] ys)

cdef vector[pair[int, int]] bottom_up_aux(int [:]xs, double [:]ys, int start_idx, int end_idx, double max_error)

cpdef np.ndarray[np.uint8_t, ndim=1] bottom_up(int [:]xs, double [:]ys, double max_error)

cdef void build_segments_pairwise_aux(Heap * heap, unordered_map[int, int] &map_node_to_heap, int [:] xs, double [:] ys,
                                      int start_idx, int end_idx)