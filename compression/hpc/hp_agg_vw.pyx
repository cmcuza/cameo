# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, infer_types=True
from compression.lpc cimport heap
from compression.lpc cimport vw
from compression.hpc cimport hp_math_lib
from compression.lpc.heap cimport Heap, Node
from libcpp.unordered_map cimport unordered_map
from libc.stdlib cimport malloc, free
from compression.hpc cimport hp_acf_agg_model
from compression.hpc.hp_acf_agg_model cimport HPAcfAgg
from numpy.math cimport INFINITY
import numpy as np
cimport numpy as np


cpdef np.ndarray[np.uint8_t, ndim=1] simplify_by_agg_vw(long [:] x, double[:] y, Py_ssize_t nlags, Py_ssize_t kappa, double acf_threshold):
    cdef:
        Py_ssize_t start, end, n = x.shape[0]
        double ace = 0.0, x_a, right_area, left_area, inf = INFINITY
        double[:] real_areas
        long double[:] aggregates = np.empty(n // kappa, dtype=np.longdouble)
        long double * raw_acf = <long double *> malloc(nlags * sizeof(long double))
        long double * c_acf = <long double *> malloc(nlags * sizeof(long double))
        Heap * area_heap = <Heap *> malloc(sizeof(Heap))
        HPAcfAgg * acf_model = <HPAcfAgg *> malloc(sizeof(HPAcfAgg))
        unordered_map[Py_ssize_t, Py_ssize_t] map_node_to_heap
        Node min_node, left, right
        np.ndarray[np.uint8_t, ndim=1] no_removed_indices = np.ones(x.shape[0], dtype=bool)

    real_areas = vw.triangle_areas_from_array(x, y) # computing the areas for all triangles
    heap.initialize_vw(area_heap, map_node_to_heap, real_areas) # Initialize the heap
    hp_acf_agg_model.initialize(acf_model, nlags) # initialize the aggregates
    hp_acf_agg_model.fit(acf_model, y, aggregates, kappa) # extract the aggregates
    hp_acf_agg_model.get_acf(acf_model, raw_acf) # get raw acf

    while area_heap.values[0].value < inf:
        min_node = heap.pop(area_heap, map_node_to_heap) # TODO: make it a reference

        if min_node.value != 0:
            start = min_node.left
            end = min_node.right
            if start + 2 < end:
                hp_acf_agg_model.interpolate_update(acf_model, y, aggregates, start, end, kappa)
            else:
                x_a = (y[end]-y[start]) / (end-start) + y[start]
                hp_acf_agg_model.update(acf_model, y, aggregates, x_a, start + 1, kappa)

            hp_acf_agg_model.get_acf(acf_model, c_acf)
            ace = hp_math_lib.mae(raw_acf, c_acf, nlags)

        if ace >= acf_threshold:
            break

        no_removed_indices[min_node.ts] = False

        heap.get_update_left_right(area_heap, map_node_to_heap, min_node, left, right)

        if right.ts != -1:
            right_area = hp_math_lib.triangle_area(x[right.ts], y[right.ts],
                                       x[right.left], y[right.left],
                                       x[right.right], y[right.right])

            if right_area <= min_node.value:
                right_area = min_node.value

            right.value = right_area
            heap.reheap(area_heap, map_node_to_heap, right)
        if left.ts != -1:
            left_area = hp_math_lib.triangle_area(x[left.ts], y[left.ts],
                                      x[left.left], y[left.left],
                                      x[left.right], y[left.right])

            if left_area <= min_node.value:
                left_area = min_node.value

            left.value = left_area
            heap.reheap(area_heap, map_node_to_heap, left)

    heap.release_memory(area_heap)
    hp_acf_agg_model.release_memory(acf_model)
    free(raw_acf)
    free(c_acf)

    return no_removed_indices