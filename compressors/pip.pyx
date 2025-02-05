from compressors.pip_heap cimport PIPHeap, PIPNode
from compressors.inc_acf cimport AcfAgg
from compressors cimport pip_heap
from compressors cimport inc_acf
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, fabs
from numpy.math cimport INFINITY
import numpy as np
cimport numpy as np
cimport cython


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef simplify(double[:] y, int nlags, double acf_threshold):
    cdef:
        int N = y.shape[0], i, start, left, order, n = y.shape[0]
        AcfAgg * acf_agg = <AcfAgg *> malloc(sizeof(AcfAgg))
        PIPNode min_node
        PIPHeap * pip_importance_heap = <PIPHeap *> malloc(sizeof(PIPHeap))
        double * raw_acf = <double *> malloc(nlags * sizeof(double))
        long [:] non_removed_points = np.zeros(y.shape[0]).astype(int)

    pip_heap.init_heap(pip_importance_heap, N)
    inc_acf.initialize(acf_agg, nlags)  # initialize the aggregates
    inc_acf.fit(acf_agg, y)  # extract the aggregates
    inc_acf.get_acf(acf_agg, raw_acf)  # get raw acf

    for i in range(N):
        pip_heap.add(pip_importance_heap, i, y[i])
    order = 1
    while pip_importance_heap.values[0].value < INFINITY:
        min_node = pip_heap.remove_at(pip_importance_heap, 0)
        start = min_node.left.ts
        end = min_node.right.ts
        if start + 2 < end:
            inc_acf.interpolate_update(acf_agg, y, start, end)
        else:
            x_a = (y[end]-y[start]) / (end-start) + y[start]
            inc_acf.update(acf_agg, y, x_a, start + 1)

        ace = 0.0
        n = y.shape[0]
        for lag in range(acf_agg.nlags):
            n -= 1
            c_acf = (n * acf_agg.sxy[lag] - acf_agg.xs[lag] * acf_agg.ys[lag]) / \
                    sqrt((n * acf_agg.xss[lag] - acf_agg.xs[lag] * acf_agg.xs[lag]) *
                         (n * acf_agg.yss[lag] - acf_agg.ys[lag] * acf_agg.ys[lag]))
            ace += fabs(raw_acf[lag] - c_acf)

        ace /= acf_agg.nlags

        if ace >= acf_threshold or N/(N - order) >= 10:
            break

        non_removed_points[min_node.ts] = order
        order += 1

    pip_heap.deinit_heap(pip_importance_heap)
    inc_acf.release_memory(acf_agg)
    free(pip_importance_heap)
    free(raw_acf)


    return non_removed_points