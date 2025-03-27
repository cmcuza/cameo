# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, infer_types=True
from compression.lpc.pip_heap cimport PIPHeap, PIPNode
from compression.hpc.hp_acf_agg_model cimport HPAcfAgg
from compression.lpc cimport pip_heap
from compression.hpc cimport hp_acf_agg_model
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, fabs
from numpy.math cimport INFINITY
import numpy as np
cimport numpy as np


cpdef simplify_by_agg_pip(double[:] y, Py_ssize_t nlags, Py_ssize_t kappa, double acf_threshold):
    cdef:
        Py_ssize_t N = y.shape[0], i, start, left, n
        HPAcfAgg * acf_agg = <HPAcfAgg *> malloc(sizeof(HPAcfAgg))
        long double[:] aggregates = np.empty(N // kappa, dtype=np.longdouble)
        PIPNode min_node
        long double c_acf, ace
        PIPHeap * pip_importance_heap = <PIPHeap *> malloc(sizeof(PIPHeap))
        long double * raw_acf = <long double *> malloc(nlags * sizeof(long double))
        np.ndarray[np.uint8_t, ndim=1] non_removed_points = np.ones(y.shape[0], dtype=bool)

    pip_heap.init_heap(pip_importance_heap, N)
    hp_acf_agg_model.initialize(acf_agg, nlags)  # initialize the aggregates
    hp_acf_agg_model.fit(acf_agg, y, aggregates, kappa)  # extract the aggregates
    hp_acf_agg_model.get_acf(acf_agg, raw_acf)  # get raw acf

    for i in range(N):
        pip_heap.add(pip_importance_heap, i, y[i])

    while pip_importance_heap.values[0].value < INFINITY:
        min_node = pip_heap.remove_at(pip_importance_heap, 0)
        start = min_node.left.ts
        end = min_node.right.ts
        if start + 2 < end:
            hp_acf_agg_model.interpolate_update(acf_agg, y, aggregates, start, end, kappa)
        else:
            x_a = (y[end]-y[start]) / (end-start) + y[start]
            hp_acf_agg_model.update(acf_agg, y, aggregates, x_a, start + 1, kappa)

        ace = 0.0
        n = acf_agg.n
        for lag in range(acf_agg.nlags):
            n -= 1
            c_acf = (n * acf_agg.sxy[lag] - acf_agg.xs[lag] * acf_agg.ys[lag]) / \
                    sqrt((n * acf_agg.xss[lag] - acf_agg.xs[lag] * acf_agg.xs[lag]) *
                         (n * acf_agg.yss[lag] - acf_agg.ys[lag] * acf_agg.ys[lag]))
            ace += fabs(raw_acf[lag] - c_acf)

        ace /= acf_agg.nlags

        if ace >= acf_threshold:
            break

        non_removed_points[min_node.ts] = False

    pip_heap.deinit_heap(pip_importance_heap)
    hp_acf_agg_model.release_memory(acf_agg)
    free(pip_importance_heap)
    free(raw_acf)


    return non_removed_points