from compression.lpc cimport heap, math_utils
from compression.lpc.heap cimport Heap, Node
from libcpp.unordered_map cimport unordered_map
from cython.parallel cimport prange, parallel
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, fabs
cimport cython
from compression.lpc cimport inc_acf_agg, inc_acf
from compression.lpc.inc_acf cimport AcfAgg
from numpy.math cimport INFINITY
import numpy as np
cimport numpy as np

cdef extern from "fenv.h":
    int fesetround(int)
    int fegetround()

# Define rounding modes manually
FE_TONEAREST = 0
FE_DOWNWARD = 1
FE_UPWARD = 2
FE_TOWARDZERO = 3

def set_rounding_mode():
    fesetround(FE_TONEAREST)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void parallel_look_ahead_reheap_mean(AcfAgg *acf_model, Heap *acf_errors, unordered_map[int, int] &map_node_to_heap,
                                     double [:]y, double[:] aggregates, double *raw_acf,
                                     const Node &removed_node, int &hops, int kappa, int num_threads):
    cdef:
        int i, ii, left_node_index, right_node_index, start, end, end_index_a, start_index_a,\
            real_size, n, lag, index, num_lags, max_nd, agg_index, num_agg_deltas, diff
        double x_a, delta, delta_ss, ys, yss, xs, xss, sxy, lag_acf, slope
        Node neighbor_node
        Node * neighbors
        double *nb_imp
        double *sum_agg_deltas
        double *sxy_s
        double *ys_s
        double *yss_s
        double *xs_s
        double *xss_s

    real_size = 0
    left_node_index = removed_node.left
    right_node_index = removed_node.right
    neighbors = <Node *> malloc(2 * hops * sizeof(Node))

    # Look for neighbor nodes left and right and store them
    max_nd = 2
    for i in range(hops):
        if left_node_index > 0:
            left_node = acf_errors.values[map_node_to_heap[left_node_index]]
            neighbors[real_size] = left_node
            left_node_index = left_node.left
            diff = left_node.right//kappa - left_node_index//kappa + 1
            if diff > max_nd:
                max_nd = diff + 1
            real_size += 1
        if right_node_index < y.shape[0] - 1:
            right_node = acf_errors.values[map_node_to_heap[right_node_index]]
            neighbors[real_size] = right_node
            right_node_index = right_node.right
            diff = right_node_index//kappa - right_node.left//kappa + 1
            if diff > max_nd:
                max_nd = diff + 1
            real_size += 1

    nb_imp = <double *> malloc(real_size * sizeof(double))

    with nogil, parallel(num_threads=num_threads):
        sxy_s = <double *> malloc(acf_model.nlags * sizeof(double))
        ys_s = <double *> malloc(acf_model.nlags * sizeof(double))
        yss_s = <double *> malloc(acf_model.nlags * sizeof(double))
        xss_s = <double *> malloc(acf_model.nlags * sizeof(double))
        xs_s = <double *> malloc(acf_model.nlags * sizeof(double))
        sum_agg_deltas = <double *> malloc(max_nd * sizeof(double))

        for i in prange(real_size):
            neighbor_node = neighbors[i]
            start = neighbor_node.left
            end = neighbor_node.right

            if start + 2 < end:
                slope = (y[end] - y[start]) / (end - start)

                for lag in range(acf_model.nlags):
                    ys_s[lag] = acf_model.ys[lag]
                    yss_s[lag] = acf_model.yss[lag]
                    sxy_s[lag] = acf_model.sxy[lag]
                    xs_s[lag] = acf_model.xs[lag]
                    xss_s[lag] = acf_model.xss[lag]

                start_index_a = (start + 1) // kappa
                end_index_a = (end - 1) // kappa

                num_agg_deltas = end_index_a - start_index_a + 1
                for ii in range(num_agg_deltas):
                    sum_agg_deltas[ii] = 0.0

                for index in range(start + 1, end):
                    agg_index = index // kappa
                    sum_agg_deltas[agg_index - start_index_a] += (slope * (index - start) + y[start] - y[index])

                ii = 0
                n = acf_model.n - 1
                for agg_index in range(start_index_a, end_index_a + 1):
                    sum_agg_deltas[ii] /= kappa # aggregate function
                    delta = sum_agg_deltas[ii]
                    delta_ss = delta * (delta + 2 * aggregates[agg_index])
                    for lag in range(acf_model.nlags):
                        if agg_index > lag:
                            ys_s[lag] += delta
                            yss_s[lag] += delta_ss
                            sxy_s[lag] += delta * aggregates[agg_index - lag - 1]
                        if agg_index < n - lag:
                            xs_s[lag] += delta
                            xss_s[lag] += delta_ss
                            sxy_s[lag] += delta * aggregates[agg_index + lag + 1]

                    ii = ii + 1

                num_lags = num_agg_deltas if num_agg_deltas < acf_model.nlags else acf_model.nlags
                nb_imp[i] = 0
                for lag in range(acf_model.nlags):
                    for ii in range(num_agg_deltas - lag - 1):
                        sxy_s[lag] = sxy_s[lag] + sum_agg_deltas[ii] * sum_agg_deltas[ii + lag + 1]
                    lag_acf = (n * sxy_s[lag] - xs_s[lag] * ys_s[lag]) / sqrt(
                        (n * xss_s[lag] - xs_s[lag] * xs_s[lag]) * (n * yss_s[lag] - ys_s[lag] * ys_s[lag]))
                    nb_imp[i] += fabs(lag_acf - raw_acf[lag])
                    n -= 1

                nb_imp[i] /= acf_model.nlags
            else:
                x_a = (y[end] - y[start]) / (end - start) + y[start]
                start = start + 1
                agg_index = start // kappa
                delta = (x_a - y[start])/kappa # aggregate function
                delta_ss = delta * (2 * aggregates[agg_index] + delta)
                nb_imp[i] = 0
                n = acf_model.n - 1
                if delta != 0:
                    for lag in range(acf_model.nlags):
                        ys = acf_model.ys[lag]
                        yss = acf_model.yss[lag]
                        xs = acf_model.xs[lag]
                        xss = acf_model.xss[lag]
                        sxy = acf_model.sxy[lag]
                        if agg_index > lag:
                            ys = ys + delta
                            yss = yss + delta_ss
                            sxy = sxy + delta * aggregates[agg_index - lag - 1]
                        if agg_index < n-lag:
                            xs = xs + delta
                            xss = xss + delta_ss
                            sxy = sxy + delta * aggregates[agg_index + lag + 1]
                        lag_acf = (n * sxy - xs * ys) / sqrt((n * xss - xs * xs) * (n * yss - ys * ys))
                        nb_imp[i] += fabs(lag_acf - raw_acf[lag])
                        n = n - 1

                    nb_imp[i] /= acf_model.nlags

        free(sum_agg_deltas)
        free(sxy_s)
        free(ys_s)
        free(yss_s)
        free(xss_s)
        free(xs_s)

    for i in range(real_size):
        neighbor_node = neighbors[i]
        neighbor_node.value = nb_imp[i]
        heap.reheap(acf_errors, map_node_to_heap, neighbor_node)

    free(neighbors)
    free(nb_imp)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void parallel_look_ahead_reheap_sum(AcfAgg *acf_model, Heap *acf_errors, unordered_map[int, int] &map_node_to_heap,
                                         double [:]y, double[:] aggregates, double *raw_acf,
                                         const Node &removed_node, int &hops, int kappa, int num_threads):
    cdef:
        int i, ii, left_node_index, right_node_index, start, end, end_index_a, start_index_a,\
            real_size, n, lag, index, num_lags, max_nd, agg_index, num_agg_deltas, diff
        double x_a, delta, delta_ss, ys, yss, xs, xss, sxy, lag_acf, slope
        Node neighbor_node
        Node * neighbors
        double *nb_imp
        double *sum_agg_deltas
        double *sxy_s
        double *ys_s
        double *yss_s
        double *xs_s
        double *xss_s

    real_size = 0
    left_node_index = removed_node.left
    right_node_index = removed_node.right
    neighbors = <Node *> malloc(2 * hops * sizeof(Node))

    # Look for neighbor nodes left and right and store them
    max_nd = 2
    for i in range(hops):
        if left_node_index > 0:
            left_node = acf_errors.values[map_node_to_heap[left_node_index]]
            neighbors[real_size] = left_node
            left_node_index = left_node.left
            diff = left_node.right//kappa - left_node_index//kappa + 1
            if diff > max_nd:
                max_nd = diff + 1
            real_size += 1
        if right_node_index < y.shape[0] - 1:
            right_node = acf_errors.values[map_node_to_heap[right_node_index]]
            neighbors[real_size] = right_node
            right_node_index = right_node.right
            diff = right_node_index//kappa - right_node.left//kappa + 1
            if diff > max_nd:
                max_nd = diff + 1
            real_size += 1

    nb_imp = <double *> malloc(real_size * sizeof(double))

    with nogil, parallel(num_threads=num_threads):
        sxy_s = <double *> malloc(acf_model.nlags * sizeof(double))
        ys_s = <double *> malloc(acf_model.nlags * sizeof(double))
        yss_s = <double *> malloc(acf_model.nlags * sizeof(double))
        xss_s = <double *> malloc(acf_model.nlags * sizeof(double))
        xs_s = <double *> malloc(acf_model.nlags * sizeof(double))
        sum_agg_deltas = <double *> malloc(max_nd * sizeof(double))

        for i in prange(real_size):
            neighbor_node = neighbors[i]
            start = neighbor_node.left
            end = neighbor_node.right

            if start + 2 < end:
                slope = (y[end] - y[start]) / (end - start)

                for lag in range(acf_model.nlags):
                    ys_s[lag] = acf_model.ys[lag]
                    yss_s[lag] = acf_model.yss[lag]
                    sxy_s[lag] = acf_model.sxy[lag]
                    xs_s[lag] = acf_model.xs[lag]
                    xss_s[lag] = acf_model.xss[lag]

                start_index_a = (start + 1) // kappa
                end_index_a = (end - 1) // kappa

                num_agg_deltas = end_index_a - start_index_a + 1
                for ii in range(num_agg_deltas):
                    sum_agg_deltas[ii] = 0.0

                for index in range(start + 1, end):
                    agg_index = index // kappa
                    sum_agg_deltas[agg_index - start_index_a] += (slope * (index - start) + y[start] - y[index])

                ii = 0
                n = acf_model.n - 1
                for agg_index in range(start_index_a, end_index_a + 1):
                    # sum_agg_deltas[ii] /= kappa # aggregate function
                    delta = sum_agg_deltas[ii]
                    delta_ss = delta * (delta + 2 * aggregates[agg_index])
                    for lag in range(acf_model.nlags):
                        if agg_index > lag:
                            ys_s[lag] += delta
                            yss_s[lag] += delta_ss
                            sxy_s[lag] += delta * aggregates[agg_index - lag - 1]
                        if agg_index < n - lag:
                            xs_s[lag] += delta
                            xss_s[lag] += delta_ss
                            sxy_s[lag] += delta * aggregates[agg_index + lag + 1]

                    ii = ii + 1

                num_lags = num_agg_deltas if num_agg_deltas < acf_model.nlags else acf_model.nlags
                nb_imp[i] = 0
                for lag in range(acf_model.nlags):
                    for ii in range(num_agg_deltas - lag - 1):
                        sxy_s[lag] = sxy_s[lag] + sum_agg_deltas[ii] * sum_agg_deltas[ii + lag + 1]
                    lag_acf = (n * sxy_s[lag] - xs_s[lag] * ys_s[lag]) / sqrt(
                        (n * xss_s[lag] - xs_s[lag] * xs_s[lag]) * (n * yss_s[lag] - ys_s[lag] * ys_s[lag]))
                    nb_imp[i] += fabs(lag_acf - raw_acf[lag])
                    n -= 1

                nb_imp[i] /= acf_model.nlags
            else:
                x_a = (y[end] - y[start]) / (end - start) + y[start]
                start = start + 1
                agg_index = start // kappa
                delta = (x_a - y[start]) # /kappa # aggregate function
                delta_ss = delta * (2 * aggregates[agg_index] + delta)
                nb_imp[i] = 0
                n = acf_model.n - 1
                if delta != 0:
                    for lag in range(acf_model.nlags):
                        ys = acf_model.ys[lag]
                        yss = acf_model.yss[lag]
                        xs = acf_model.xs[lag]
                        xss = acf_model.xss[lag]
                        sxy = acf_model.sxy[lag]
                        if agg_index > lag:
                            ys = ys + delta
                            yss = yss + delta_ss
                            sxy = sxy + delta * aggregates[agg_index - lag - 1]
                        if agg_index < n-lag:
                            xs = xs + delta
                            xss = xss + delta_ss
                            sxy = sxy + delta * aggregates[agg_index + lag + 1]
                        lag_acf = (n * sxy - xs * ys) / sqrt((n * xss - xs * xs) * (n * yss - ys * ys))
                        nb_imp[i] += fabs(lag_acf - raw_acf[lag])
                        n = n - 1

                    nb_imp[i] /= acf_model.nlags

        free(sum_agg_deltas)
        free(sxy_s)
        free(ys_s)
        free(yss_s)
        free(xss_s)
        free(xs_s)

    for i in range(real_size):
        neighbor_node = neighbors[i]
        neighbor_node.value = nb_imp[i]
        heap.reheap(acf_errors, map_node_to_heap, neighbor_node)

    free(neighbors)
    free(nb_imp)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void parallel_look_ahead_reheap_max(AcfAgg *acf_model, Heap *acf_errors, unordered_map[int, int] &map_node_to_heap,
                                         double [:]y, double[:] aggregates, double *raw_acf,
                                         const Node &removed_node, int &hops, int kappa, int num_threads):
    cdef:
        int i, ii, left_node_index, right_node_index, start, end, end_index_a, start_index_a,\
            real_size, n, lag, index, num_lags, max_nd, agg_index, num_agg_deltas, diff
        double x_a, delta, delta_ss, ys, yss, xs, xss, sxy, lag_acf, slope
        Node neighbor_node
        Node * neighbors
        double *nb_imp
        double *deltas
        double *sxy_s
        double *ys_s
        double *yss_s
        double *xs_s
        double *xss_s

    real_size = 0
    left_node_index = removed_node.left
    right_node_index = removed_node.right
    neighbors = <Node *> malloc(2 * hops * sizeof(Node))

    # Look for neighbor nodes left and right and store them
    max_nd = 2
    for i in range(hops):
        if left_node_index > 0:
            left_node = acf_errors.values[map_node_to_heap[left_node_index]]
            neighbors[real_size] = left_node
            left_node_index = left_node.left
            diff = left_node.right//kappa - left_node_index//kappa + 1
            if diff > max_nd:
                max_nd = diff + 1
            real_size += 1
        if right_node_index < y.shape[0] - 1:
            right_node = acf_errors.values[map_node_to_heap[right_node_index]]
            neighbors[real_size] = right_node
            right_node_index = right_node.right
            diff = right_node_index//kappa - right_node.left//kappa + 1
            if diff > max_nd:
                max_nd = diff + 1
            real_size += 1

    nb_imp = <double *> malloc(real_size * sizeof(double))

    with nogil, parallel(num_threads=num_threads):
        sxy_s = <double *> malloc(acf_model.nlags * sizeof(double))
        ys_s = <double *> malloc(acf_model.nlags * sizeof(double))
        yss_s = <double *> malloc(acf_model.nlags * sizeof(double))
        xss_s = <double *> malloc(acf_model.nlags * sizeof(double))
        xs_s = <double *> malloc(acf_model.nlags * sizeof(double))
        deltas = <double *> malloc(max_nd * sizeof(double))

        for i in prange(real_size):
            neighbor_node = neighbors[i]
            start = neighbor_node.left
            end = neighbor_node.right

            if start + 2 < end:
                slope = (y[end] - y[start]) / (end - start)

                for lag in range(acf_model.nlags):
                    ys_s[lag] = acf_model.ys[lag]
                    yss_s[lag] = acf_model.yss[lag]
                    sxy_s[lag] = acf_model.sxy[lag]
                    xs_s[lag] = acf_model.xs[lag]
                    xss_s[lag] = acf_model.xss[lag]

                start_index_a = (start + 1) // kappa
                end_index_a = (end - 1) // kappa

                num_agg_deltas = end_index_a - start_index_a + 1
                for ii in range(num_agg_deltas):
                    deltas[ii] = -INFINITY

                for index in range(start + 1, end):
                    x_a = slope * (index - start) + y[start]
                    agg_index = index // kappa
                    ii = agg_index - start_index_a
                    deltas[ii] = max(x_a, deltas[ii])

                ii = 0
                n = acf_model.n - 1
                for agg_index in range(start_index_a, end_index_a + 1):
                    deltas[ii] = deltas[ii] - aggregates[agg_index] # aggregate function
                    delta = deltas[ii]
                    delta_ss = delta * (delta + 2 * aggregates[agg_index])
                    for lag in range(acf_model.nlags):
                        if agg_index > lag:
                            ys_s[lag] += delta
                            yss_s[lag] += delta_ss
                            sxy_s[lag] += delta * aggregates[agg_index - lag - 1]
                        if agg_index < n - lag:
                            xs_s[lag] += delta
                            xss_s[lag] += delta_ss
                            sxy_s[lag] += delta * aggregates[agg_index + lag + 1]

                    ii = ii + 1

                num_lags = num_agg_deltas if num_agg_deltas < acf_model.nlags else acf_model.nlags
                nb_imp[i] = 0
                for lag in range(acf_model.nlags):
                    for ii in range(num_agg_deltas - lag - 1):
                        sxy_s[lag] = sxy_s[lag] + deltas[ii] * deltas[ii + lag + 1]
                    lag_acf = (n * sxy_s[lag] - xs_s[lag] * ys_s[lag]) / sqrt(
                        (n * xss_s[lag] - xs_s[lag] * xs_s[lag]) * (n * yss_s[lag] - ys_s[lag] * ys_s[lag]))
                    nb_imp[i] += fabs(lag_acf - raw_acf[lag])
                    n -= 1

                nb_imp[i] /= acf_model.nlags
            else:
                x_a = (y[end] - y[start]) / (end - start) + y[start]
                start = start + 1
                agg_index = start // kappa
                delta = max(x_a, aggregates[agg_index]) - aggregates[agg_index] # /kappa # aggregate function
                delta_ss = delta * (2 * aggregates[agg_index] + delta)
                nb_imp[i] = 0
                n = acf_model.n - 1
                if delta != 0:
                    for lag in range(acf_model.nlags):
                        ys = acf_model.ys[lag]
                        yss = acf_model.yss[lag]
                        xs = acf_model.xs[lag]
                        xss = acf_model.xss[lag]
                        sxy = acf_model.sxy[lag]
                        if agg_index > lag:
                            ys = ys + delta
                            yss = yss + delta_ss
                            sxy = sxy + delta * aggregates[agg_index - lag - 1]
                        if agg_index < n-lag:
                            xs = xs + delta
                            xss = xss + delta_ss
                            sxy = sxy + delta * aggregates[agg_index + lag + 1]
                        lag_acf = (n * sxy - xs * ys) / sqrt((n * xss - xs * xs) * (n * yss - ys * ys))
                        nb_imp[i] += fabs(lag_acf - raw_acf[lag])
                        n = n - 1

                    nb_imp[i] /= acf_model.nlags

        free(deltas)
        free(sxy_s)
        free(ys_s)
        free(yss_s)
        free(xss_s)
        free(xs_s)

    for i in range(real_size):
        neighbor_node = neighbors[i]
        neighbor_node.value = nb_imp[i]
        heap.reheap(acf_errors, map_node_to_heap, neighbor_node)

    free(neighbors)
    free(nb_imp)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void parallel_look_ahead_reheap_sum_mse(AcfAgg *acf_model, Heap *acf_errors, unordered_map[int, int] &map_node_to_heap,
                                     double [:]y, double[:] aggregates, double *raw_acf,
                                     const Node &removed_node, int &hops, int kappa, int num_threads):
    cdef:
        int i, ii, left_node_index, right_node_index, start, end, end_index_a, start_index_a,\
            real_size, n, lag, index, num_lags, max_nd, agg_index, num_agg_deltas, diff
        double x_a, delta, delta_ss, ys, yss, xs, xss, sxy, lag_acf, slope
        Node neighbor_node
        Node * neighbors
        double *nb_imp
        double *sum_agg_deltas
        double *sxy_s
        double *ys_s
        double *yss_s
        double *xs_s
        double *xss_s

    real_size = 0
    left_node_index = removed_node.left
    right_node_index = removed_node.right
    neighbors = <Node *> malloc(2 * hops * sizeof(Node))

    # Look for neighbor nodes left and right and store them
    max_nd = 2
    for i in range(hops):
        if left_node_index > 0:
            left_node = acf_errors.values[map_node_to_heap[left_node_index]]
            neighbors[real_size] = left_node
            left_node_index = left_node.left
            diff = left_node.right//kappa - left_node_index//kappa + 1
            if diff > max_nd:
                max_nd = diff + 1
            real_size += 1
        if right_node_index < y.shape[0] - 1:
            right_node = acf_errors.values[map_node_to_heap[right_node_index]]
            neighbors[real_size] = right_node
            right_node_index = right_node.right
            diff = right_node_index//kappa - right_node.left//kappa + 1
            if diff > max_nd:
                max_nd = diff + 1
            real_size += 1

    nb_imp = <double *> malloc(real_size * sizeof(double))

    with nogil, parallel(num_threads=num_threads):
        sxy_s = <double *> malloc(acf_model.nlags * sizeof(double))
        ys_s = <double *> malloc(acf_model.nlags * sizeof(double))
        yss_s = <double *> malloc(acf_model.nlags * sizeof(double))
        xss_s = <double *> malloc(acf_model.nlags * sizeof(double))
        xs_s = <double *> malloc(acf_model.nlags * sizeof(double))
        sum_agg_deltas = <double *> malloc(max_nd * sizeof(double))

        for i in prange(real_size):
            neighbor_node = neighbors[i]
            start = neighbor_node.left
            end = neighbor_node.right

            if start + 2 < end:
                slope = (y[end] - y[start]) / (end - start)

                for lag in range(acf_model.nlags):
                    ys_s[lag] = acf_model.ys[lag]
                    yss_s[lag] = acf_model.yss[lag]
                    sxy_s[lag] = acf_model.sxy[lag]
                    xs_s[lag] = acf_model.xs[lag]
                    xss_s[lag] = acf_model.xss[lag]

                start_index_a = (start + 1) // kappa
                end_index_a = (end - 1) // kappa

                num_agg_deltas = end_index_a - start_index_a + 1
                for ii in range(num_agg_deltas):
                    sum_agg_deltas[ii] = 0.0

                for index in range(start + 1, end):
                    agg_index = index // kappa
                    sum_agg_deltas[agg_index - start_index_a] += (slope * (index - start) + y[start] - y[index])

                ii = 0
                n = acf_model.n - 1
                for agg_index in range(start_index_a, end_index_a + 1):
                    # sum_agg_deltas[ii] /= kappa # aggregate function
                    delta = sum_agg_deltas[ii]
                    delta_ss = delta * (delta + 2 * aggregates[agg_index])
                    for lag in range(acf_model.nlags):
                        if agg_index > lag:
                            ys_s[lag] += delta
                            yss_s[lag] += delta_ss
                            sxy_s[lag] += delta * aggregates[agg_index - lag - 1]
                        if agg_index < n - lag:
                            xs_s[lag] += delta
                            xss_s[lag] += delta_ss
                            sxy_s[lag] += delta * aggregates[agg_index + lag + 1]

                    ii = ii + 1

                num_lags = num_agg_deltas if num_agg_deltas < acf_model.nlags else acf_model.nlags
                nb_imp[i] = 0
                for lag in range(acf_model.nlags):
                    for ii in range(num_agg_deltas - lag - 1):
                        sxy_s[lag] = sxy_s[lag] + sum_agg_deltas[ii] * sum_agg_deltas[ii + lag + 1]
                    lag_acf = (n * sxy_s[lag] - xs_s[lag] * ys_s[lag]) / sqrt(
                        (n * xss_s[lag] - xs_s[lag] * xs_s[lag]) * (n * yss_s[lag] - ys_s[lag] * ys_s[lag]))
                    # nb_imp[i] += fabs(lag_acf - raw_acf[lag])
                    nb_imp[i] += (lag_acf - raw_acf[lag])*(lag_acf - raw_acf[lag])
                    n -= 1

                nb_imp[i] /= acf_model.nlags
            else:
                x_a = (y[end] - y[start]) / (end - start) + y[start]
                start = start + 1
                agg_index = start // kappa
                delta = (x_a - y[start]) # /kappa # aggregate function
                delta_ss = delta * (2 * aggregates[agg_index] + delta)
                nb_imp[i] = 0
                n = acf_model.n - 1
                if delta != 0:
                    for lag in range(acf_model.nlags):
                        ys = acf_model.ys[lag]
                        yss = acf_model.yss[lag]
                        xs = acf_model.xs[lag]
                        xss = acf_model.xss[lag]
                        sxy = acf_model.sxy[lag]
                        if agg_index > lag:
                            ys = ys + delta
                            yss = yss + delta_ss
                            sxy = sxy + delta * aggregates[agg_index - lag - 1]
                        if agg_index < n-lag:
                            xs = xs + delta
                            xss = xss + delta_ss
                            sxy = sxy + delta * aggregates[agg_index + lag + 1]
                        lag_acf = (n * sxy - xs * ys) / sqrt((n * xss - xs * xs) * (n * yss - ys * ys))
                        nb_imp[i] += (lag_acf - raw_acf[lag])*(lag_acf - raw_acf[lag])
                        n = n - 1

                    nb_imp[i] /= acf_model.nlags

        free(sum_agg_deltas)
        free(sxy_s)
        free(ys_s)
        free(yss_s)
        free(xss_s)
        free(xs_s)

    for i in range(real_size):
        neighbor_node = neighbors[i]
        neighbor_node.value = nb_imp[i]
        heap.reheap(acf_errors, map_node_to_heap, neighbor_node)

    free(neighbors)
    free(nb_imp)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void look_ahead_reheap_mean(AcfAgg *acf_model, Heap *acf_errors, unordered_map[int, int] &map_node_to_heap,
                                 double [:]y, double[:] aggregates, double *raw_acf,
                                 const Node &removed_node, int &hops, int kappa):
    cdef:
        int i, ii, left_node_index, right_node_index, start, end, end_index_a, start_index_a,\
            real_size, n, lag, index, num_lags, max_nd, agg_index, num_agg_deltas, diff
        double x_a, delta, delta_ss, ys, yss, xs, xss, sxy, lag_acf, slope
        Node neighbor_node
        Node * neighbors = <Node *> malloc(2 * hops * sizeof(Node))
        double *nb_imp
        double *sum_agg_deltas
        double *sxy_s
        double *ys_s
        double *yss_s
        double *xs_s
        double *xss_s

    real_size = 0
    left_node_index = removed_node.left
    right_node_index = removed_node.right

    # Look for neighbor nodes left and right and store them
    max_nd = 2
    for i in range(hops):
        if left_node_index > 0:
            left_node = acf_errors.values[map_node_to_heap[left_node_index]]
            neighbors[real_size] = left_node
            left_node_index = left_node.left
            diff = left_node.right//kappa - left_node_index//kappa + 1
            if diff > max_nd:
                max_nd = diff + 1
            real_size += 1
        if right_node_index < y.shape[0] - 1:
            right_node = acf_errors.values[map_node_to_heap[right_node_index]]
            neighbors[real_size] = right_node
            right_node_index = right_node.right
            diff = right_node_index//kappa - right_node.left//kappa + 1
            if diff > max_nd:
                max_nd = diff + 1
            real_size += 1

    nb_imp = <double *> malloc(real_size * sizeof(double))
    sxy_s = <double *> malloc(acf_model.nlags * sizeof(double))
    ys_s = <double *> malloc(acf_model.nlags * sizeof(double))
    yss_s = <double *> malloc(acf_model.nlags * sizeof(double))
    xss_s = <double *> malloc(acf_model.nlags * sizeof(double))
    xs_s = <double *> malloc(acf_model.nlags * sizeof(double))
    sum_agg_deltas = <double *> malloc(max_nd * sizeof(double))

    for i in range(real_size):
        neighbor_node = neighbors[i]
        start = neighbor_node.left
        end = neighbor_node.right

        if start + 2 < end:
            slope = (y[end] - y[start]) / (end - start)

            for lag in range(acf_model.nlags):
                ys_s[lag] = acf_model.ys[lag]
                yss_s[lag] = acf_model.yss[lag]
                sxy_s[lag] = acf_model.sxy[lag]
                xs_s[lag] = acf_model.xs[lag]
                xss_s[lag] = acf_model.xss[lag]

            start_index_a = (start + 1) // kappa
            end_index_a = (end - 1) // kappa

            num_agg_deltas = end_index_a - start_index_a + 1
            for ii in range(num_agg_deltas):
                sum_agg_deltas[ii] = 0.0

            for index in range(start + 1, end):
                agg_index = index // kappa
                sum_agg_deltas[agg_index - start_index_a] += (slope * (index - start) + y[start] - y[index])

            ii = 0
            n = acf_model.n - 1

            for agg_index in range(start_index_a, end_index_a + 1):
                sum_agg_deltas[ii] /= kappa # aggregate function
                delta = sum_agg_deltas[ii]

                delta_ss = delta * (delta + 2 * aggregates[agg_index])
                for lag in range(acf_model.nlags):
                    if agg_index > lag:
                        ys_s[lag] += delta
                        yss_s[lag] += delta_ss
                        sxy_s[lag] += delta * aggregates[agg_index - lag - 1]
                    if agg_index < n - lag:
                        xs_s[lag] += delta
                        xss_s[lag] += delta_ss
                        sxy_s[lag] += delta * aggregates[agg_index + lag + 1]

                ii = ii + 1

            num_lags = num_agg_deltas if num_agg_deltas < acf_model.nlags else acf_model.nlags
            nb_imp[i] = 0
            for lag in range(acf_model.nlags):
                for ii in range(num_agg_deltas - lag - 1):
                    sxy_s[lag] = sxy_s[lag] + sum_agg_deltas[ii] * sum_agg_deltas[ii + lag + 1]
                lag_acf = (n * sxy_s[lag] - xs_s[lag] * ys_s[lag]) / sqrt(
                    (n * xss_s[lag] - xs_s[lag] * xs_s[lag]) * (n * yss_s[lag] - ys_s[lag] * ys_s[lag]))
                nb_imp[i] += fabs(lag_acf - raw_acf[lag])
                n -= 1

            nb_imp[i] /= acf_model.nlags
        else:
            x_a = (y[end] - y[start]) / (end - start) + y[start]
            start = start + 1
            agg_index = start // kappa
            delta = (x_a - y[start])/kappa
            delta_ss = delta * (2 * aggregates[agg_index] + delta)
            nb_imp[i] = 0
            n = acf_model.n - 1
            if delta != 0:
                for lag in range(acf_model.nlags):
                    ys = acf_model.ys[lag]
                    yss = acf_model.yss[lag]
                    xs = acf_model.xs[lag]
                    xss = acf_model.xss[lag]
                    sxy = acf_model.sxy[lag]
                    if agg_index > lag:
                        ys = ys + delta
                        yss = yss + delta_ss
                        sxy = sxy + delta * aggregates[agg_index - lag - 1]
                    if agg_index < n-lag:
                        xs = xs + delta
                        xss = xss + delta_ss
                        sxy = sxy + delta * aggregates[agg_index + lag + 1]
                    lag_acf = (n * sxy - xs * ys) / sqrt((n * xss - xs * xs) * (n * yss - ys * ys))
                    nb_imp[i] += fabs(lag_acf - raw_acf[lag])
                    n = n - 1

                nb_imp[i] /= acf_model.nlags

    for i in range(real_size):
        neighbor_node = neighbors[i]
        neighbor_node.value = nb_imp[i]
        heap.reheap(acf_errors, map_node_to_heap, neighbor_node)

    free(neighbors)
    free(nb_imp)
    free(sum_agg_deltas)
    free(sxy_s)
    free(ys_s)
    free(yss_s)
    free(xss_s)
    free(xs_s)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void look_ahead_reheap_sum(AcfAgg *acf_model, Heap *acf_errors, unordered_map[int, int] &map_node_to_heap,
                                 double [:]y, double[:] aggregates, double *raw_acf,
                                 const Node &removed_node, int &hops, int kappa):
    cdef:
        int i, ii, left_node_index, right_node_index, start, end, end_index_a, start_index_a,\
            real_size, n, lag, index, num_lags, max_nd, agg_index, num_agg_deltas, diff
        double x_a, delta, delta_ss, ys, yss, xs, xss, sxy, lag_acf, slope
        Node neighbor_node
        Node * neighbors = <Node *> malloc(2 * hops * sizeof(Node))
        double *nb_imp
        double *sum_agg_deltas
        double *sxy_s
        double *ys_s
        double *yss_s
        double *xs_s
        double *xss_s

    real_size = 0
    left_node_index = removed_node.left
    right_node_index = removed_node.right

    # Look for neighbor nodes left and right and store them
    max_nd = 2
    for i in range(hops):
        if left_node_index > 0:
            left_node = acf_errors.values[map_node_to_heap[left_node_index]]
            neighbors[real_size] = left_node
            left_node_index = left_node.left
            diff = left_node.right//kappa - left_node_index//kappa + 1
            if diff > max_nd:
                max_nd = diff + 1
            real_size += 1
        if right_node_index < y.shape[0] - 1:
            right_node = acf_errors.values[map_node_to_heap[right_node_index]]
            neighbors[real_size] = right_node
            right_node_index = right_node.right
            diff = right_node_index//kappa - right_node.left//kappa + 1
            if diff > max_nd:
                max_nd = diff + 1
            real_size += 1

    nb_imp = <double *> malloc(real_size * sizeof(double))
    sxy_s = <double *> malloc(acf_model.nlags * sizeof(double))
    ys_s = <double *> malloc(acf_model.nlags * sizeof(double))
    yss_s = <double *> malloc(acf_model.nlags * sizeof(double))
    xss_s = <double *> malloc(acf_model.nlags * sizeof(double))
    xs_s = <double *> malloc(acf_model.nlags * sizeof(double))
    sum_agg_deltas = <double *> malloc(max_nd * sizeof(double))

    for i in range(real_size):
        neighbor_node = neighbors[i]
        start = neighbor_node.left
        end = neighbor_node.right

        if start + 2 < end:
            slope = (y[end] - y[start]) / (end - start)

            for lag in range(acf_model.nlags):
                ys_s[lag] = acf_model.ys[lag]
                yss_s[lag] = acf_model.yss[lag]
                sxy_s[lag] = acf_model.sxy[lag]
                xs_s[lag] = acf_model.xs[lag]
                xss_s[lag] = acf_model.xss[lag]

            start_index_a = (start + 1) // kappa
            end_index_a = (end - 1) // kappa

            num_agg_deltas = end_index_a - start_index_a + 1
            for ii in range(num_agg_deltas):
                sum_agg_deltas[ii] = 0.0

            for index in range(start + 1, end):
                agg_index = index // kappa
                sum_agg_deltas[agg_index - start_index_a] += (slope * (index - start) + y[start] - y[index])

            ii = 0
            n = acf_model.n - 1

            for agg_index in range(start_index_a, end_index_a + 1):
                # sum_agg_deltas[ii] /= kappa # aggregate function
                delta = sum_agg_deltas[ii]

                delta_ss = delta * (delta + 2 * aggregates[agg_index])
                for lag in range(acf_model.nlags):
                    if agg_index > lag:
                        ys_s[lag] += delta
                        yss_s[lag] += delta_ss
                        sxy_s[lag] += delta * aggregates[agg_index - lag - 1]
                    if agg_index < n - lag:
                        xs_s[lag] += delta
                        xss_s[lag] += delta_ss
                        sxy_s[lag] += delta * aggregates[agg_index + lag + 1]

                ii = ii + 1

            num_lags = num_agg_deltas if num_agg_deltas < acf_model.nlags else acf_model.nlags
            nb_imp[i] = 0
            for lag in range(acf_model.nlags):
                for ii in range(num_agg_deltas - lag - 1):
                    sxy_s[lag] = sxy_s[lag] + sum_agg_deltas[ii] * sum_agg_deltas[ii + lag + 1]
                lag_acf = (n * sxy_s[lag] - xs_s[lag] * ys_s[lag]) / sqrt(
                    (n * xss_s[lag] - xs_s[lag] * xs_s[lag]) * (n * yss_s[lag] - ys_s[lag] * ys_s[lag]))
                nb_imp[i] += fabs(lag_acf - raw_acf[lag])
                n -= 1

            nb_imp[i] /= acf_model.nlags
        else:
            x_a = (y[end] - y[start]) / (end - start) + y[start]
            start = start + 1
            agg_index = start // kappa
            delta = (x_a - y[start]) #/kappa # aggregate function
            delta_ss = delta * (2 * aggregates[agg_index] + delta)
            nb_imp[i] = 0
            n = acf_model.n - 1
            if delta != 0:
                for lag in range(acf_model.nlags):
                    ys = acf_model.ys[lag]
                    yss = acf_model.yss[lag]
                    xs = acf_model.xs[lag]
                    xss = acf_model.xss[lag]
                    sxy = acf_model.sxy[lag]
                    if agg_index > lag:
                        ys = ys + delta
                        yss = yss + delta_ss
                        sxy = sxy + delta * aggregates[agg_index - lag - 1]
                    if agg_index < n-lag:
                        xs = xs + delta
                        xss = xss + delta_ss
                        sxy = sxy + delta * aggregates[agg_index + lag + 1]
                    lag_acf = (n * sxy - xs * ys) / sqrt((n * xss - xs * xs) * (n * yss - ys * ys))
                    nb_imp[i] += fabs(lag_acf - raw_acf[lag])
                    n = n - 1

                nb_imp[i] /= acf_model.nlags

    for i in range(real_size):
        neighbor_node = neighbors[i]
        neighbor_node.value = nb_imp[i]
        heap.reheap(acf_errors, map_node_to_heap, neighbor_node)

    free(neighbors)
    free(nb_imp)
    free(sum_agg_deltas)
    free(sxy_s)
    free(ys_s)
    free(yss_s)
    free(xss_s)
    free(xs_s)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.uint8_t, ndim=1] simplify_single_thread_mean(double[:] y, int hops,
                                                            int nlags, int kappa, double acf_threshold):
    cdef int start, end, lag, n, num_agg, i
    cdef double ace, x_a, right_area, left_area, c_acf, inf
    cdef double * raw_acf
    cdef double * error_values
    cdef double[:] aggregates
    cdef Heap * acf_errors
    cdef AcfAgg * acf_model
    cdef unordered_map[int, int] map_node_to_heap
    cdef Node min_node, left, right
    cdef np.ndarray[np.uint8_t, ndim=1] no_removed_indices
    cdef short int overflow

    set_rounding_mode()


    n = y.shape[0]
    num_agg = n // kappa
    overflow = 0 if n % kappa == 0 else 1

    i = 0
    ace = 0.0
    inf = INFINITY
    aggregates = np.empty(num_agg)
    raw_acf = <double *> malloc(nlags * sizeof(double))
    error_values = <double *> malloc(n * sizeof(double))
    acf_errors = <Heap *> malloc(sizeof(Heap))
    acf_model = <AcfAgg *> malloc(sizeof(AcfAgg))
    no_removed_indices = np.ones(y.shape[0], dtype=bool)

    inc_acf_agg.initialize(acf_model, nlags)  # initialize the aggregates
    inc_acf_agg.fit_mean(acf_model, y, aggregates, kappa)  # extract the aggregates
    inc_acf.get_acf(acf_model, raw_acf)  # get raw acf
    math_utils.compute_acf_agg_mean_fall(acf_model, y, aggregates, raw_acf, error_values, n, kappa)
    heap.initialize_from_pointer(acf_errors, map_node_to_heap, error_values, n) # Initialize the heap
    while acf_errors.values[0].value < inf:
        min_node = heap.pop(acf_errors, map_node_to_heap) # TODO: make it a reference
        if min_node.value != 0:
            start = min_node.left
            end = min_node.right
            if start + 2 < end:
                inc_acf_agg.interpolate_update_mean(acf_model, y, aggregates, start, end, kappa)
            else:
                x_a = (y[end]-y[start]) / (end-start) + y[start]
                inc_acf_agg.update_mean(acf_model, y, aggregates, x_a, start + 1, kappa)

            # Putting together
            # inc_acf.get_acf(acf_agg, c_acf)
            # ace = math_utils.mae(raw_acf, c_acf, nlags)
            ace = 0.0
            n = acf_model.n
            for lag in range(acf_model.nlags):
                n -= 1
                c_acf = (n * acf_model.sxy[lag] - acf_model.xs[lag] * acf_model.ys[lag]) / \
                              sqrt((n * acf_model.xss[lag] - acf_model.xs[lag] * acf_model.xs[lag]) *
                                   (n * acf_model.yss[lag] - acf_model.ys[lag] * acf_model.ys[lag]))
                ace += fabs(raw_acf[lag] - c_acf)# *(raw_acf[lag] - c_acf)

            ace /= acf_model.nlags

        if ace >= acf_threshold:
            break

        no_removed_indices[min_node.ts] = False
        heap.update_left_right(acf_errors, map_node_to_heap, min_node.left, min_node.right)
        look_ahead_reheap_mean(acf_model, acf_errors, map_node_to_heap, y, aggregates, raw_acf, min_node, hops, kappa)

    heap.release_memory(acf_errors)
    inc_acf.release_memory(acf_model)
    free(error_values)
    free(raw_acf)

    return no_removed_indices


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.uint8_t, ndim=1] simplify_single_thread_sum(double[:] y, int hops,
                                                                int nlags, int kappa, double acf_threshold):
    cdef int start, end, lag, n, num_agg, i
    cdef double ace, x_a, right_area, left_area, c_acf, inf
    cdef double * raw_acf
    cdef double * error_values
    cdef double[:] aggregates
    cdef Heap * acf_errors
    cdef AcfAgg * acf_model
    cdef unordered_map[int, int] map_node_to_heap
    cdef Node min_node, left, right
    cdef np.ndarray[np.uint8_t, ndim=1] no_removed_indices
    cdef short int overflow

    n = y.shape[0]
    num_agg = n // kappa
    overflow = 0 if n % kappa == 0 else 1

    i = 0
    ace = 0.0
    inf = INFINITY
    aggregates = np.empty(num_agg)
    raw_acf = <double *> malloc(nlags * sizeof(double))
    error_values = <double *> malloc(n * sizeof(double))
    acf_errors = <Heap *> malloc(sizeof(Heap))
    acf_model = <AcfAgg *> malloc(sizeof(AcfAgg))
    no_removed_indices = np.ones(y.shape[0], dtype=bool)

    inc_acf_agg.initialize(acf_model, nlags)  # initialize the aggregates
    inc_acf_agg.fit_sum(acf_model, y, aggregates, kappa)  # extract the aggregates
    inc_acf.get_acf(acf_model, raw_acf)  # get raw acf
    math_utils.compute_acf_agg_sum_fall(acf_model, y, aggregates, raw_acf, error_values, n, kappa)
    heap.initialize_from_pointer(acf_errors, map_node_to_heap, error_values, n) # Initialize the heap
    while acf_errors.values[0].value < inf:
        min_node = heap.pop(acf_errors, map_node_to_heap) # TODO: make it a reference
        if min_node.value != 0:
            start = min_node.left
            end = min_node.right
            if start + 2 < end:
                inc_acf_agg.interpolate_update_sum(acf_model, y, aggregates, start, end, kappa)
            else:
                x_a = (y[end]-y[start]) / (end-start) + y[start]
                inc_acf_agg.update_sum(acf_model, y, aggregates, x_a, start + 1, kappa)

            ace = 0.0
            n = acf_model.n
            for lag in range(acf_model.nlags):
                n -= 1
                c_acf = (n * acf_model.sxy[lag] - acf_model.xs[lag] * acf_model.ys[lag]) / \
                              sqrt((n * acf_model.xss[lag] - acf_model.xs[lag] * acf_model.xs[lag]) *
                                   (n * acf_model.yss[lag] - acf_model.ys[lag] * acf_model.ys[lag]))
                ace += fabs(raw_acf[lag] - c_acf)# *(raw_acf[lag] - c_acf)

            ace /= acf_model.nlags

        if ace >= acf_threshold:
            break

        no_removed_indices[min_node.ts] = False
        heap.update_left_right(acf_errors, map_node_to_heap, min_node.left, min_node.right)
        look_ahead_reheap_sum(acf_model, acf_errors, map_node_to_heap, y, aggregates, raw_acf, min_node, hops, kappa)

    heap.release_memory(acf_errors)
    inc_acf.release_memory(acf_model)
    free(error_values)
    free(raw_acf)

    return no_removed_indices


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.uint8_t, ndim=1] parallel_simplify_by_blocking_mean(double[:] y, int hops,
                                                            int nlags, int kappa, double acf_threshold, int num_threads):
    cdef:
        int start, end, lag, n = y.shape[0], num_agg = n//kappa, i=0
        double ace = 0.0, x_a, right_area, left_area, c_acf, inf = INFINITY
        double * raw_acf = <double *> malloc(nlags * sizeof(double))
        double * error_values = <double *> malloc(n * sizeof(double))
        double[:] aggregates = np.empty(num_agg)
        Heap * acf_errors = <Heap *> malloc(sizeof(Heap))
        AcfAgg * acf_model = <AcfAgg *> malloc(sizeof(AcfAgg))
        unordered_map[int, int] map_node_to_heap
        Node min_node, left, right
        np.ndarray[np.uint8_t, ndim=1] no_removed_indices = np.ones(y.shape[0], dtype=bool)


    inc_acf_agg.initialize(acf_model, nlags)  # initialize the aggregates
    inc_acf_agg.fit_mean(acf_model, y, aggregates, kappa)  # extract the aggregates
    inc_acf.get_acf(acf_model, raw_acf)  # get raw acf
    math_utils.compute_acf_agg_mean_fall(acf_model, y, aggregates, raw_acf, error_values, n, kappa) # computing the areas for all triangles
    heap.initialize_from_pointer(acf_errors, map_node_to_heap, error_values, n) # Initialize the heap
    # logs = list()


    while acf_errors.values[0].value < inf:
        min_node = heap.pop(acf_errors, map_node_to_heap) # TODO: make it a reference
        # logs.append((min_node.value, min_node.ts))
        if min_node.value != 0:
            start = min_node.left
            end = min_node.right
            if start + 2 < end:
                inc_acf_agg.interpolate_update_mean(acf_model, y, aggregates, start, end, kappa)
            else:
                x_a = (y[end]-y[start]) / (end-start) + y[start]
                inc_acf_agg.update_mean(acf_model, y, aggregates, x_a, start + 1, kappa)

            # Putting together
            # inc_acf.get_acf(acf_agg, c_acf)
            # ace = math_utils.mae(raw_acf, c_acf, nlags)
            ace = 0.0
            n = acf_model.n
            for lag in range(acf_model.nlags):
                n -= 1
                c_acf = (n * acf_model.sxy[lag] - acf_model.xs[lag] * acf_model.ys[lag]) / \
                              sqrt((n * acf_model.xss[lag] - acf_model.xs[lag] * acf_model.xs[lag]) *
                                   (n * acf_model.yss[lag] - acf_model.ys[lag] * acf_model.ys[lag]))
                ace += fabs(raw_acf[lag] - c_acf)

            ace /= acf_model.nlags

        if ace >= acf_threshold:
            break

        no_removed_indices[min_node.ts] = False
        heap.update_left_right(acf_errors, map_node_to_heap, min_node.left, min_node.right)
        parallel_look_ahead_reheap_mean(acf_model, acf_errors, map_node_to_heap, y, aggregates, raw_acf, min_node, hops, kappa, num_threads)

    # np.save('./logs/removing_order_cython', logs)
    heap.release_memory(acf_errors)
    inc_acf.release_memory(acf_model)
    free(raw_acf)
    free(error_values)

    return no_removed_indices

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.uint8_t, ndim=1] parallel_simplify_by_blocking_sum(double[:] y, int hops,
                                                                       int nlags, int kappa, double acf_threshold, int num_threads):
    cdef:
        int start, end, lag, n = y.shape[0], num_agg = n//kappa, i=0
        double ace = 0.0, x_a, right_area, left_area, c_acf, inf = INFINITY
        double * raw_acf = <double *> malloc(nlags * sizeof(double))
        double * error_values = <double *> malloc(n * sizeof(double))
        double[:] aggregates = np.empty(num_agg)
        Heap * acf_errors = <Heap *> malloc(sizeof(Heap))
        AcfAgg * acf_model = <AcfAgg *> malloc(sizeof(AcfAgg))
        unordered_map[int, int] map_node_to_heap
        Node min_node, left, right
        np.ndarray[np.uint8_t, ndim=1] no_removed_indices = np.ones(y.shape[0], dtype=bool)


    inc_acf_agg.initialize(acf_model, nlags)  # initialize the aggregates
    inc_acf_agg.fit_sum(acf_model, y, aggregates, kappa)  # extract the aggregates
    inc_acf.get_acf(acf_model, raw_acf)  # get raw acf
    math_utils.compute_acf_agg_sum_fall(acf_model, y, aggregates, raw_acf, error_values, n, kappa) # computing the areas for all triangles
    heap.initialize_from_pointer(acf_errors, map_node_to_heap, error_values, n) # Initialize the heap
    # logs = list()


    while acf_errors.values[0].value < inf:
        min_node = heap.pop(acf_errors, map_node_to_heap) # TODO: make it a reference
        # logs.append((min_node.value, min_node.ts))
        if min_node.value != 0:
            start = min_node.left
            end = min_node.right
            if start + 2 < end:
                inc_acf_agg.interpolate_update_sum(acf_model, y, aggregates, start, end, kappa)
            else:
                x_a = (y[end]-y[start]) / (end-start) + y[start]
                inc_acf_agg.update_sum(acf_model, y, aggregates, x_a, start + 1, kappa)

            ace = 0.0
            n = acf_model.n
            for lag in range(acf_model.nlags):
                n -= 1
                c_acf = (n * acf_model.sxy[lag] - acf_model.xs[lag] * acf_model.ys[lag]) / \
                              sqrt((n * acf_model.xss[lag] - acf_model.xs[lag] * acf_model.xs[lag]) *
                                   (n * acf_model.yss[lag] - acf_model.ys[lag] * acf_model.ys[lag]))
                ace += fabs(raw_acf[lag] - c_acf)

            ace /= acf_model.nlags

        if ace >= acf_threshold:
            break

        no_removed_indices[min_node.ts] = False
        heap.update_left_right(acf_errors, map_node_to_heap, min_node.left, min_node.right)
        parallel_look_ahead_reheap_sum(acf_model, acf_errors, map_node_to_heap, y, aggregates, raw_acf, min_node, hops, kappa, num_threads)

    # np.save('./logs/removing_order_cython', logs)
    heap.release_memory(acf_errors)
    inc_acf.release_memory(acf_model)
    free(raw_acf)
    free(error_values)

    return no_removed_indices

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.uint8_t, ndim=1] parallel_simplify_by_blocking_sum_mse(double[:] y, int hops,
                                                                           int nlags, int kappa,
                                                                           double acf_threshold, int num_threads):
    cdef:
        int start, end, lag, n = y.shape[0], num_agg = n//kappa, i=0
        double ace = 0.0, x_a, right_area, left_area, c_acf, inf = INFINITY
        double * raw_acf = <double *> malloc(nlags * sizeof(double))
        double * error_values = <double *> malloc(n * sizeof(double))
        double[:] aggregates = np.empty(num_agg)
        Heap * acf_errors = <Heap *> malloc(sizeof(Heap))
        AcfAgg * acf_model = <AcfAgg *> malloc(sizeof(AcfAgg))
        unordered_map[int, int] map_node_to_heap
        Node min_node, left, right
        np.ndarray[np.uint8_t, ndim=1] no_removed_indices = np.ones(y.shape[0], dtype=bool)


    inc_acf_agg.initialize(acf_model, nlags)  # initialize the aggregates
    inc_acf_agg.fit_sum(acf_model, y, aggregates, kappa)  # extract the aggregates
    inc_acf.get_acf(acf_model, raw_acf)  # get raw acf
    math_utils.compute_acf_agg_sum_fall(acf_model, y, aggregates, raw_acf, error_values, n, kappa) # computing the areas for all triangles
    heap.initialize_from_pointer(acf_errors, map_node_to_heap, error_values, n) # Initialize the heap
    # logs = list()

    while acf_errors.values[0].value < inf:
        min_node = heap.pop(acf_errors, map_node_to_heap) # TODO: make it a reference
        # logs.append((min_node.value, min_node.ts))
        if min_node.value != 0:
            start = min_node.left
            end = min_node.right
            if start + 2 < end:
                inc_acf_agg.interpolate_update_sum(acf_model, y, aggregates, start, end, kappa)
            else:
                x_a = (y[end]-y[start]) / (end-start) + y[start]
                inc_acf_agg.update_sum(acf_model, y, aggregates, x_a, start + 1, kappa)

            ace = 0.0
            n = acf_model.n
            for lag in range(acf_model.nlags):
                n -= 1
                c_acf = (n * acf_model.sxy[lag] - acf_model.xs[lag] * acf_model.ys[lag]) / \
                              sqrt((n * acf_model.xss[lag] - acf_model.xs[lag] * acf_model.xs[lag]) *
                                   (n * acf_model.yss[lag] - acf_model.ys[lag] * acf_model.ys[lag]))
                ace += (raw_acf[lag] - c_acf)*(raw_acf[lag] - c_acf)

            ace /= acf_model.nlags

        if ace >= acf_threshold:
            break

        no_removed_indices[min_node.ts] = False
        heap.update_left_right(acf_errors, map_node_to_heap, min_node.left, min_node.right)
        parallel_look_ahead_reheap_sum_mse(acf_model, acf_errors, map_node_to_heap, y, aggregates, raw_acf, min_node, hops, kappa, num_threads)

    # np.save('./logs/removing_order_cython', logs)
    heap.release_memory(acf_errors)
    inc_acf.release_memory(acf_model)
    free(raw_acf)
    free(error_values)

    return no_removed_indices

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.uint8_t, ndim=1] parallel_simplify_by_blocking_max(double[:] y, int hops,
                                                                       int nlags, int kappa, double acf_threshold, int num_threads):
    cdef:
        int start, end, lag, n = y.shape[0], num_agg = n//kappa, i=0
        double ace = 0.0, x_a, right_area, left_area, c_acf, inf = INFINITY
        double * raw_acf = <double *> malloc(nlags * sizeof(double))
        double * error_values = <double *> malloc(n * sizeof(double))
        double[:] aggregates = np.empty(num_agg)
        Heap * acf_errors = <Heap *> malloc(sizeof(Heap))
        AcfAgg * acf_model = <AcfAgg *> malloc(sizeof(AcfAgg))
        unordered_map[int, int] map_node_to_heap
        Node min_node, left, right
        np.ndarray[np.uint8_t, ndim=1] no_removed_indices = np.ones(y.shape[0], dtype=bool)


    inc_acf_agg.initialize(acf_model, nlags)  # initialize the aggregates
    inc_acf_agg.fit_mean(acf_model, y, aggregates, kappa)  # extract the aggregates
    inc_acf.get_acf(acf_model, raw_acf)  # get raw acf
    math_utils.compute_acf_agg_mean_fall(acf_model, y, aggregates, raw_acf, error_values, n, kappa) # computing the areas for all triangles
    heap.initialize_from_pointer(acf_errors, map_node_to_heap, error_values, n) # Initialize the heap
    # logs = list()


    while acf_errors.values[0].value < inf:
        min_node = heap.pop(acf_errors, map_node_to_heap) # TODO: make it a reference
        # logs.append((min_node.value, min_node.ts))
        if min_node.value != 0:
            start = min_node.left
            end = min_node.right
            if start + 2 < end:
                inc_acf_agg.interpolate_update_mean(acf_model, y, aggregates, start, end, kappa)
            else:
                x_a = (y[end]-y[start]) / (end-start) + y[start]
                inc_acf_agg.update_max(acf_model, y, aggregates, x_a, start + 1, kappa)

            # Putting together
            # inc_acf.get_acf(acf_agg, c_acf)
            # ace = math_utils.mae(raw_acf, c_acf, nlags)
            ace = 0.0
            n = acf_model.n
            for lag in range(acf_model.nlags):
                n -= 1
                c_acf = (n * acf_model.sxy[lag] - acf_model.xs[lag] * acf_model.ys[lag]) / \
                              sqrt((n * acf_model.xss[lag] - acf_model.xs[lag] * acf_model.xs[lag]) *
                                   (n * acf_model.yss[lag] - acf_model.ys[lag] * acf_model.ys[lag]))
                ace += fabs(raw_acf[lag] - c_acf)

            ace /= acf_model.nlags

        if ace >= acf_threshold:
            break

        no_removed_indices[min_node.ts] = False
        heap.update_left_right(acf_errors, map_node_to_heap, min_node.left, min_node.right)
        parallel_look_ahead_reheap_mean(acf_model, acf_errors, map_node_to_heap, y, aggregates, raw_acf, min_node, hops, kappa, num_threads)

    # np.save('./logs/removing_order_cython', logs)
    heap.release_memory(acf_errors)
    inc_acf.release_memory(acf_model)
    free(raw_acf)
    free(error_values)

    return no_removed_indices
