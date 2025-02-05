import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from compressors cimport math_utils
from compressors.inc_acf cimport AcfPtr
from numpy.math cimport INFINITY
cimport cython


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void initialize(AcfPtr model, const int& nlags):
    model.sxy = <double*> malloc(nlags * sizeof(double))
    model.xs = <double*> malloc(nlags * sizeof(double))
    model.ys = <double*> malloc(nlags * sizeof(double))
    model.xss = <double*> malloc(nlags * sizeof(double))
    model.yss = <double*> malloc(nlags * sizeof(double))
    model.nlags = nlags


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fit_mean(AcfPtr model, double[:] x, double [:] aggregates, int kappa):
    cdef int n = x.shape[0], an=aggregates.shape[0], lag, i, j
    cdef double[:] cum_sum = np.empty_like(aggregates)
    cdef double[:] power_cum_sum = np.empty_like(aggregates)

    j = 0
    i = 0
    while i < n-kappa+1:
        aggregates[j] = math_utils.csum(x[i:i+kappa], kappa)/kappa # aggregate function
        j += 1
        i += kappa

    if j == an-1:
        aggregates[j] = math_utils.csum(x[i:n], n-i)/(n-i)

    math_utils.cumsum_cumsum(aggregates, cum_sum, power_cum_sum)
    model.n = an

    for lag in range(model.nlags):
        model.xs[lag] = cum_sum[an - lag - 2]
        model.ys[lag] = cum_sum[an - 1] - cum_sum[lag]
        model.xss[lag] = power_cum_sum[an - lag - 2]
        model.yss[lag] = power_cum_sum[an - 1] - power_cum_sum[lag]
        model.sxy[lag] = math_utils.dot_product(aggregates[:an - lag - 1], aggregates[lag + 1:])


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fit_sum(AcfPtr model, double[:] x, double [:] aggregates, int kappa):
    cdef int n = x.shape[0], an=aggregates.shape[0], lag, i, j
    cdef double[:] cum_sum = np.empty_like(aggregates)
    cdef double[:] power_cum_sum = np.empty_like(aggregates)

    j = 0
    i = 0
    while i < n-kappa+1:
        aggregates[j] = math_utils.csum(x[i:i+kappa], kappa) # aggregate function
        j += 1
        i += kappa

    if j == an-1:
        aggregates[j] = math_utils.csum(x[i:n], n-i)/(n-i)

    math_utils.cumsum_cumsum(aggregates, cum_sum, power_cum_sum)
    model.n = an

    for lag in range(model.nlags):
        model.xs[lag] = cum_sum[an - lag - 2]
        model.ys[lag] = cum_sum[an - 1] - cum_sum[lag]
        model.xss[lag] = power_cum_sum[an - lag - 2]
        model.yss[lag] = power_cum_sum[an - 1] - power_cum_sum[lag]
        model.sxy[lag] = math_utils.dot_product(aggregates[:an - lag - 1], aggregates[lag + 1:])


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fit_max(AcfPtr model, double[:] x, double [:] aggregates, int kappa):
    cdef int n = x.shape[0], an=aggregates.shape[0], lag, i, j, k
    cdef double max_value
    cdef double[:] cum_sum = np.empty_like(aggregates)
    cdef double[:] power_cum_sum = np.empty_like(aggregates)

    j = 0
    i = 0

    while i < n-kappa+1:
        max_value = -INFINITY

        for k in range(i, i+kappa):
            max_value = max(x[k], max_value)

        aggregates[j] = max_value
        j += 1
        i += kappa

    if j == an-1:
        aggregates[j] = math_utils.csum(x[i:n], n-i)/(n-i)

    math_utils.cumsum_cumsum(aggregates, cum_sum, power_cum_sum)
    model.n = an

    for lag in range(model.nlags):
        model.xs[lag] = cum_sum[an - lag - 2]
        model.ys[lag] = cum_sum[an - 1] - cum_sum[lag]
        model.xss[lag] = power_cum_sum[an - lag - 2]
        model.yss[lag] = power_cum_sum[an - 1] - power_cum_sum[lag]
        model.sxy[lag] = math_utils.dot_product(aggregates[:an - lag - 1], aggregates[lag + 1:])

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void update_mean(AcfPtr model, double[:] x, double[:] aggregates,
                 const double &x_a, const int &index, const int &kappa):
    cdef int agg_index = index//kappa
    cdef double delta, delta_ss
    delta = (x_a - x[index])/kappa # aggregate function
    delta_ss = delta * (2 * aggregates[agg_index] + delta)
    if delta != 0:
        if agg_index <= model.nlags or agg_index >= model.n-model.nlags:
            update_inside_lags(model, aggregates, delta, delta_ss, agg_index)
        else:
            update_outside_lags(model, aggregates, delta, delta_ss, agg_index)

        x[index] = x_a
        aggregates[agg_index] += delta


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void update_sum(AcfPtr model, double[:] x, double[:] aggregates,
                     const double &x_a, const int &index, const int &kappa):
    cdef int agg_index = index//kappa
    cdef double delta, delta_ss
    delta = (x_a - x[index]) # aggregate function
    delta_ss = delta * (2 * aggregates[agg_index] + delta)
    if delta != 0:
        if agg_index <= model.nlags or agg_index >= model.n-model.nlags:
            update_inside_lags(model, aggregates, delta, delta_ss, agg_index)
        else:
            update_outside_lags(model, aggregates, delta, delta_ss, agg_index)

        x[index] = x_a
        aggregates[agg_index] += delta

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void update_max(AcfPtr model, double[:] x, double[:] aggregates,
                     double &x_a, const int &index, const int &kappa):
    cdef int agg_index = index//kappa
    cdef double delta, delta_ss
    delta = max(x_a, aggregates[agg_index]) - aggregates[agg_index] # aggregate function
    delta_ss = delta * (2 * aggregates[agg_index] + delta)
    if delta != 0:
        if agg_index <= model.nlags or agg_index >= model.n-model.nlags:
            update_inside_lags(model, aggregates, delta, delta_ss, agg_index)
        else:
            update_outside_lags(model, aggregates, delta, delta_ss, agg_index)

        x[index] = x_a
        aggregates[agg_index] += delta

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void update_inside_lags(AcfPtr model, double[:] aggregates,
                                    const double &delta,
                                    const double &delta_ss,
                                    const int &index_a):
    cdef int lag, n = model.n - 1
    for lag in range(model.nlags):
        if index_a >= lag+1:
            model.ys[lag] += delta
            model.yss[lag] += delta_ss
            model.sxy[lag] += delta*aggregates[index_a - lag - 1]
        if index_a < n-lag:
            model.xs[lag] += delta
            model.xss[lag] += delta_ss
            model.sxy[lag] += delta*aggregates[index_a + lag + 1]



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void update_outside_lags(AcfPtr model, double[:] aggregates,
                                     const double &delta,
                                     const double &delta_ss,
                                     const int &index_a):
    cdef int lag
    for lag in range(model.nlags):
        model.ys[lag] += delta
        model.yss[lag] += delta_ss
        model.xs[lag] += delta
        model.xss[lag] += delta_ss
        model.sxy[lag] += delta * (aggregates[index_a + lag + 1] + aggregates[index_a-lag-1])


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void interpolate_update_mean(AcfPtr model, double[:] x, double[:] aggregates, int &start, int &end, int kappa):
    cdef int start_index_a = (start+1) // kappa
    cdef int end_index_a = (end-1) // kappa

    if start_index_a <= model.nlags or end_index_a >= model.n-model.nlags:
        interpolate_update_inside_lags_mean(model, x, aggregates, start, end, start_index_a, end_index_a, kappa)
    else:
        interpolate_update_outside_lags_mean(model, x, aggregates, start, end, start_index_a, end_index_a, kappa)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void interpolate_update_sum(AcfPtr model, double[:] x, double[:] aggregates, int &start, int &end, int kappa):
    cdef int start_index_a = (start+1) // kappa
    cdef int end_index_a = (end-1) // kappa

    if start_index_a <= model.nlags or end_index_a >= model.n-model.nlags:
        interpolate_update_inside_lags_sum(model, x, aggregates, start, end, start_index_a, end_index_a, kappa)
    else:
        interpolate_update_outside_lags_sum(model, x, aggregates, start, end, start_index_a, end_index_a, kappa)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void interpolate_update_max(AcfPtr model, double[:] x, double[:] aggregates, int &start, int &end, int kappa):
    cdef int start_index_a = (start+1) // kappa
    cdef int end_index_a = (end-1) // kappa

    if start_index_a <= model.nlags or end_index_a >= model.n-model.nlags:
        interpolate_update_inside_lags_max(model, x, aggregates, start, end, start_index_a, end_index_a, kappa)
    else:
        interpolate_update_outside_lags_max(model, x, aggregates, start, end, start_index_a, end_index_a, kappa)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void interpolate_update_outside_lags_mean(AcfPtr model, double[:] x, double[:] aggregates,
                                               int &start, int &end, int &start_index_a,
                                               int &end_index_a, int &kappa):
    cdef double delta, delta_ss, slope
    cdef int i, index, lag, num_agg_deltas, agg_index, j
    cdef double * sum_agg_deltas
    cdef double x_a
    num_agg_deltas = end_index_a-start_index_a+1
    slope = (x[end] - x[start]) / (end - start)
    sum_agg_deltas = <double*> malloc(num_agg_deltas*sizeof(double))

    for i in range(num_agg_deltas):
        sum_agg_deltas[i] = 0.0

    for index in range(start+1, end):
        x_a = slope * (index - start) + x[start]
        agg_index = index//kappa
        i = agg_index - start_index_a
        sum_agg_deltas[i] += (x_a - x[index])
        x[index] = x_a

    i = 0
    for agg_index in range(start_index_a, end_index_a+1):
        sum_agg_deltas[i] /= kappa # aggregate function
        delta = sum_agg_deltas[i]
        delta_ss = delta * (delta + 2 * aggregates[agg_index])
        for lag in range(model.nlags):
            model.xs[lag] += delta
            model.ys[lag] += delta
            model.yss[lag] += delta_ss
            model.xss[lag] += delta_ss
            model.sxy[lag] += delta * (aggregates[agg_index - lag - 1] + aggregates[agg_index + lag + 1])
        i += 1

    num_lags = num_agg_deltas if num_agg_deltas < model.nlags else model.nlags

    for lag in range(num_lags):
        for i in range(num_agg_deltas - lag - 1):
            model.sxy[lag] += sum_agg_deltas[i] * sum_agg_deltas[i + lag + 1]

    i = 0
    for index in range(start_index_a, end_index_a+1):
        aggregates[index] += sum_agg_deltas[i]
        i += 1

    free(sum_agg_deltas)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void interpolate_update_outside_lags_sum(AcfPtr model, double[:] x, double[:] aggregates,
                                          int &start, int &end, int &start_index_a,
                                          int &end_index_a, int &kappa):
    cdef double delta, delta_ss, slope
    cdef int i, index, lag, num_agg_deltas, agg_index, j
    cdef double *sum_agg_deltas
    cdef double x_a
    num_agg_deltas = end_index_a-start_index_a+1
    slope = (x[end] - x[start]) / (end - start)
    sum_agg_deltas = <double*> malloc(num_agg_deltas*sizeof(double))

    for i in range(num_agg_deltas):
        sum_agg_deltas[i] = 0.0

    for index in range(start+1, end):
        x_a = slope * (index - start) + x[start]
        agg_index = index//kappa
        i = agg_index - start_index_a
        sum_agg_deltas[i] += (x_a - x[index])
        x[index] = x_a

    i = 0
    for agg_index in range(start_index_a, end_index_a+1):
        # sum_agg_deltas[i] /= kappa # aggregate function
        delta = sum_agg_deltas[i]
        delta_ss = delta * (delta + 2 * aggregates[agg_index])
        for lag in range(model.nlags):
            model.xs[lag] += delta
            model.ys[lag] += delta
            model.yss[lag] += delta_ss
            model.xss[lag] += delta_ss
            model.sxy[lag] += delta * (aggregates[agg_index - lag - 1] + aggregates[agg_index + lag + 1])
        i += 1

    num_lags = num_agg_deltas if num_agg_deltas < model.nlags else model.nlags

    for lag in range(num_lags):
        for i in range(num_agg_deltas - lag - 1):
            model.sxy[lag] += sum_agg_deltas[i] * sum_agg_deltas[i + lag + 1]

    i = 0
    for index in range(start_index_a, end_index_a+1):
        aggregates[index] += sum_agg_deltas[i]
        i += 1

    free(sum_agg_deltas)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void interpolate_update_outside_lags_max(AcfPtr model, double[:] x, double[:] aggregates,
                                              int &start, int &end, int &start_index_a,
                                              int &end_index_a, int &kappa):
    cdef double delta, delta_ss, slope
    cdef int i, index, lag, num_agg_deltas, agg_index, j
    cdef double * deltas
    cdef double x_a
    num_agg_deltas = end_index_a-start_index_a+1
    slope = (x[end] - x[start]) / (end - start)
    deltas = <double*> malloc(num_agg_deltas*sizeof(double))

    for i in range(num_agg_deltas):
        deltas[i] = -INFINITY

    for index in range(start+1, end):
        x_a = slope * (index - start) + x[start]
        agg_index = index//kappa
        i = agg_index - start_index_a
        deltas[i] = max(x_a, deltas[i])
        x[index] = x_a

    i = 0
    for agg_index in range(start_index_a, end_index_a+1):
        deltas[i] = deltas[i] - aggregates[agg_index] # aggregate function
        delta = deltas[i]
        delta_ss = delta * (delta + 2 * aggregates[agg_index])
        if delta != 0:
            for lag in range(model.nlags):
                model.xs[lag] += delta
                model.ys[lag] += delta
                model.yss[lag] += delta_ss
                model.xss[lag] += delta_ss
                model.sxy[lag] += delta * (aggregates[agg_index - lag - 1] + aggregates[agg_index + lag + 1])
        i += 1

    num_lags = num_agg_deltas if num_agg_deltas < model.nlags else model.nlags

    for lag in range(num_lags):
        for i in range(num_agg_deltas - lag - 1):
            model.sxy[lag] += deltas[i] * deltas[i + lag + 1]

    i = 0
    for index in range(start_index_a, end_index_a+1):
        aggregates[index] += deltas[i]
        i += 1

    free(deltas)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void interpolate_update_inside_lags_mean(AcfPtr model, double[:] x, double[:] aggregates,
                                          int &start, int &end, int &start_index_a,
                                          int &end_index_a, int &kappa):
    cdef double delta, delta_ss, slope
    cdef int i, index, lag, num_agg_deltas, agg_index, j, n = model.n - 1
    cdef double *sum_agg_deltas
    cdef double x_a
    num_agg_deltas = end_index_a-start_index_a+1
    slope = (x[end] - x[start]) / (end - start)
    sum_agg_deltas = <double*> malloc(num_agg_deltas*sizeof(double))

    for i in range(num_agg_deltas):
        sum_agg_deltas[i] = 0.0

    for index in range(start+1, end):
        x_a = slope * (index - start) + x[start]
        agg_index = index//kappa
        i = agg_index - start_index_a
        sum_agg_deltas[i] += (x_a - x[index])
        x[index] = x_a

    i = 0
    for agg_index in range(start_index_a, end_index_a+1):
        sum_agg_deltas[i] /= kappa # aggregate function
        delta = sum_agg_deltas[i]
        delta_ss = delta * (delta + 2 * aggregates[agg_index])
        for lag in range(model.nlags):
            if agg_index > lag:
                model.ys[lag] += delta
                model.yss[lag] += delta_ss
                model.sxy[lag] += delta * aggregates[agg_index - lag - 1]
            if agg_index < n - lag:
                model.xs[lag] += delta
                model.xss[lag] += delta_ss
                model.sxy[lag] += delta * aggregates[agg_index + lag + 1]
        i += 1

    num_lags = num_agg_deltas if num_agg_deltas < model.nlags else model.nlags

    for lag in range(num_lags):
        for i in range(num_agg_deltas - lag - 1):
            model.sxy[lag] += sum_agg_deltas[i] * sum_agg_deltas[i + lag + 1]

    i = 0
    for index in range(start_index_a, end_index_a+1):
        aggregates[index] += sum_agg_deltas[i]
        i += 1

    free(sum_agg_deltas)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void interpolate_update_inside_lags_sum(AcfPtr model, double[:] x, double[:] aggregates,
                                             int &start, int &end, int &start_index_a,
                                             int &end_index_a, int &kappa):
    cdef double delta, delta_ss, slope
    cdef int i, index, lag, num_agg_deltas, agg_index, j, n = model.n - 1
    cdef double *sum_agg_deltas
    cdef double x_a
    num_agg_deltas = end_index_a-start_index_a+1
    slope = (x[end] - x[start]) / (end - start)
    sum_agg_deltas = <double*> malloc(num_agg_deltas*sizeof(double))

    for i in range(num_agg_deltas):
        sum_agg_deltas[i] = 0.0

    for index in range(start+1, end):
        x_a = slope * (index - start) + x[start]
        agg_index = index//kappa
        i = agg_index - start_index_a
        sum_agg_deltas[i] += (x_a - x[index])
        x[index] = x_a

    i = 0
    for agg_index in range(start_index_a, end_index_a+1):
        # sum_agg_deltas[i] /= kappa # aggregate function
        delta = sum_agg_deltas[i]
        delta_ss = delta * (delta + 2 * aggregates[agg_index])
        for lag in range(model.nlags):
            if agg_index > lag:
                model.ys[lag] += delta
                model.yss[lag] += delta_ss
                model.sxy[lag] += delta * aggregates[agg_index - lag - 1]
            if agg_index < n - lag:
                model.xs[lag] += delta
                model.xss[lag] += delta_ss
                model.sxy[lag] += delta * aggregates[agg_index + lag + 1]
        i += 1

    num_lags = num_agg_deltas if num_agg_deltas < model.nlags else model.nlags

    for lag in range(num_lags):
        for i in range(num_agg_deltas - lag - 1):
            model.sxy[lag] += sum_agg_deltas[i] * sum_agg_deltas[i + lag + 1]

    i = 0
    for index in range(start_index_a, end_index_a+1):
        aggregates[index] += sum_agg_deltas[i]
        i += 1

    free(sum_agg_deltas)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void interpolate_update_inside_lags_max(AcfPtr model, double[:] x, double[:] aggregates,
                                             int &start, int &end, int &start_index_a,
                                             int &end_index_a, int &kappa):
    cdef double delta, delta_ss, slope
    cdef int i, index, lag, num_agg_deltas, agg_index, j, n = model.n - 1
    cdef double * deltas
    cdef double x_a
    num_agg_deltas = end_index_a-start_index_a+1
    slope = (x[end] - x[start]) / (end - start)
    deltas = <double*> malloc(num_agg_deltas*sizeof(double))

    for i in range(num_agg_deltas):
        deltas[i] = -INFINITY

    for index in range(start+1, end):
        x_a = slope * (index - start) + x[start]
        agg_index = index//kappa
        i = agg_index - start_index_a
        deltas[i] = max(x_a, deltas[i]) # aggregate function
        x[index] = x_a

    i = 0
    for agg_index in range(start_index_a, end_index_a+1):
        deltas[i] = deltas[i] - aggregates[agg_index] # aggregate function
        delta = deltas[i]
        delta_ss = delta * (delta + 2 * aggregates[agg_index])
        if delta != 0:
            for lag in range(model.nlags):
                if agg_index > lag:
                    model.ys[lag] += delta
                    model.yss[lag] += delta_ss
                    model.sxy[lag] += delta * aggregates[agg_index - lag - 1]
                if agg_index < n - lag:
                    model.xs[lag] += delta
                    model.xss[lag] += delta_ss
                    model.sxy[lag] += delta * aggregates[agg_index + lag + 1]
        i += 1

    num_lags = num_agg_deltas if num_agg_deltas < model.nlags else model.nlags

    for lag in range(num_lags):
        for i in range(num_agg_deltas - lag - 1):
            model.sxy[lag] += deltas[i] * deltas[i + lag + 1]

    i = 0
    for index in range(start_index_a, end_index_a+1):
        aggregates[index] += deltas[i]
        i += 1

    free(deltas)
