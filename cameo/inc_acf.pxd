cdef struct AcfAgg:
    int nlags, n
    double *sxy
    double *xs
    double *ys
    double *xss
    double *yss

ctypedef AcfAgg* AcfPtr
cdef void initialize(AcfPtr model, const int& nlags)
cdef void fit(AcfPtr model, double[:] x)
cdef void get_acf(AcfPtr model, double* result)
cdef void update(AcfPtr model, double[:] x, const double &x_a, const int &index)
cdef double look_ahead_impact(AcfPtr model, double[:] x, double *raw_acf, const double &x_a, const int &index) nogil
cdef inline void update_inside_lags(AcfPtr model, double[:] x,
                                     const double &delta,
                                     const double& delta_ss,
                                     const int& index)
cdef inline void update_outside_lags(AcfPtr model, double[:] x,
                                     const double &delta,
                                     const double& delta_ss,
                                     const int& index)
cdef void interpolate_update(AcfPtr model, double[:] x, const int &start, int &end)
cdef double look_ahead_interpolated_impact(AcfPtr model, double[:] x, double *raw_acf, const int &start, int &end) nogil
cdef void interpolate_update_outside_lags(AcfPtr model, double[:] x, const int &start, int &end)
cdef void interpolate_update_inside_lags(AcfPtr model, double[:] x, const int &start, int &end)

cdef void release_memory(AcfPtr model)