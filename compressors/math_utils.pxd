from compressors.inc_acf cimport AcfAgg

cdef double dot_product(double[:] x, double[:] y)

cdef void cumsum_cumsum(double[:] arr, double[:] cumsum1, double[:] cumsum2)

cdef double triangle_area(const double &x_p1, const double &y_p1,
                                 const double &x_p2, const double &y_p2,
                                 const double &x_p3, const double &y_p3)

cdef double no_gil_mae(double* x, double* y, int n) nogil

cdef double mae(double* x, double* y, int n)

cdef double csum(double[:] x, int n)

cdef double corrcoef(double[:] x, double[:] y)

cdef double mean(double[:] x)

cdef double std(double[:] x, const double & mu)

cdef void compute_acf_fall(AcfAgg *model, double[:] x, double * raw_acf, double * acf_error) nogil

cdef void compute_acf_agg_mean_fall(AcfAgg *model, double [:]x, double [:]aggregates,
                               double *raw_acf, double *acf_error, int x_n, int kappa) nogil

cdef void compute_acf_agg_sum_fall(AcfAgg *model, double [:]x, double [:]aggregates,
                               double *raw_acf, double *acf_error, int x_n, int kappa) nogil