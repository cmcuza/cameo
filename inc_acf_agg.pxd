from cython_modules.inc_acf cimport AcfPtr

cdef void fit(AcfPtr model, double[:] x, double [:] aggregates, int kappa)

cdef void initialize(AcfPtr model, const int& nlags)

cdef void update(AcfPtr model, double[:] x, double[:] aggregates,
                 const double &x_a, const int &index, const int &kappa)

cdef inline void update_inside_lags(AcfPtr model, double[:] x,
                                    const double &delta,
                                    const double &delta_ss,
                                    const int &index)

cdef inline void update_outside_lags(AcfPtr model, double[:] aggregates,
                                     const double &delta,
                                     const double &delta_ss,
                                     const int &index)

cdef void interpolate_update(AcfPtr model, double[:] x, double[:] aggregates, int &start, int &end, int kappa)

cdef void interpolate_update_outside_lags(AcfPtr model, double[:] x, double[:] aggregates,
                                          int &start, int &end, int &start_index_a,
                                          int &end_index_a, int &kappa)

cdef void interpolate_update_inside_lags(AcfPtr model, double[:] x, double[:] aggregates,
                                          int &start, int &end, int &start_index_a,
                                          int &end_index_a, int &kappa)