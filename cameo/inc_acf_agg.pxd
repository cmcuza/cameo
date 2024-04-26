from cameo.inc_acf cimport AcfPtr

cdef void fit_mean(AcfPtr model, double[:] x, double [:] aggregates, int kappa)

cdef void fit_sum(AcfPtr model, double[:] x, double [:] aggregates, int kappa)

cdef void fit_max(AcfPtr model, double[:] x, double [:] aggregates, int kappa)

cdef void initialize(AcfPtr model, const int& nlags)

cdef void update_mean(AcfPtr model, double[:] x, double[:] aggregates,
                      const double &x_a, const int &index, const int &kappa)

cdef void update_max(AcfPtr model, double[:] x, double[:] aggregates,
                     double &x_a, const int &index, const int &kappa)

cdef void update_sum(AcfPtr model, double[:] x, double[:] aggregates,
                 const double &x_a, const int &index, const int &kappa)

cdef inline void update_inside_lags(AcfPtr model, double[:] x,
                                    const double &delta,
                                    const double &delta_ss,
                                    const int &index)

cdef inline void update_outside_lags(AcfPtr model, double[:] aggregates,
                                     const double &delta,
                                     const double &delta_ss,
                                     const int &index)

cdef void interpolate_update_mean(AcfPtr model, double[:] x, double[:] aggregates, int &start, int &end, int kappa)

cdef void interpolate_update_sum(AcfPtr model, double[:] x, double[:] aggregates, int &start, int &end, int kappa)

cdef void interpolate_update_max(AcfPtr model, double[:] x, double[:] aggregates, int &start, int &end, int kappa)

cdef void interpolate_update_outside_lags_mean(AcfPtr model, double[:] x, double[:] aggregates,
                                               int &start, int &end, int &start_index_a,
                                               int &end_index_a, int &kappa)

cdef void interpolate_update_inside_lags_mean(AcfPtr model, double[:] x, double[:] aggregates,
                                              int &start, int &end, int &start_index_a,
                                              int &end_index_a, int &kappa)

cdef void interpolate_update_inside_lags_sum(AcfPtr model, double[:] x, double[:] aggregates,
                                             int &start, int &end, int &start_index_a,
                                             int &end_index_a, int &kappa)

cdef void interpolate_update_outside_lags_sum(AcfPtr model, double[:] x, double[:] aggregates,
                                              int &start, int &end, int &start_index_a,
                                              int &end_index_a, int &kappa)

cdef void interpolate_update_inside_lags_max(AcfPtr model, double[:] x, double[:] aggregates,
                                             int &start, int &end, int &start_index_a,
                                             int &end_index_a, int &kappa)

cdef void interpolate_update_outside_lags_max(AcfPtr model, double[:] x, double[:] aggregates,
                                              int &start, int &end, int &start_index_a,
                                              int &end_index_a, int &kappa)