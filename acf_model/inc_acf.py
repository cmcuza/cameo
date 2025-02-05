import numpy as np
from multiprocessing import Pool


def compute_pacf_helper(args):
    __pacf, pw_acf = args
    return __pacf(pw_acf)


class IncACF:
    def __init__(self, nlags):
        # This class computes the acf incrementally given that the ts changes
        # The raw acf has a precision of up to 11 decimal points
        self.__sxy = np.zeros(nlags, dtype=np.float64)
        self.__x_mean = np.empty(nlags, dtype=np.float64)
        self.__y_mean = np.empty(nlags, dtype=np.float64)
        self.__x_std = np.empty(nlags, dtype=np.float64)
        self.__y_std = np.empty(nlags, dtype=np.float64)
        self.__xss = np.empty(nlags, dtype=np.float64)
        self.__yss = np.empty(nlags, dtype=np.float64)
        self.__n = np.empty(nlags, dtype=np.float64)
        self.__x = np.empty(0, dtype=np.float64)
        self.__nlags = nlags
        self.__nls = np.arange(1, nlags + 1)
        self.max_inter = 0
        self.precision = 15

    def __extract_means(self):
        cum_sum = np.cumsum(self.__x)
        n = len(self.__x)
        __xs = np.empty(self.__nlags, dtype=np.float64)
        __ys = np.empty(self.__nlags, dtype=np.float64)

        for lag in range(1, self.__nlags + 1):
            __xs[lag - 1] = cum_sum[-lag - 1]
            __ys[lag - 1] = cum_sum[-1] - cum_sum[lag-1]
            self.__n[lag - 1] = n - lag

        self.__x_mean = __xs / self.__n
        self.__y_mean = __ys / self.__n

    def __extract_sxy(self):
        for lag in range(1, self.__nlags + 1):
            self.__sxy[lag - 1] = self.__x[:-lag] @ self.__x[lag:]

    def __extract_sums_squared(self):
        cum_sum = np.cumsum(np.power(self.__x, 2))
        for lag in range(1, self.__nlags + 1):
            self.__xss[lag - 1] = cum_sum[-lag - 1]
            self.__yss[lag - 1] = cum_sum[-1] - cum_sum[lag-1]

    def __extract_stds(self):
        self.__x_std = self.__xss / self.__n - self.__x_mean ** 2
        self.__y_std = self.__yss / self.__n - self.__y_mean ** 2

    def __extract_all(self):
        x_cum_sum = np.cumsum(self.__x)
        power_cum_sum = np.cumsum(np.power(self.__x, 2))
        n = len(self.__x)
        __xs = np.empty(self.__nlags, dtype=np.float64)
        __ys = np.empty(self.__nlags, dtype=np.float64)
        for lag in range(1, self.__nlags + 1):
            __xs[lag - 1] = x_cum_sum[-lag - 1]
            __ys[lag - 1] = x_cum_sum[-1] - x_cum_sum[lag-1]
            self.__xss[lag - 1] = power_cum_sum[-lag - 1]
            self.__yss[lag - 1] = power_cum_sum[-1] - power_cum_sum[lag - 1]
            self.__sxy[lag - 1] = self.__x[:-lag] @ self.__x[lag:]
            self.__n[lag - 1] = n - lag

        self.__x_mean = __xs / self.__n
        self.__y_mean = __ys / self.__n
        self.__x_std = self.__xss / self.__n - self.__x_mean ** 2
        self.__y_std = self.__yss / self.__n - self.__y_mean ** 2

    def fit(self, x):
        self.__x = x.copy()

        self.__extract_all()
        # self.__extract_means()
        # self.__extract_sums_squared()
        # self.__extract_sxy()
        # self.__extract_stds()

    def pacf_helper(self, acf_):
        order = self.__nlags + 1
        acf_lags = np.empty(order)
        acf_lags[0] = 1.
        acf_lags[1:] = acf_
        phi = np.zeros((order, order))
        phi[1, 1] = acf_lags[1]
        for k in range(2, order):
            phi[k, k] = acf_lags[k] - np.dot(phi[1:k, k - 1], acf_lags[1:k][::-1])
            phi[k, k] /= 1. - np.dot(phi[1:k, k - 1], acf_lags[1:k])
            for j in range(1, k):
                phi[j, k] = phi[j, k - 1] - phi[k, k] * phi[k - j, k - 1]

        pacf_ = np.diag(phi).copy()
        pacf_[0] = 1.0
        return pacf_

    def acf(self):
        return np.around((self.__sxy/self.__n - self.__x_mean * self.__y_mean)/(np.sqrt(self.__x_std * self.__y_std)), self.precision)

    def acovf(self):
        return np.around((self.__sxy/(self.__n-1) - (self.__n**2)*self.__x_mean * self.__y_mean/((self.__n-1)**2)), self.precision)

    def pacf(self):
        order = self.__nlags + 1
        acf_lags = np.empty(order)
        acf_lags[0] = 1.
        acf_lags[1:] = self.acf()
        phi = np.zeros((order, order))
        phi[1, 1] = acf_lags[1]
        for k in range(2, order):
            phi[k, k] = acf_lags[k] - np.dot(phi[1:k, k - 1], acf_lags[1:k][::-1])
            phi[k, k] /= 1. - np.dot(phi[1:k, k - 1], acf_lags[1:k])
            for j in range(1, k):
                phi[j, k] = phi[j, k - 1] - phi[k, k] * phi[k - j, k - 1]

        pacf_ = np.diag(phi).copy()
        pacf_[0] = 1.0
        return pacf_

    def levinson_durbin_derivation(self):
        order = self.__nlags + 1
        sxx_m = np.empty(order)
        sxx_m[0] = np.var(self.__x)
        sxx_m[1:] = self.acovf()
        phi = np.zeros((order, order))
        sig = np.zeros(order)
        phi[1, 1] = sxx_m[1] / sxx_m[0]
        sig[1] = sxx_m[0] - phi[1, 1] * sxx_m[1]
        for k in range(2, order):
            phi[k, k] = (sxx_m[k] - np.dot(phi[1:k, k - 1], sxx_m[1:k][::-1])) / sig[k - 1]
            for j in range(1, k):
                phi[j, k] = phi[j, k - 1] - phi[k, k] * phi[k - j, k - 1]
            sig[k] = sig[k - 1] * (1 - phi[k, k] ** 2)

        pacf_ = np.diag(phi).copy()
        pacf_[0] = 1.0

        return pacf_

    def get_updated_acf(self, x_a, index):
        index = int(index)
        delta = x_a - self.__x[index]

        x_std = self.__x_std.copy()
        y_std = self.__y_std.copy()
        x_mean = self.__x_mean.copy()
        y_mean = self.__y_mean.copy()

        delta_ss = delta * (delta + 2*self.__x[index])
        delta_mean = delta / self.__n

        mask1 = index >= self.__nls
        yss = self.__yss[mask1] + delta_ss
        tmp_y_mean = self.__y_mean[mask1] + delta_mean[mask1]
        y_std[mask1] = yss / self.__n[mask1] - tmp_y_mean ** 2
        y_mean[mask1] = tmp_y_mean

        mask2 = index < self.__n
        xss = self.__xss[mask2] + delta_ss
        tmp_x_mean = self.__x_mean[mask2] + delta_mean[mask2]
        x_std[mask2] = xss / self.__n[mask2] - tmp_x_mean ** 2
        x_mean[mask2] = tmp_x_mean

        delta_sxy = (self.__x[index - self.__nls]) * mask1
        delta_sxy += (self.__x[index + self.__nls * mask2]) * mask2
        delta_sxy = delta * delta_sxy
        sxy = self.__sxy + delta_sxy

        return (sxy/self.__n - x_mean * y_mean) / (np.sqrt(x_std * y_std))

    def get_delta_pacf(self, x_a, index):
        acf_ = self.get_updated_acf(x_a, index)
        return self.pacf_helper(acf_)

    def get_delta_acovf(self, x_a, index):
        index = int(index)
        delta = x_a - self.__x[index]

        x_std = self.__x_std.copy()
        y_std = self.__y_std.copy()
        x_mean = self.__x_mean.copy()
        y_mean = self.__y_mean.copy()
        sxy = self.__sxy.copy()

        delta_ss = delta * (delta + 2*self.__x[index])
        delta_mean = delta / self.__n

        mask1 = index >= self.__nls
        yss = self.__yss[mask1] + delta_ss
        tmp_y_mean = self.__y_mean[mask1] + delta_mean[mask1]
        y_std[mask1] = yss / self.__n[mask1] - tmp_y_mean ** 2
        y_mean[mask1] = tmp_y_mean

        mask2 = index < self.__n
        xss = self.__xss[mask2] + delta_ss
        tmp_x_mean = self.__x_mean[mask2] + delta_mean[mask2]
        x_std[mask2] = xss / self.__n[mask2] - tmp_x_mean ** 2
        x_mean[mask2] = tmp_x_mean

        mask1_and_mask2 = mask1 & mask2
        nls = self.__nls[mask1_and_mask2]
        sxy[mask1_and_mask2] = self.__sxy[mask1_and_mask2] + delta * (self.__x[index - nls] + self.__x[index + nls])
        neg_mask1 = ~mask1
        neg_mask2 = ~mask2
        sxy[neg_mask1] = self.__sxy[neg_mask1] + delta * self.__x[index + self.__nls[neg_mask1]]
        sxy[neg_mask2] = self.__sxy[neg_mask2] + delta * self.__x[index - self.__nls[neg_mask2]]

        acovf_ = (sxy/self.__n - x_mean * y_mean)

        return acovf_

    def __update_inside_lags(self, delta, index):
        mask1 = index >= self.__nls
        mask2 = index < self.__n
        mask1_and_mask2 = mask1 & mask2

        delta_mean = delta / self.__n
        self.__y_mean[mask1] += delta_mean[mask1]
        self.__x_mean[mask2] += delta_mean[mask2]

        delta_ss = delta * (2 * self.__x[index] + delta)

        self.__yss[mask1] += delta_ss
        self.__xss[mask2] += delta_ss

        self.__y_std[mask1] = self.__yss[mask1] / self.__n[mask1] - self.__y_mean[mask1] ** 2
        self.__x_std[mask2] = self.__xss[mask2] / self.__n[mask2] - self.__x_mean[mask2] ** 2

        nls = self.__nls[mask1_and_mask2]
        self.__sxy[mask1_and_mask2] += delta * (self.__x[index - nls] + self.__x[index + nls])
        self.__sxy[~mask1] += delta * self.__x[index + self.__nls[~mask1]]
        self.__sxy[~mask2] += delta * self.__x[index - self.__nls[~mask2]]

    def __update_outside_lags(self, delta, index):
        delta_mean = delta / self.__n
        self.__y_mean += delta_mean
        self.__x_mean += delta_mean

        delta_ss = delta * (2 * self.__x[index] + delta)

        self.__yss += delta_ss
        self.__xss += delta_ss

        self.__y_std = self.__yss / self.__n - self.__y_mean ** 2
        self.__x_std = self.__xss / self.__n - self.__x_mean ** 2

        self.__sxy += delta * (self.__x[index - self.__nls] + self.__x[index + self.__nls])

    def __compute_pw_acf_outside_lags(self, indices, deltas, delta_mean, delta_ss):
        y_mean = self.__y_mean + delta_mean
        yss = self.__yss + delta_ss[:, np.newaxis]
        y_std = np.sqrt(yss / self.__n - y_mean ** 2)

        x_mean = self.__x_mean + delta_mean
        xss = self.__xss + delta_ss[:, np.newaxis]
        x_std = np.sqrt(xss / self.__n - x_mean ** 2)

        sxy = deltas[:, np.newaxis] * (self.__x[indices[:, np.newaxis] - self.__nls] +
                                       self.__x[indices[:, np.newaxis] + self.__nls]) + self.__sxy

        return (sxy/self.__n - x_mean * y_mean) / (x_std * y_std)

    def __compute_pw_acovf_outside_lags(self, indices, deltas, delta_mean):
        y_mean = self.__y_mean + delta_mean

        x_mean = self.__x_mean + delta_mean

        sxy = deltas[:, np.newaxis] * (self.__x[indices[:, np.newaxis] - self.__nls] +
                                       self.__x[indices[:, np.newaxis] + self.__nls]) + self.__sxy

        return sxy/self.__n - x_mean * y_mean

    def __compute_pw_acf_bellow_lower_lags(self, indices, deltas, delta_mean, delta_ss):
        mask1 = np.tri(self.__nlags, self.__nlags, dtype=bool, k=0)

        y_mean = self.__y_mean + delta_mean*mask1
        yss = self.__yss + delta_ss[:, np.newaxis]*mask1
        y_std = np.sqrt(yss / self.__n - y_mean ** 2)

        x_mean = self.__x_mean + delta_mean
        xss = self.__xss + delta_ss[:, np.newaxis]
        x_std = np.sqrt(xss / self.__n - x_mean ** 2)

        delta_sxy = (self.__x[indices[:, np.newaxis] - self.__nls]) * mask1
        delta_sxy += (self.__x[indices[:, np.newaxis] + self.__nls])
        delta_sxy = deltas[:, np.newaxis] * delta_sxy
        sxy = delta_sxy + self.__sxy

        return (sxy/self.__n - x_mean * y_mean) / (x_std * y_std)

    def __compute_pw_acovf_bellow_lower_lags(self, indices, deltas, delta_mean):
        mask1 = np.tri(self.__nlags, self.__nlags, dtype=bool, k=0)

        y_mean = self.__y_mean + delta_mean*mask1

        x_mean = self.__x_mean + delta_mean

        delta_sxy = (self.__x[indices[:, np.newaxis] - self.__nls]) * mask1
        delta_sxy += (self.__x[indices[:, np.newaxis] + self.__nls])
        delta_sxy = deltas[:, np.newaxis] * delta_sxy
        sxy = delta_sxy + self.__sxy

        return sxy/self.__n - x_mean * y_mean

    def __compute_pw_acf_above_upper_lags(self, indices, deltas, delta_mean, delta_ss):
        mask2 = indices[:, np.newaxis] < self.__n

        y_mean = self.__y_mean + delta_mean
        yss = self.__yss + delta_ss[:, np.newaxis]
        y_std = np.sqrt(yss / self.__n - y_mean ** 2)

        x_mean = self.__x_mean + delta_mean*mask2
        xss = self.__xss + delta_ss[:, np.newaxis]*mask2
        x_std = np.sqrt(xss / self.__n - x_mean ** 2)

        delta_sxy = (self.__x[indices[:, np.newaxis] - self.__nls])
        delta_sxy += (self.__x[indices[:, np.newaxis] + self.__nls*mask2]) * mask2
        delta_sxy = deltas[:, np.newaxis] * delta_sxy
        sxy = delta_sxy + self.__sxy

        return (sxy/self.__n - x_mean * y_mean) / (x_std * y_std)

    def __compute_pw_acovf_above_upper_lags(self, indices, deltas, delta_mean):
        mask2 = indices[:, np.newaxis] < self.__n

        y_mean = self.__y_mean + delta_mean

        x_mean = self.__x_mean + delta_mean*mask2

        delta_sxy = (self.__x[indices[:, np.newaxis] - self.__nls])
        delta_sxy += (self.__x[indices[:, np.newaxis] + self.__nls*mask2]) * mask2
        delta_sxy = deltas[:, np.newaxis] * delta_sxy
        sxy = delta_sxy + self.__sxy

        return sxy/self.__n - x_mean * y_mean

    def update(self, x_a, index):
        index = int(index)
        delta = x_a - self.__x[index]
        if delta != 0:
            if (index < self.__nls[-1]) | (index >= self.__n[-1]):
                self.__update_inside_lags(delta, index)
            else:
                self.__update_outside_lags(delta, index)

            self.__x[index] = x_a

    def compute_acf_fall(self):
        new_points = (self.__x[2:] + self.__x[:-2]) * 0.5
        indices = np.arange(1, int(self.__n[0]))
        deltas = new_points - self.__x[indices]

        delta_ss = deltas * (deltas + 2*self.__x[indices])
        delta_mean = deltas[:, np.newaxis] / self.__n

        outside = self.__compute_pw_acf_outside_lags(indices[self.__nlags:-self.__nlags],
                                                     deltas[self.__nlags:-self.__nlags],
                                                     delta_mean[self.__nlags:-self.__nlags],
                                                     delta_ss[self.__nlags:-self.__nlags])
        lower_inside = self.__compute_pw_acf_bellow_lower_lags(indices[:self.__nlags],
                                                               deltas[:self.__nlags],
                                                               delta_mean[:self.__nlags],
                                                               delta_ss[:self.__nlags])
        upper_inside = self.__compute_pw_acf_above_upper_lags(indices[-self.__nlags:],
                                                              deltas[-self.__nlags:],
                                                              delta_mean[-self.__nlags:],
                                                              delta_ss[-self.__nlags:])

        acf_ = np.around(np.concatenate([lower_inside, outside, upper_inside]), self.precision)

        return acf_

    def compute_acovf_fall(self):
        new_points = (self.__x[2:] + self.__x[:-2]) * 0.5
        indices = np.arange(1, int(self.__n[0]))
        deltas = new_points - self.__x[indices]

        delta_mean = deltas[:, np.newaxis] / self.__n

        outside = self.__compute_pw_acovf_outside_lags(indices[self.__nlags:-self.__nlags],
                                                       deltas[self.__nlags:-self.__nlags],
                                                       delta_mean[self.__nlags:-self.__nlags])
        lower_inside = self.__compute_pw_acovf_bellow_lower_lags(indices[:self.__nlags],
                                                                 deltas[:self.__nlags],
                                                                 delta_mean[:self.__nlags])
        upper_inside = self.__compute_pw_acovf_above_upper_lags(indices[-self.__nlags:],
                                                                deltas[-self.__nlags:],
                                                                delta_mean[-self.__nlags:])

        acovf_ = np.around(np.concatenate([lower_inside, outside, upper_inside]), self.precision)

        return acovf_

    def compute_pacf_fall(self):
        pw_acf = self.compute_acf_fall()
        pw_pacf = np.zeros((pw_acf.shape[0], self.__nlags+1))

        for i in range(pw_acf.shape[0]):
            pw_pacf[i] = self.pacf_helper(pw_acf[i, :])

        return pw_pacf

    def compute_pacf_fall_par(self):
        pw_acf = self.compute_acf_fall()
        pw_pacf = np.zeros((pw_acf.shape[0], self.__nlags + 1))

        with Pool() as pool:
            results = pool.map(compute_pacf_helper, [(self.pacf_helper, pw_acf[i, :]) for i in range(pw_acf.shape[0])])

        for i, result in enumerate(results):
            pw_pacf[i] = result

        return pw_pacf

    def __interpolate_update_outside_lags(self, start, end):
        slope = (end[1] - start[1]) / (end[0] - start[0])
        indices = np.arange(start[0] + 1, end[0], dtype=int)
        x_as = slope * (indices - start[0]) + start[1]

        deltas = x_as - self.__x[indices]

        delta_mean = np.sum(deltas[:, np.newaxis] / self.__n, axis=0)

        self.__y_mean += delta_mean
        self.__x_mean += delta_mean

        self.__sxy += np.sum(deltas[:, np.newaxis] * (self.__x[indices[:, np.newaxis] - self.__nls] +
                                                      self.__x[indices[:, np.newaxis] + self.__nls]), axis=0)

        for lag in self.__nls:
            self.__sxy[lag - 1] += deltas[:-lag] @ deltas[lag:]

        delta_ss = np.sum(deltas * (deltas + 2*self.__x[indices]), axis=0)

        self.__yss += delta_ss
        self.__xss += delta_ss

        self.__y_std = self.__yss / self.__n - self.__y_mean ** 2
        self.__x_std = self.__xss / self.__n - self.__x_mean ** 2

        self.__x[indices] = x_as

    def __get_interpolated_acf_outside_lags(self, start, end):
        slope = (end[1] - start[1]) / (end[0] - start[0])
        indices = np.arange(start[0] + 1, end[0], dtype=int)
        x_as = slope * (indices - start[0]) + start[1]

        deltas = x_as - self.__x[indices]

        delta_mean = np.sum(deltas[:, np.newaxis] / self.__n, axis=0)

        __y_mean = self.__y_mean + delta_mean
        __x_mean = self.__x_mean + delta_mean

        __sxy = self.__sxy + np.sum(deltas[:, np.newaxis] * (self.__x[indices[:, np.newaxis] - self.__nls] +
                                                             self.__x[indices[:, np.newaxis] + self.__nls]), axis=0)

        for lag in self.__nls:
            __sxy[lag - 1] += deltas[:-lag] @ deltas[lag:]

        delta_ss = np.sum(deltas * (deltas + 2*self.__x[indices]), axis=0)

        __yss = self.__yss + delta_ss
        __xss = self.__xss + delta_ss

        __y_std = __yss / self.__n - __y_mean ** 2
        __x_std = __xss / self.__n - __x_mean ** 2

        return np.around((__sxy/self.__n - __x_mean * __y_mean)/(np.sqrt(__x_std * __y_std)), self.precision)

    def __get_interpolated_acovf_outside_lags(self, start, end):
        slope = (end[1] - start[1]) / (end[0] - start[0])
        indices = np.arange(start[0] + 1, end[0], dtype=int)
        x_as = slope * (indices - start[0]) + start[1]

        deltas = x_as - self.__x[indices]

        delta_mean = np.sum(deltas[:, np.newaxis] / self.__n, axis=0)

        __y_mean = self.__y_mean + delta_mean
        __x_mean = self.__x_mean + delta_mean

        __sxy = self.__sxy + np.sum(deltas[:, np.newaxis] * (self.__x[indices[:, np.newaxis] - self.__nls] +
                                                             self.__x[indices[:, np.newaxis] + self.__nls]), axis=0)

        for lag in self.__nls:
            __sxy[lag - 1] += deltas[:-lag] @ deltas[lag:]

        return np.around((__sxy/self.__n - __x_mean * __y_mean), self.precision)

    def __interpolate_update_inside_lags(self, start, end):
        slope = (end[1] - start[1]) / (end[0] - start[0])
        indices = np.arange(start[0] + 1, end[0], dtype=int)
        x_as = slope * (indices - start[0]) + start[1]

        mask1 = indices[:, np.newaxis] >= self.__nls
        mask2 = indices[:, np.newaxis] < self.__n

        deltas = x_as - self.__x[indices]

        delta_mean = deltas[:, np.newaxis] / self.__n

        self.__y_mean += np.sum(delta_mean*mask1, axis=0)
        self.__x_mean += np.sum(delta_mean*mask2, axis=0)

        delta_ss = deltas * (deltas + 2*self.__x[indices])

        self.__yss += np.sum(delta_ss[:, np.newaxis] * mask1, axis=0)
        self.__xss += np.sum(delta_ss[:, np.newaxis] * mask2, axis=0)

        self.__y_std = self.__yss / self.__n - self.__y_mean ** 2
        self.__x_std = self.__xss / self.__n - self.__x_mean ** 2

        delta_sxy = (self.__x[indices[:, np.newaxis] - self.__nls]) * mask1
        delta_sxy += (self.__x[indices[:, np.newaxis] + self.__nls * mask2]) * mask2
        delta_sxy = deltas[:, np.newaxis] * delta_sxy

        self.__sxy += np.sum(delta_sxy, axis=0)

        for lag in self.__nls:
            self.__sxy[lag-1] += deltas[:-lag]@deltas[lag:]

        self.__x[indices] = x_as

    def __get_interpolated_acf_inside_lags(self, start, end):
        slope = (end[1] - start[1]) / (end[0] - start[0])
        indices = np.arange(start[0] + 1, end[0], dtype=int)
        x_as = slope * (indices - start[0]) + start[1]

        mask1 = indices[:, np.newaxis] >= self.__nls
        mask2 = indices[:, np.newaxis] < self.__n

        deltas = x_as - self.__x[indices]

        delta_mean = deltas[:, np.newaxis] / self.__n

        __y_mean = self.__y_mean + np.sum(delta_mean * mask1, axis=0)
        __x_mean = self.__x_mean + np.sum(delta_mean * mask2, axis=0)

        delta_ss = deltas * (deltas + 2*self.__x[indices])

        __yss = self.__yss + np.sum(delta_ss[:, np.newaxis] * mask1, axis=0)
        __xss = self.__xss + np.sum(delta_ss[:, np.newaxis] * mask2, axis=0)

        __y_std = __yss / self.__n - __y_mean ** 2
        __x_std = __xss / self.__n - __x_mean ** 2

        delta_sxy = (self.__x[indices[:, np.newaxis] - self.__nls]) * mask1
        delta_sxy += (self.__x[indices[:, np.newaxis] + self.__nls * mask2]) * mask2
        delta_sxy = deltas[:, np.newaxis] * delta_sxy

        __sxy = self.__sxy + np.sum(delta_sxy, axis=0)

        for lag in self.__nls:
            __sxy[lag - 1] += deltas[:-lag] @ deltas[lag:]

        return np.around((__sxy/self.__n - __x_mean * __y_mean) / (np.sqrt(__x_std * __y_std)), self.precision)

    def __get_interpolated_acovf_inside_lags(self, start, end):
        slope = (end[1] - start[1]) / (end[0] - start[0])
        indices = np.arange(start[0] + 1, end[0], dtype=int)
        x_as = slope * (indices - start[0]) + start[1]

        mask1 = indices[:, np.newaxis] >= self.__nls
        mask2 = indices[:, np.newaxis] < self.__n

        deltas = x_as - self.__x[indices]

        delta_mean = deltas[:, np.newaxis] / self.__n

        __y_mean = self.__y_mean + np.sum(delta_mean * mask1, axis=0)
        __x_mean = self.__x_mean + np.sum(delta_mean * mask2, axis=0)

        delta_sxy = (self.__x[indices[:, np.newaxis] - self.__nls]) * mask1
        delta_sxy += (self.__x[indices[:, np.newaxis] + self.__nls * mask2]) * mask2
        delta_sxy = deltas[:, np.newaxis] * delta_sxy

        __sxy = self.__sxy + np.sum(delta_sxy, axis=0)

        for lag in self.__nls:
            __sxy[lag - 1] += deltas[:-lag] @ deltas[lag:]

        return np.around(__sxy/self.__n - __x_mean * __y_mean, self.precision)

    def interpolate_and_update(self, start, end):
        if ((start[0] + 1) < self.__nls[-1]) | ((end[0]-1) >= self.__n[-1]):
            self.__interpolate_update_inside_lags(start, end)
        else:
            self.__interpolate_update_outside_lags(start, end)

    def get_interpolated_acf(self, start, end):
        if ((start[0] + 1) < self.__nls[-1]) | ((end[0]-1) >= self.__n[-1]):
            return self.__get_interpolated_acf_inside_lags(start, end)
        else:
            return self.__get_interpolated_acf_outside_lags(start, end)

    def get_interpolated_acovf(self, start, end):
        if ((start[0] + 1) < self.__nls[-1]) | ((end[0]-1) >= self.__n[-1]):
            return self.__get_interpolated_acovf_inside_lags(start, end)
        else:
            return self.__get_interpolated_acovf_outside_lags(start, end)

    def get_interpolated_pacf(self, start, end):
        if ((start[0] + 1) < self.__nls[-1]) | ((end[0]-1) >= self.__n[-1]):
            acf_ = self.__get_interpolated_acf_inside_lags(start, end)
        else:
            acf_ = self.__get_interpolated_acf_outside_lags(start, end)

        return self.pacf_helper(acf_)

    def get_x_mean(self):
        return self.__x_mean

    def get_y_mean(self):
        return self.__y_mean

    def get_x_std(self):
        return np.sqrt(self.__x_std)

    def get_y_std(self):
        return np.sqrt(self.__y_std)

    def get_sxy(self):
        return self.__sxy/self.__n

    def get_n(self):
        return self.__n

    def get_actual_acf(self):
        return np.asarray([np.corrcoef(self.__x[:-lag], self.__x[lag:])[0, 1] for lag in range(1, self.__nlags+1)])

    def get_actual_side_acf(self, start, end):
        x = self.__x.copy()
        slope = (end[1] - start[1]) / (end[0] - start[0])
        indices = np.arange(start[0] + 1, end[0], dtype=int)
        x[indices] = slope * (indices - start[0]) + start[1]
        return np.asarray([np.corrcoef(x[:-lag], x[lag:])[0, 1] for lag in range(1, self.__nlags+1)])

    def get_actual_pw_acf(self):
        nlags_xs = list()
        nlags_ys = list()

        for lag in range(1, self.__nlags + 1):
            nlags_xs.append(self.__x[:-lag].copy())
            nlags_ys.append(self.__x[lag:].copy())

        np_acf = np.empty((int(self.__n[0]-1), self.__nlags))

        for i in range(1, int(self.__n[0])):
            x_a = (self.__x[i - 1] + self.__x[i + 1]) / 2
            org_x = self.__x[i]

            for lag in range(1, self.__nlags + 1):
                if i < len(nlags_xs[lag - 1]):
                    nlags_xs[lag - 1][i] = x_a
                if i - lag >= 0:
                    nlags_ys[lag - 1][i - lag] = x_a

                np_acf[i-1][lag - 1] = np.round(np.corrcoef(nlags_xs[lag - 1], nlags_ys[lag - 1])[0, 1], self.precision - 1)

            for lag in range(1, self.__nlags + 1):
                if i < len(nlags_xs[lag - 1]):
                    nlags_xs[lag - 1][i] = org_x
                if i - lag >= 0:
                    nlags_ys[lag - 1][i - lag] = org_x

        return np_acf

    def get_x(self):
        return self.__x

    def get_interpolated_exy(self, start, end):
        if ((start[0] + 1) < self.__nls[-1]) | ((end[0] - 1) >= self.__n[-1]):
            return self.__get_interpolated_exy_inside_lags(start, end)
        else:
            return self.__get_interpolated_exy_outside_lags(start, end)

    def __get_interpolated_exy_inside_lags(self, start, end):
        slope = (end[1] - start[1]) / (end[0] - start[0])
        indices = np.arange(start[0] + 1, end[0], dtype=int)
        x_as = slope * (indices - start[0]) + start[1]

        mask1 = indices[:, np.newaxis] >= self.__nls
        mask2 = indices[:, np.newaxis] < self.__n

        deltas = x_as - self.__x[indices]

        delta_sxy = (self.__x[indices[:, np.newaxis] - self.__nls]) * mask1
        delta_sxy += (self.__x[indices[:, np.newaxis] + self.__nls * mask2]) * mask2
        delta_sxy = deltas[:, np.newaxis] * delta_sxy

        delta_sxy = np.sum(delta_sxy, axis=0)

        for lag in self.__nls:
            delta_sxy += deltas[:-lag] @ deltas[lag:]

        delta_sxy = delta_sxy / np.sum(1 + 2 * deltas * self.__x[indices])

        return np.sum(np.abs(delta_sxy))

    def __get_interpolated_exy_outside_lags(self, start, end):
        slope = (end[1] - start[1]) / (end[0] - start[0])
        indices = np.arange(start[0] + 1, end[0], dtype=int)
        x_as = slope * (indices - start[0]) + start[1]

        deltas = x_as - self.__x[indices]

        delta_sxy = np.sum(deltas[:, np.newaxis] * (self.__x[indices[:, np.newaxis] - self.__nls] +
                                                    self.__x[indices[:, np.newaxis] + self.__nls]), axis=0)

        for lag in self.__nls:
            delta_sxy += deltas[:-lag] @ deltas[lag:]

        delta_sxy = delta_sxy/np.sum(1+2*deltas*self.__x[indices])

        return np.sum(np.abs(delta_sxy))

    def get_updated_exy(self, x_a, index):
        index = int(index)
        delta = x_a - self.__x[index]

        mask1 = index >= self.__nls

        mask2 = index < self.__n

        delta_sxy = (self.__x[index - self.__nls]) * mask1
        delta_sxy += (self.__x[index + self.__nls * mask2]) * mask2
        delta_sxy = delta * delta_sxy

        delta_sxy = delta_sxy / (1 + 2 * delta * self.__x[index])

        return np.abs(np.sum(delta_sxy))

