import numpy as np
from compression.lpc.cameo import simplify_by_sip
from compression.lpc.visvalingam_whyat import simplify_by_vw
from compression.lpc.pip import simplify_by_pip
from compression.lpc.turning_point import simplify_by_tp
from compression.lpc.swab import simplify_by_swab


def compute_error(args):
    x_1, x_2, function, raw_acf, dist = args
    return dist(raw_acf, function(x_1, x_2))


class LineSimplification(object):
    sip_error_bounds = np.round(np.linspace(0.01, 0.1, 10), 3)
    agg_sip_error_bounds = np.round(np.linspace(0.001, 0.01, 5), 3)
    solar_sip_error_bounds = np.round(np.linspace(0.0001, 0.001, 5), 4)
    targets = []

    def __init__(self):
        self.x = None
        self.n = None
        self.target = None
        self.acf_threshold = None
        self.kappa = None
        self.nlags = None
        self.blocking = None
        self.no_removed_indices = None

    def set_target(self, target):
        self.target = target

    def simplify_vw(self):
        self.no_removed_indices = simplify_by_vw(self.x[:, 0].astype(int), self.x[:, 1], self.nlags, self.acf_threshold)

    def simplify_pip(self):
        self.no_removed_indices = simplify_by_pip(self.x[:, 1], self.nlags, self.acf_threshold)

    def simplify_tp(self):
        self.no_removed_indices = simplify_by_tp(self.x[:, 1], self.nlags, self.acf_threshold)

    def simplify_swab(self):
        self.no_removed_indices = simplify_by_swab(self.x[:, 0].astype(int), self.x[:, 1], self.acf_threshold)

    def simplify_sip(self):
        self.no_removed_indices = simplify_by_sip(self.x[:, 0].astype(int), self.x[:, 1],
                                                       self.blocking,
                                                       self.nlags, self.acf_threshold)

    def from_acf_threshold(self):
        pos = np.where(self.no_removed_indices)[0]
        return self.x[pos]

    def compress(self, pts, acf_threshold, nlags=None, blocking=None, kappa=None):
        x = list(range(len(pts)))
        self.x = np.asarray(list(zip(x, pts)))
        self.n = self.x.shape[0]
        self.kappa = kappa
        self.nlags = nlags
        self.blocking = blocking
        if kappa and self.n % kappa != 0.:
            self.n = (self.n // kappa) * kappa
            self.x = self.x[:self.n, :]

        self.acf_threshold = acf_threshold
        self.no_removed_indices = np.zeros(self.n, dtype=int)

        if self.target == 'sip':
            self.simplify_sip()
        elif self.target == 'vw':
            self.simplify_vw()
        elif self.target == 'pip':
            self.simplify_pip()
        elif self.target == 'tp':
            self.simplify_tp()
        elif self.target == 'swap':
            self.simplify_swab()
        else:
            raise Exception("No right target defined...")

        remaining_points = self.from_acf_threshold()
        return remaining_points

    def decompress(self, remaining_points):
        x = np.arange(0, int(remaining_points[-1, 0])+1)
        return np.interp(x, remaining_points[:, 0], remaining_points[:, 1])


