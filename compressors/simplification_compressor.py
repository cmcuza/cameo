from utils.heap import Heap
from acf_model.inc_acf import IncACF
from utils.stats import triangle_areas_from_array, np, triangle_area
from utils.metrics import mae
from compressors.cameo import simplify_by_blocking
from compressors.pip import simplify as simplify_by_pip
from compressors.turning_point import simplify as simplify_by_turning_point


def compute_error(args):
    x_1, x_2, function, raw_acf, dist = args
    return dist(raw_acf, function(x_1, x_2))


class ACFSimplifier(object):
    coefficients = [4, 5, 6, 7, 8, 9, 10]
   
    def __init__(self, target='acf'):
        self.x = None
        self.n = None
        self.target = target
        self.inc_acf = None
        self.raw_acf = None
        self.acf_threshold = None
        self.thresholds = None
        self.kappa = None
        self.nlags = None
        self.no_removed_indices = None
        self.dist = mae
        self.ordered_thresholds = None

    def pw_acf_error(self):
        normal_ace = np.empty(self.n, dtype=np.float64)
        normal_ace[0] = np.inf
        normal_ace[-1] = np.inf

        point_wise_acf = self.inc_acf.compute_acf_fall()
        error = self.dist(self.raw_acf, point_wise_acf, axis=1)
        normal_ace[1:-1] = error
        return normal_ace

    def get_pw_acf_error(self, ace_heap):
        min_node = 0
        min_node_value = np.inf

        for node in ace_heap.heap:
            if node[0] != np.inf:
                if node[2] + 2 < node[3]:
                    cd = self.dist(self.raw_acf, self.inc_acf.get_interpolated_acf(self.x[node[2]], self.x[node[3]]))
                else:
                    start, end = self.x[node[2]], self.x[node[3]]
                    inter = end - start
                    slope = inter[1] / inter[0]
                    x_a = slope + start[1]
                    cd = self.dist(self.raw_acf, self.inc_acf.get_updated_acf(x_a, start[0] + 1))
                if cd < min_node_value:
                    min_node_value = cd
                    min_node = node[1]

        return min_node

    def update(self, start, end):
        if start[0] + 2 < end[0]:
            self.inc_acf.interpolate_and_update(start, end)
        else:
            self.__simple_update_acf(start, end)

    def __get_pointwise_ace(self):
        return self.dist(self.raw_acf, self.inc_acf.compute_acf_fall(), axis=1)

    def __simple_update_acf(self, start, end):
        inter = end - start
        slope = inter[1] / inter[0]
        x_a = slope + start[1]

        self.inc_acf.update(x_a, start[0] + 1)

    def simplify_vw(self):
        real_areas = triangle_areas_from_array(self.x)
        areas_heap = Heap(real_areas)
        self.inc_acf = IncACF(self.nlags)
        self.inc_acf.fit(self.x[:, 1])
        self.raw_acf = self.inc_acf.acf()
        ace = 0.0
        order = 1 
        while areas_heap.top()[0] < np.inf:
            min_node = areas_heap.pop()
            this_area = min_node[0]

            if this_area != 0:
                self.update(self.x[min_node[2]], self.x[min_node[3]])
                ace = self.dist(self.raw_acf, self.inc_acf.acf())

            if self.x.shape[0]/(self.x.shape[0]-order) > self.acf_threshold:
                break

            self.no_removed_indices[min_node[1]] = order
            order += 1
            left, right = areas_heap.get_left_right(min_node)

            if right:
                right_area = triangle_area(self.x[right[1]],
                                           self.x[right[2]],
                                           self.x[right[3]])

                if right_area <= this_area:
                    right_area = this_area

                right = (right_area,) + right[1:]
                areas_heap.reheap(right)
                real_areas[right[1]] = right_area
            if left:
                left_area = triangle_area(self.x[left[1]],
                                          self.x[left[2]],
                                          self.x[left[3]])

                if left_area <= this_area:
                    left_area = this_area

                left = (left_area,) + left[1:]
                areas_heap.reheap(left)
                real_areas[left[1]] = left_area

    def simplify_pip(self):
        self.no_removed_indices = simplify_by_pip(self.x[:, 1], self.nlags, self.acf_threshold)

    def simplify_tp(self):
        self.no_removed_indices = simplify_by_turning_point(self.x[:, 1], self.nlags, self.acf_threshold)

    def simplify_cameo(self):
        self.no_removed_indices = simplify_by_blocking(self.x[:, 0].astype(int), self.x[:, 1],
                                                       np.log2(self.x.shape[0]).astype(int)*10,
                                                       self.nlags, self.acf_threshold)

    def from_acf_threshold(self):
        if self.target != 'tp':
            pos = np.where(np.array(self.no_removed_indices) == 0)[0]
        else:
            pos = np.where(np.array(self.no_removed_indices) != 0)[0]

        return self.x[pos]

    def compress(self, pts, acf_threshold, nlags=None, kappa=None):
        x = list(range(len(pts)))
        self.x = np.asarray(list(zip(x, pts)))
        self.n = self.x.shape[0]
        self.kappa = kappa
        self.nlags = nlags
        if kappa and self.n % kappa != 0.:
            raise ValueError('N and Kappa are wrong')

        self.acf_threshold = acf_threshold
        self.thresholds = np.empty(len(pts), dtype=np.float64)
        self.thresholds[0] = np.inf
        self.thresholds[-1] = np.inf
        self.no_removed_indices = np.zeros(self.n, dtype=int)

        if self.target == 'cameo':
            self.simplify_cameo()
        elif self.target == 'vw':
            self.simplify_vw()
        elif self.target == 'pip':
            self.simplify_pip()
        elif self.target == 'tp':
            self.simplify_tp()

        remaining_points = self.from_acf_threshold()
        return remaining_points

    def decompress(self, remaining_points):
        x = np.arange(0, int(remaining_points[-1, 0])+1)
        return np.interp(x, remaining_points[:, 0], remaining_points[:, 1])


