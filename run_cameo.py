import sys
import numpy as np
from utils.metrics import nrmse
from data_loader import DataFactory
from compression.line_simplification import LineSimplification


if __name__ == '__main__':
    data_name = sys.argv[1]
    error_bound = float(sys.argv[2])
    factory = DataFactory()
    data_loader = factory.load_data(data_name, 'data')
    nlags = data_loader.seasonality
    y = np.squeeze(data_loader.data.values)
    x = np.arange(y.shape[0])
    hops = int(np.log(y.shape[0])*10)
    kappa = data_loader.aggregation

    line_simp = LineSimplification()
    line_simp.set_target(target='tp') # can be vw, tp, pip
    comp_y = line_simp.compress(y.copy(), error_bound, nlags, hops, kappa)
    decomp_y = line_simp.decompress(comp_y)
    print('Compression ratio:', round(y.shape[0]/comp_y.shape[0], 2))
    print('Decompression NRMSE:', np.round(nrmse(decomp_y, y), 4))






