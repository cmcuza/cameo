import sys
from cameo import simplify_by_blocking
from data_loader import *


if __name__ == '__main__':
    data_name = sys.argv[1]
    acf_threshold = float(sys.argv[2])
    data_loader = DataFactory().load_data(data_name, 'data')
    acf = data_loader.seasonality
    y = data_loader.data.values
    hops = np.log(y.shape[0])*10
    remaining_points = simplify_by_blocking(y, hops, acf, acf_threshold)
    print('Compression ratio', y.shape[0]/np.sum(remaining_points))

