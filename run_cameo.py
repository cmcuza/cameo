import sys
from compressors.cameo import simplify_by_blocking
from data_loader import *


if __name__ == '__main__':
    data_name = sys.argv[1]
    acf_threshold = float(sys.argv[2])
    factory = DataFactory()
    data_loader = factory.load_data(data_name, 'data')
    acf = data_loader.seasonality
    y = np.squeeze(data_loader.data.values)
    x = np.arange(y.shape[0])
    hops = np.log(y.shape[0])*10
    cameo_out = simplify_by_blocking(x, y.copy(), hops, acf, acf_threshold)
    decompressed_points = np.interp(np.arange(y.shape[0]), x[cameo_out], y[cameo_out])
    print('Compression ratio:', round(y.shape[0]/np.sum(cameo_out), 2))
    print('Decompression MSE:', np.mean((decompressed_points-y)**2))

