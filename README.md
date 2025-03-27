# CAMEO: Autocorrelation-preserving Line Simplification for Lossy Time Series Compression 

CAMEO is a lossy time series compression algorithm capable of 
preserving the AutoCorrelation Function (ACF) of the decompressed data within an error bound of the original ACF.
CAMEO efficiently re-computes the ACF by keeping basic aggregates. CAMEO is implemented in Cython to increase its efficiency, avoid python GIL constraints, and take full advantage
of multi-threading.  

## Installation

Clone the repository:

```bash
git clone https://github.com/cmcuza/cameo.git
cd cameo
```

## Setting Up a Virtual Environment
(Optional but recommended) Set up a Anaconda virtual environment:

```bash
conda create --name cameo_env
conda activate cameo_en  # On Windows use `venv\Scripts\activate`
```
This way you can still all dependencies one by one. Another way is to create the environment directly using the provided `.yml`
## Installing Dependencies
Install the required Python package:
```bash
conda env create -f environment.yml
```

## Compiling Cython Code

Compile the Cython modules:

```bash
python setup.py build_ext --inplace
```

### Running CAMEO

To run a simple example run the main script. The main script expects a dataset name and the acf error-bound, for example:

```bash
python run_cameo.py hepc 0.001
```

### Running VW

To run any of the line-simplification methods implemented in `./compressors/` you can do the following:

```python
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
    kappa = data_loader.aggregation

    line_simp = LineSimplification()
    line_simp.set_target(target='vw') # can be vw, tp, pip
    comp_y = line_simp.compress(y.copy(), error_bound, nlags, None, kappa)
    decomp_y = line_simp.decompress(comp_y)
    print('Compression ratio:', round(y.shape[0]/comp_y.shape[0], 2))
    print('Decompression NRMSE:', np.round(nrmse(decomp_y, y), 4))
```

The same procedure applies for `turning points (tp)`, `perceptual important points (pip)` and `swab (swab)`. 

### Running PMC, SWING and SP

Install [TerseTS](https://github.com/cmcuza/TerseTS/) and you are ready to go!

### Running Anomaly Detection Experiments

Just run `run_anomaly_detection_exp.py` with the desired `compressor` and `error bound`.

### Running Forecasting Experiments

Just run `run_forecasting_exp.py` with the desired `compressor` and `error bound`.

