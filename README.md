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
import compressors.visvalingam_whyat as vw
from data_loader import DataFactory
import numpy as np

factory = DataFactory()
data_loader = factory.load_data('hepc', 'data')
nlags = data_loader.seasonality
y = np.squeeze(data_loader.data.values)
x = np.arange(y.shape[0])
error_bound = 0.01
vw_output = vw.simplify(x, y, nlags, error_bound)
decompressed_points = np.interp(np.arange(y.shape[0]), x[vw_output], y[vw_output])
print('Compression ratio:', round(y.shape[0]/np.sum(vw_output), 2))
print('Decompression MSE:', np.mean((decompressed_points-y)**2))
```

The same procedure applies for `turning points` and `perceptual important points`. 

### Running PMC, SWING and SP

Install [TerseTS](https://github.com/cmcuza/TerseTS/) and you are ready to go!

### Running SWAB

```python
from compressors.swab import swab
from data_loader import DataFactory
import numpy as np

factory = DataFactory()
data_loader = factory.load_data('hepc', 'data')
nlags = data_loader.seasonality
y = np.squeeze(data_loader.data.values)
x = np.arange(y.shape[0])
error_bound = 0.01
segments = swab(x, y, error_bound)
remaining_points = np.concatenate(segments)
decompressed_points = np.interp(np.arange(y.shape[0]), x[remaining_points], y[remaining_points])
print('Compression ratio:', round(y.shape[0]/len(remaining_points), 2))
print('Decompression MSE:', np.mean((decompressed_points-y)**2))
```

### Running Anomaly Detection Experiments

Just run `run_anomaly_detection_exp.py` with the desired `compressor` and `error bound`.

### Running Forecasting Experiments

Just run `run_forecasting_exp.py` with the desired `compressor` and `error bound`.

