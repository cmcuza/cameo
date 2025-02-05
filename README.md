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
(Optional but recommended) Set up a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
## Installing Dependencies

Install the required Python package:
```bash
pip install -r requirements.txt
```

## Compiling Cython Code

Compile the Cython modules:

```bash
python setup.py build_ext --inplace
```

### Running the CAMEO

To run a simple example run the main script. The main script expects a dataset name and the acf error-bound, for example:

```bash
python run_cameo.py hepc 0.001
```

### Running other Compressors

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
compressed_values = vw.simplify(x, y, nlags, error_bound)
```

The same procedure applies for `turning points` and `perceptual important points`. 




