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
pip install Cython
pip install pandas
```

## Compiling Cython Code

Compile the Cython modules:

```bash
python setup.py build_ext --inplace
```

### Running the Application

To run a simple example run the main script. The main script expects a dataset name and the acf error-bound, for example:

```bash
python main hepc 0.001
```






