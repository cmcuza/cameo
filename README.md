# CAMEO: Autocorrelation-preserving Line Simplification for Lossy Time Series Compression 

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
python main 'hepc' 0.001
```






