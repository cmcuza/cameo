from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys

if sys.platform.startswith("win"):
    openmp_arg = '-openmp'
    opt_compiler = '/O2'
    # opt_compiler = '/O0'
else:
    openmp_arg = '-fopenmp'
    opt_compiler = '-O3'
    # opt_compiler = '-Og'

extensions = [
    Extension(
        name="compression.cython.heap",
        sources=["heap.pyx"],
        language="c++",
        extra_compile_args=[opt_compiler]
    ),
    Extension(
        name="compression.cython.cameo",
        sources=["cameo.pyx"],
        language="c++",
        extra_compile_args=[openmp_arg, opt_compiler],
        extra_link_args=[openmp_arg] if '-f' in openmp_arg else []
    ),
    Extension(
        name="compression.cython.agg_cameo",
        sources=["agg_cameo.pyx"],
        language="c++",
        extra_compile_args=[openmp_arg, opt_compiler],
        extra_link_args=[openmp_arg] if '-f' in openmp_arg else []
    ),
    Extension(
        name="compression.cython.inc_acf",
        sources=["inc_acf.pyx"],
        language="c++",
        extra_compile_args=[opt_compiler]
    ),
    Extension(
        name="compression.cython.inc_acf_agg",
        sources=["inc_acf_agg.pyx"],
        language="c++",
        extra_compile_args=[opt_compiler]
    ),
    Extension(
        name="compression.cython.math_utils",
        sources=["math_utils.pyx"],
        language="c++",
        extra_compile_args=[openmp_arg, opt_compiler],
        extra_link_args=[openmp_arg] if '-f' in openmp_arg else []
    ),
]

setup(
    name='ccameo',
    version="0.1",
    packages=["cython_modules"],
    # extra_compile_args=["-g"],
    ext_modules=cythonize(extensions,
                          show_all_warnings=True,
                          # compiler_directives={'linetrace': True, 'binding': True},
                          annotate=True),
    include_dirs=[np.get_include()]
)