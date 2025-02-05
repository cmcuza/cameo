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
        name="compressors.cameo",
        sources=["compressors/cameo.pyx"],
        language="c++",
        extra_compile_args=[openmp_arg, opt_compiler],
        extra_link_args=[openmp_arg] if '-f' in openmp_arg else []
    ),
    Extension(
        name="compressors.heap",
        sources=["compressors/heap.pyx"],
        language="c++",
        extra_compile_args=[opt_compiler]
    ),
    Extension(
        name="compressors.pip",
        sources=["compressors/pip.pyx"],
        language="c++",
        extra_compile_args=[opt_compiler]
    ),
    Extension(
        name="compressors.pip_heap",
        sources=["compressors/pip_heap.pyx"],
        language="c++",
        extra_compile_args=[opt_compiler]
    ),
    Extension(
        name="compressors.visvalingam_whyat",
        sources=["compressors/visvalingam_whyat.pyx"],
        language="c++",
        extra_compile_args=[opt_compiler]
    ),
    Extension(
        name="compressors.turning_point",
        sources=["compressors/turning_point.pyx"],
        language="c++",
        extra_compile_args=[opt_compiler]
    ),
    Extension(
        name="compressors.agg_cameo",
        sources=["compressors/agg_cameo.pyx"],
        language="c++",
        extra_compile_args=[openmp_arg, opt_compiler],
        extra_link_args=[openmp_arg] if '-f' in openmp_arg else []
    ),
    Extension(
        name="compressors.inc_acf",
        sources=["compressors/inc_acf.pyx"],
        language="c++",
        extra_compile_args=[opt_compiler]
    ),
    Extension(
        name="compressors.inc_acf_agg",
        sources=["compressors/inc_acf_agg.pyx"],
        language="c++",
        extra_compile_args=[opt_compiler]
    ),
    Extension(
        name="compressors.math_utils",
        sources=["compressors/math_utils.pyx"],
        language="c++",
        extra_compile_args=[openmp_arg, opt_compiler],
        extra_link_args=[openmp_arg] if '-f' in openmp_arg else []
    ),
]

setup(
    name='cameo',
    version="0.1",
    packages=["compressors"],
    # extra_compile_args=["-g"],
    ext_modules=cythonize(extensions,
                          show_all_warnings=True,
                          # compiler_directives={'linetrace': True, 'binding': True},
                          annotate=True),
    include_dirs=[np.get_include()]
)