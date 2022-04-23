try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

triangle_hash_module = Extension(
    'utils.libmesh.triangle_hash',
    sources=[
        'utils/libmesh/triangle_hash.pyx'
    ],
    libraries=['m'],  # Unix-like specific
    include_dirs=[numpy_include_dir]
)

# Gather all extension modules
ext_modules = [
    triangle_hash_module,
]

setup(
    ext_modules=cythonize(ext_modules),
)
