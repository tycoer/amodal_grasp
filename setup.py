from setuptools import setup, find_packages
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
# from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy
numpy_include_dir = numpy.get_include()


if __name__ == '__main__':
    triangle_hash_module = Extension(
        'utils.libmesh.triangle_hash',
        sources=[
            'utils/libmesh/triangle_hash.pyx'
        ],
        libraries=['m'],  # Unix-like specific
        include_dirs=[numpy_include_dir]
    )

    setup(
        name = "amodal_grip",
        version = "0.1",
        packages = find_packages(),
        ext_modules=cythonize([triangle_hash_module])
        )
