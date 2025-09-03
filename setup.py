from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'Math',                # Module name
        ['Math/bindings.cpp','Math/storage.cpp','Math/Tensor.cpp','Math/Autograd.cpp'
         ],          # Source files
        include_dirs=[pybind11.get_include()],  # Include pybind11 headers
        language='c++'
    ),
]

setup(
    name='Math',
    ext_modules=ext_modules,
    zip_safe=False,
)