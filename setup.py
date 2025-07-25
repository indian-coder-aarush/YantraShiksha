from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'Ganit',                # Module name
        ['Ganit/bindings.cpp','Ganit/storage.cpp','Ganit/Tanitra.cpp','Ganit/Autograd.cpp'
         ],          # Source files
        include_dirs=[pybind11.get_include()],  # Include pybind11 headers
        language='c++'
    ),
]

setup(
    name='Ganit',
    ext_modules=ext_modules,
    zip_safe=False,
)