"""Build the C++ extension with pybind11's setuptools helpers.

``Pybind11Extension`` sets the include paths, C++ standard, and platform flags; ``build_ext``
compiles it. This is the lightweight packaging path (no CMake). For larger projects, prefer
scikit-build-core + CMake, which ``pybind11`` also supports.

Build in place for a quick try:      python setup.py build_ext --inplace
Or install the example package:      pip install --no-build-isolation .
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

import bausatz  # the library ships the C++ headers; locate them with get_include()

ext_modules = [
    Pybind11Extension(
        "affine._affine",   # -> affine/_affine*.so
        ["src/affine.cpp"],
        include_dirs=[bausatz.get_include()],   # #include <bausatz/component.hpp> / <bausatz/pybind.hpp>
        cxx_std=17,
    ),
]

setup(
    name="bausatz-affine",
    version="0.1.0",
    description="Example: a C++ AffineKernel driven from Python as a bausatz component",
    packages=["affine"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.12",
    install_requires=["bausatz", "pybind11>=2.10"],
)
