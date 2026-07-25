"""C++-backed bausatz components (example package).

The component classes live in :mod:`affine.components`; the C++ types come from the compiled
``_affine`` extension (built from ``../src/affine.cpp``). This module just re-exports the public API.
"""

from ._affine import AffineKernel, Transform
from ._affine import Component as NativeComponent
from .components import Affine, NativeSequential

__all__ = ["Affine", "AffineKernel", "NativeComponent", "NativeSequential", "Transform"]
