"""End-to-end test of the C++-backed component.

Run after building the extension (`python setup.py build_ext --inplace`):
    python -m pytest examples/pybind/test_affine.py
"""

from __future__ import annotations

# The built ``bausatz_affine`` package (with its compiled ``_affine`` extension) imports directly
# because pytest's default (prepend) import mode puts this directory on sys.path when collecting.
import pytest
from bausatz_affine import Affine, AffineKernel

from bausatz import ComponentFactory


def test_raw_kernel_is_a_bound_cpp_class():
    k = AffineKernel(scale=3.0, bias=1.0)
    assert k.apply(2.0) == 7.0
    assert k.scale == 3.0 and k.bias == 1.0
    assert repr(k).startswith("AffineKernel(scale=3")


def test_factory_builds_cpp_component_from_spec():
    # A spec dict -> pydantic-validated Config -> Affine -> C++ AffineKernel, all through the
    # ordinary factory. The component self-registered as "affine" on import.
    comp = ComponentFactory.get().create({"class_name": "affine", "scale": 10.0, "bias": -1.0})
    assert isinstance(comp, Affine)
    assert comp.forward(2.0) == 19.0             # 10*2 - 1, computed in C++
    assert comp.engine is comp.engine            # engine cached


def test_batched_path_crosses_stl_type_caster():
    comp = Affine(Affine.Config(scale=2.0, bias=0.5))
    # Python list -> std::vector -> C++ loop (GIL released) -> std::vector -> Python list.
    assert comp.forward_many([0.0, 1.0, 2.0]) == [0.5, 2.5, 4.5]


def test_cpp_exception_becomes_python_valueerror():
    # AffineKernel's C++ ctor throws std::invalid_argument for scale==0; pybind11 translates it.
    with pytest.raises(ValueError, match="scale must be non-zero"):
        Affine(Affine.Config(scale=0.0)).engine


def test_round_trips_through_config():
    comp = ComponentFactory.get().create({"class_name": "affine", "scale": 4.0})
    clone = ComponentFactory.get().create(comp.to_dict())
    assert clone.forward(3.0) == comp.forward(3.0)
