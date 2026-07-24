"""End-to-end test of the C++-backed components.

Run after building the extension (`python setup.py build_ext --inplace`):
    python -m pytest examples/pybind/test_affine.py
"""

from __future__ import annotations

# The built ``bausatz_affine`` package (with its compiled ``_affine`` extension) imports directly
# because pytest's default (prepend) import mode puts this directory on sys.path when collecting.
import pytest
from bausatz_affine import (
    Affine,
    AffineConfig,
    AffineKernel,
    NativeComponent,
    NativeSequential,
    Transform,
)

from bausatz import ComponentFactory


# --------------------------------------------------------------------------- #
# The C++ layer: library base, app compute interface, config struct, leaf.
# --------------------------------------------------------------------------- #
def test_config_struct_is_a_bound_cpp_type():
    cfg = AffineConfig(scale=3.0, bias=1.0)
    assert cfg.scale == 3.0 and cfg.bias == 1.0
    assert AffineKernel(cfg).forward(2.0) == 7.0
    assert repr(cfg).startswith("AffineConfig(scale=3")


def test_forward_lives_on_the_app_interface_not_the_library_base():
    # bausatz::Component (the library base) carries no compute method; the example's Transform does.
    assert issubclass(Transform, NativeComponent)
    assert hasattr(Transform, "forward")
    assert not hasattr(NativeComponent, "forward")


def test_kernel_is_a_transform_and_a_library_component():
    k = AffineKernel(AffineConfig(scale=2.0, bias=0.5))
    assert isinstance(k, Transform)         # AffineKernel : Transform
    assert isinstance(k, NativeComponent)   # ... : bausatz::Component
    assert k.forward(10.0) == 20.5


def test_name_plumbing_on_the_library_base():
    k = AffineKernel(AffineConfig(scale=1.0))
    assert k.name == ""          # default from bausatz::Component
    k.name = "layer-0"
    assert k.name == "layer-0"


def test_zero_scale_cpp_exception_becomes_python_valueerror():
    with pytest.raises(ValueError, match="scale must be non-zero"):
        AffineKernel(AffineConfig(scale=0.0))


# --------------------------------------------------------------------------- #
# Affine: a CppComponent using config-struct passing.
# --------------------------------------------------------------------------- #
def test_factory_builds_affine_via_config_struct():
    comp = ComponentFactory.get().create(
        {"class_name": "affine", "name": "enc", "scale": 10.0, "bias": -1.0}
    )
    assert isinstance(comp, Affine)
    assert comp.forward(2.0) == 19.0             # 10*2 - 1, computed in C++
    assert isinstance(comp.engine, AffineKernel)
    assert comp.engine is comp.engine            # cached
    assert comp.engine.name == "enc"             # config.name carried onto the C++ base
    # The bad-config path surfaces the C++ exception through the component too.
    with pytest.raises(ValueError, match="scale must be non-zero"):
        Affine(Affine.Config(scale=0.0)).engine


def test_batched_path_crosses_stl_type_caster():
    comp = Affine(Affine.Config(scale=2.0, bias=0.5))
    assert comp.forward_many([0.0, 1.0, 2.0]) == [0.5, 2.5, 4.5]


# --------------------------------------------------------------------------- #
# NativeSequential: the whole chain runs in C++.
# --------------------------------------------------------------------------- #
def test_native_sequential_runs_the_chain_in_cpp():
    seq = ComponentFactory.get().create(
        {
            "class_name": "native-sequential",
            "stages": [
                {"class_name": "affine", "scale": 2.0, "bias": 0.0},   # x -> 2x
                {"class_name": "affine", "scale": 1.0, "bias": 5.0},   # x -> x + 5
            ],
        }
    )
    assert isinstance(seq, NativeSequential)
    # The engine is a single native composite (a Transform / bausatz::Component), not a Python loop.
    assert isinstance(seq.engine, Transform)
    assert seq.engine.size == 2
    assert seq.forward(3.0) == 11.0              # 2*3 = 6, then 6 + 5 = 11 — all in C++


def test_native_sequential_nests():
    nested = ComponentFactory.get().create(
        {
            "class_name": "native-sequential",
            "stages": [
                {"class_name": "native-sequential",
                 "stages": [{"class_name": "affine", "scale": 2.0}]},   # x -> 2x
                {"class_name": "affine", "scale": 1.0, "bias": 100.0},  # x -> x + 100
            ],
        }
    )
    assert nested.forward(3.0) == 106.0          # (2*3) then +100


def test_native_sequential_round_trips_through_config():
    spec = {
        "class_name": "native-sequential",
        "stages": [
            {"class_name": "affine", "scale": 2.0, "bias": 0.0},
            {"class_name": "affine", "scale": 1.0, "bias": 5.0},
        ],
    }
    seq = ComponentFactory.get().create(spec)
    clone = ComponentFactory.get().create(seq.to_dict())   # typed children round-trip
    assert clone.forward(3.0) == seq.forward(3.0)
