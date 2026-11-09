"""End-to-end test of the C++-backed components.

Run after building the extension (`python setup.py build_ext --inplace`):
    python -m pytest examples/bausatz/test_affine.py
"""

from __future__ import annotations

# The built ``affine`` package (with its compiled ``_affine`` extension) imports directly because
# pytest's default (prepend) import mode puts this directory on sys.path when collecting.
import pytest
from affine import (
    Affine,
    AffineKernel,
    NativeComponent,
    NativeSequential,
    Transform,
)

from bausatz import ComponentFactory


# --------------------------------------------------------------------------- #
# The C++ layer: library base, app compute interface, leaf kernel.
# --------------------------------------------------------------------------- #
def test_kernel_ctor_takes_config_fields_and_name_as_kwargs():
    # The constructor takes the config fields (and name) directly — this is what lets CppComponent
    # build it generically by splatting the validated config.
    k = AffineKernel(scale=3.0, bias=1.0, name="k")
    assert k.forward(2.0) == 7.0
    assert repr(k) == "AffineKernel(name='k')"      # default __repr__ from bind_component
    assert repr(AffineKernel(scale=1.0)) == "AffineKernel()"


def test_forward_lives_on_the_app_interface_not_the_library_base():
    # bausatz::Component (the library base) carries no compute method; the example's Transform does.
    assert issubclass(Transform, NativeComponent)
    assert hasattr(Transform, "forward")
    assert not hasattr(NativeComponent, "forward")


def test_kernel_is_a_transform_and_a_library_component():
    k = AffineKernel(scale=2.0, bias=0.5)
    assert isinstance(k, Transform)         # AffineKernel : Transform
    assert isinstance(k, NativeComponent)   # ... : bausatz::Component
    assert k.forward(10.0) == 20.5


def test_name_is_set_at_construction_and_immutable():
    assert AffineKernel(scale=1.0).name == ""             # default
    assert AffineKernel(scale=1.0, name="layer-0").name == "layer-0"
    with pytest.raises(AttributeError):
        AffineKernel(scale=1.0, name="layer-0").name = "changed"   # read-only, no setter


def test_zero_scale_cpp_exception_becomes_python_valueerror():
    with pytest.raises(ValueError, match="scale must be non-zero"):
        AffineKernel(scale=0.0)


# --------------------------------------------------------------------------- #
# Affine: a declarative CppComponent — no construction code, just `component=`.
# --------------------------------------------------------------------------- #
def test_factory_builds_affine_declaratively():
    comp = ComponentFactory.get().create(
        {"class_name": "affine", "name": "enc", "scale": 10.0, "bias": -1.0}
    )
    assert isinstance(comp, Affine)
    assert comp.forward(2.0) == 19.0                 # 10*2 - 1, computed in C++
    assert isinstance(comp.component, AffineKernel)
    assert comp.component is comp.component           # cached
    assert comp.component.name == "enc"              # config.name carried onto the native component
    # The bad-config path surfaces the C++ exception through the component too.
    with pytest.raises(ValueError, match="scale must be non-zero"):
        Affine(Affine.Config(scale=0.0)).component


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
    # The component is a single native composite (a Transform / bausatz::Component), not a Python loop.
    assert isinstance(seq.component, Transform)
    assert seq.component.size == 2
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


def test_native_sequential_rejects_a_none_stage():
    # pybind11 turns None into a null shared_ptr; the C++ ctor rejects it (ValueError), so
    # forward() can never dereference a null stage (it used to segfault).
    from affine._affine import Sequential

    with pytest.raises(ValueError, match=r"stages\[1\] is None"):
        Sequential([AffineKernel(scale=1.0), None])


def test_native_sequential_constructed_directly_builds_its_stages():
    # Not through the factory: first access to `component` builds the child stages first, so the
    # C++ Sequential gets them (rather than being cached with zero stages).
    seq = NativeSequential(
        NativeSequential.Config(
            stages=[{"class_name": "affine", "scale": 2.0}, {"class_name": "affine", "bias": 5.0}]
        )
    )
    assert seq.component.size == 2
    assert seq.forward(3.0) == 11.0


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
