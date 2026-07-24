"""CppComponent bridge: lazy engine construction from the validated config, delegation, and
factory integration. The 'engine' here is a pure-Python stand-in so the core test suite needs no
compiler — the compiled pybind11 counterpart lives in examples/pybind with its own test.
"""

from __future__ import annotations

from typing import Literal

import pytest

from bausatz import Component, ComponentFactory, CppComponent, Registry


class FakeKernel:
    """Stands in for a pybind11-exposed C++ class: constructed from plain scalars, does compute."""

    instances = 0

    def __init__(self, scale: float, bias: float) -> None:
        FakeKernel.instances += 1
        self.scale = scale
        self.bias = bias

    def apply(self, x: float) -> float:
        return self.scale * x + self.bias


class Affine(CppComponent):
    class Config(Component.Config):
        class_name: Literal["affine-stub"] = "affine-stub"
        scale: float = 1.0
        bias: float = 0.0

    config: Affine.Config

    def _make_engine(self) -> FakeKernel:
        return FakeKernel(self.config.scale, self.config.bias)

    def forward(self, x: float) -> float:
        return self.engine.apply(x)


@pytest.fixture
def factory() -> ComponentFactory:
    f = ComponentFactory(Registry())
    f.register("affine-stub", Affine)
    return f


def test_builds_from_spec_and_delegates_to_engine(factory):
    comp = factory.create({"class_name": "affine-stub", "scale": 3.0, "bias": 1.0})
    assert isinstance(comp, Affine)
    assert comp.forward(2.0) == 7.0  # 3*2 + 1, computed by the engine


def test_engine_is_built_lazily_once_and_cached():
    before = FakeKernel.instances
    comp = Affine(Affine.Config(scale=2.0, bias=0.0))
    assert FakeKernel.instances == before  # not built at construction
    engine = comp.engine
    assert FakeKernel.instances == before + 1  # built on first access
    assert comp.engine is engine  # cached, not rebuilt


def test_config_still_validates_and_round_trips(factory):
    # The pydantic Config keeps doing its job even though compute is "native".
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        factory.create({"class_name": "affine-stub", "scale": "not-a-number"})

    comp = factory.create({"class_name": "affine-stub", "scale": 5.0})
    assert comp.to_dict() == {"name": None, "class_name": "affine-stub", "scale": 5.0, "bias": 0.0}
    assert factory.create(comp.to_dict()).forward(1.0) == comp.forward(1.0)


def test_missing_make_engine_raises_helpfully():
    class NoEngine(CppComponent):
        class Config(Component.Config):
            class_name: Literal["no-engine"] = "no-engine"

    with pytest.raises(NotImplementedError, match="_make_engine"):
        _ = NoEngine(NoEngine.Config()).engine
